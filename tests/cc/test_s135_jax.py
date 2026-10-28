"""Deterministic gates for the optional full-Q5 JAX action."""

from __future__ import annotations

import importlib
import os
from dataclasses import replace

import numpy as np
import pytest

from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    ComponentStaticProblem,
    component_eom_context,
    component_block_action,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.s135_full import CompiledFullS135Action
from c3l.cc.s135_full import FullS135ResponseOperator
from tests.s135_fixture import s135_validation_model


@pytest.fixture(scope="module")
def small_full_operator() -> FullS135ResponseOperator:
    cc = factorized_cc_problem(s135_validation_model(), "CCSD", m_up=0.3)
    static = ComponentStaticProblem.compile(cc)
    coordinate = np.arange(cc.num_variables, dtype=np.float64)
    amplitudes = cc.tilde.unpack(6.0e-4 * np.cos(coordinate + 0.19))
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, 1)
    context = component_eom_context(problem, amplitudes)
    compiled = CompiledFullS135Action(
        context,
        materialization_batch_size=4,
    )
    return FullS135ResponseOperator(compiled, s5_tile_size=7)


def test_eom_plan_construction_does_not_hash_inputs(small_full_operator, monkeypatch):
    import hashlib
    from c3l.cc.component_jax import _retained_plan
    from c3l.cc.s135_jax import build_full_s135_jax_plan

    def forbid_hash(*args, **kwargs):
        raise AssertionError("kernel setup must not compute checkpoint identities")

    monkeypatch.setattr(hashlib, "sha256", forbid_hash)
    full = build_full_s135_jax_plan(small_full_operator)
    retained = _retained_plan(small_full_operator.action.context)
    assert full.dimension == small_full_operator.dimension
    assert retained.dimension == small_full_operator.p_dimension


def test_checkpoint_support_owns_the_eom_identity(small_full_operator):
    from c3l.green import checkpointed_arnoldi

    identity = getattr(checkpointed_arnoldi, "eom_checkpoint_identity", None)
    assert callable(identity)
    context = small_full_operator.action.context
    assert len(identity(context)) == 64


def test_plan_preserves_dimensions_keys_and_owned_bytes(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """Dropping a response block or compiled tensor must change/fail the plan."""

    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    plan = s135_jax.build_full_s135_jax_plan(small_full_operator)

    assert (plan.p_dimension, plan.s5_dimension, plan.dimension) == (
        small_full_operator.p_dimension,
        small_full_operator.s5_dimension,
        small_full_operator.dimension,
    )
    assert plan.coordinate_keys == tuple(
        block.key for block in small_full_operator.coordinates.blocks
    )
    assert plan.creation_bytes == small_full_operator.action.storage_bytes
    assert not hasattr(plan, "fingerprint")


def test_problem_identity_uses_compact_inputs(small_full_operator, monkeypatch):
    from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity

    context = small_full_operator.action.context
    key = eom_checkpoint_identity(context)
    assert len(key) == 64
    blocks = {key: values.copy() for key, values in context.amplitudes.blocks.items()}
    next(iter(blocks.values())).flat[0] += 0.01
    changed = replace(context, amplitudes=replace(context.amplitudes, blocks=blocks))
    assert eom_checkpoint_identity(changed) != key

    oracle = context.problem.static.oracle
    product = oracle.products[0]
    for changed_oracle in (
        replace(oracle, rapidities=oracle.rapidities + 0.01),
        replace(oracle, products=(replace(product, coefficient=2 * product.coefficient), *oracle.products[1:])),
    ):
        static = replace(context.problem.static, oracle=changed_oracle)
        changed = replace(context, problem=replace(context.problem, static=static))
        assert eom_checkpoint_identity(changed) != key

    shell = context.problem.s5_coordinates
    changed = replace(context, problem=replace(context.problem, s5_coordinates=replace(shell, charge=-shell.charge)))
    assert eom_checkpoint_identity(changed) != key

    def forbid_materialization(*args, **kwargs):
        raise AssertionError("identity checking must not materialize the Q5 shell")

    monkeypatch.setattr(type(shell), "subset", forbid_materialization)
    assert eom_checkpoint_identity(context) == key


def test_problem_identity_ignores_labels_and_tile_size(small_full_operator):
    from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity

    context = small_full_operator.action.context
    oracle = context.problem.static.oracle
    products = tuple(replace(record, label="different-label") for record in oracle.products)
    static = replace(context.problem.static, oracle=replace(oracle, products=products))
    changed = replace(context, problem=replace(context.problem, static=static))
    assert eom_checkpoint_identity(changed) == eom_checkpoint_identity(context)
    other = FullS135ResponseOperator(small_full_operator.action, s5_tile_size=3)
    assert eom_checkpoint_identity(other.action.context) == eom_checkpoint_identity(context)


def test_runtime_rejects_disabled_x64_and_non_gpu_production() -> None:
    """A production run must not silently demote complex128 or use the CPU."""

    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    disabled = s135_jax.JaxRuntime(
        jax_version="0.9.2",
        jaxlib_version="0.9.2",
        enable_x64=False,
        platform="gpu",
        device_kind="NVIDIA A100-SXM4-40GB",
    )
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        s135_jax.require_jax_runtime(disabled, production=True)

    cpu = s135_jax.JaxRuntime(
        jax_version="0.9.2",
        jaxlib_version="0.9.2",
        enable_x64=True,
        platform="cpu",
        device_kind="cpu",
    )
    with pytest.raises(RuntimeError, match="GPU"):
        s135_jax.require_jax_runtime(cpu, production=True)


def test_jax_unpack_reconstructs_literal_group_antisymmetry() -> None:
    """A missing permutation or wrong fermionic sign must fail this fixture."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    block = s135_jax.CoordinateBlockPlan(
        key=(2, 1, 0, 0),
        vector_positions=np.asarray((0, 1), dtype=np.int64),
        indices=np.asarray(((0, 2, 1), (1, 2, 0)), dtype=np.int64),
        permutations=(((0, 1, 2), 1), ((1, 0, 2), -1)),
    )
    values = np.asarray((2.0 + 3.0j, -1.0 + 0.5j), dtype=np.complex128)
    expected = np.zeros((3, 3, 3), dtype=np.complex128)
    expected[0, 2, 1] = values[0]
    expected[2, 0, 1] = -values[0]
    expected[1, 2, 0] = values[1]
    expected[2, 1, 0] = -values[1]

    observed = s135_jax._unpack_block(jnp, jnp.asarray(values), block, 3)

    np.testing.assert_array_equal(np.asarray(observed), expected)


def test_precomputed_gather_unpack_equals_literal_antisymmetric_scatter() -> None:
    """The fast unpack map must preserve zeros, positions, and fermionic signs."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    block = s135_jax.CoordinateBlockPlan(
        key=(2, 1, 0, 0),
        vector_positions=np.asarray((3, 7), dtype=np.int64),
        indices=np.asarray(((0, 2, 1), (1, 2, 0)), dtype=np.int64),
        permutations=(((0, 1, 2), 1), ((1, 0, 2), -1)),
    )
    value = np.arange(10, dtype=np.float64).astype(np.complex128)
    value += 0.25j * value
    lookup, signs = s135_jax._coordinate_gather_maps(block, 3)

    observed = s135_jax._unpack_from_gather(
        jnp, jnp.asarray(value), jnp.asarray(lookup), jnp.asarray(signs)
    )
    expected = s135_jax._unpack_block(jnp, jnp.asarray(value), block, 3)

    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_precomputed_gather_assembly_restores_global_coordinate_order() -> None:
    """Replacing indexed updates must not assume blocks are already contiguous."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    positions = (
        np.asarray((2, 0), dtype=np.int64),
        np.asarray((3, 1), dtype=np.int64),
    )
    lookup = s135_jax._assembly_gather_lookup(positions, 4)
    blocks = (
        jnp.asarray((20.0 + 2.0j, 10.0 + 1.0j)),
        jnp.asarray((40.0 + 4.0j, 30.0 + 3.0j)),
    )

    observed = s135_jax._assemble_from_gather(
        jnp,
        blocks,
        jnp.asarray(lookup),
        jnp.zeros(4, dtype=jnp.complex128),
        jnp.zeros(4, dtype=jnp.complex128),
    )

    np.testing.assert_array_equal(
        np.asarray(observed),
        np.asarray((10.0 + 1.0j, 30.0 + 3.0j, 20.0 + 2.0j, 40.0 + 4.0j)),
    )


def test_jax_interior_uses_bilinear_contraction_without_conjugation() -> None:
    """Conjugating the annihilator or contracting the wrong axis must fail."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    vector = np.asarray((1.0 + 2.0j, -0.5 + 0.25j), dtype=np.complex128)
    tensor = np.asarray(
        ((2.0 - 1.0j, 0.5 + 3.0j), (-1.5 + 0.2j, 4.0 - 0.5j)),
        dtype=np.complex128,
    )
    expected = np.tensordot(vector, tensor, axes=(0, 0))

    observed = s135_jax._interior_blocks(
        jnp,
        {0: jnp.asarray(vector)},
        {(1, 1, 0, 0): jnp.asarray(tensor)},
    )

    np.testing.assert_allclose(
        np.asarray(observed[(0, 1, 0, 0)]), expected, atol=0.0, rtol=0.0
    )


def test_jax_wedge_values_obeys_literal_antisymmetric_sign() -> None:
    """Replacing the exterior wedge with pointwise multiplication must fail."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    left = np.asarray((1.0 + 0.5j, -2.0j, 3.0 - 1.0j), dtype=np.complex128)
    right = np.asarray((-0.5j, 2.0 + 1.0j, -1.0 + 4.0j), dtype=np.complex128)
    coordinates = np.asarray(((0, 2), (1, 2)), dtype=np.int64)
    expected = np.asarray(
        (
            left[0] * right[2] - left[2] * right[0],
            left[1] * right[2] - left[2] * right[1],
        ),
        dtype=np.complex128,
    )

    observed = s135_jax._wedge_values(
        jnp,
        jnp.asarray(left),
        (1, 0, 0, 0),
        jnp.asarray(right),
        (1, 0, 0, 0),
        jnp.asarray(coordinates),
    )

    np.testing.assert_allclose(np.asarray(observed), expected, atol=0.0, rtol=0.0)


def test_jax_action_matches_all_four_explicit_blocks(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """Dropping APP, AP5, A5P, A55, or either diagonal block must fail."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    exact = component_block_action(
        small_full_operator.action.context,
        small_full_operator.coordinates,
        small_full_operator.coordinates,
        batch_size=11,
    )
    rng = np.random.default_rng(20260722)
    x = rng.normal(size=small_full_operator.dimension) + 1j * rng.normal(
        size=small_full_operator.dimension
    )
    p = small_full_operator.p_dimension
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )

    observed = accelerated.block_action(x)

    np.testing.assert_allclose(observed.APP, exact[:p, :p] @ x[:p], atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(observed.AP5, exact[:p, p:] @ x[p:], atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(observed.A5P, exact[p:, :p] @ x[:p], atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(observed.A55, exact[p:, p:] @ x[p:], atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(
        accelerated.matvec(x), exact @ x, atol=3e-11, rtol=3e-11
    )


def test_jax_right_action_preserves_plain_bilinear_transpose(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """Conjugating GPU coefficients must violate the CPU plain transpose gate."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    rng = np.random.default_rng(20260723)
    right = rng.normal(size=small_full_operator.dimension) + 1j * rng.normal(
        size=small_full_operator.dimension
    )
    left = rng.normal(size=small_full_operator.dimension) + 1j * rng.normal(
        size=small_full_operator.dimension
    )
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )

    exact = component_block_action(
        small_full_operator.action.context,
        small_full_operator.coordinates,
        small_full_operator.coordinates,
        batch_size=11,
    )
    lhs = left @ accelerated.matvec(right)
    rhs = (exact.T @ left) @ right

    np.testing.assert_allclose(lhs, rhs, atol=3e-11, rtol=3e-11)


def test_numpy_matvec_result_is_writable_for_in_place_arnoldi(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """The matrix-free API must return an independent mutable host work vector."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )
    vector = np.ones(accelerated.dimension, dtype=np.complex128)

    observed = accelerated.matvec(vector)

    assert observed.flags.writeable
    observed -= 0.5 * observed


def test_plan_batches_every_acted_rank_and_target_block(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """A scalar/vector-only lowering must fail on the exact Q5-input paths."""

    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    plan = s135_jax.build_full_s135_jax_plan(small_full_operator)

    assert plan.acted_degrees == (0, 1, 2, 3, 4, 5)
    assert len(plan.prefix_buckets) < len(plan.prefixes)
    assert {kernel.target_key for kernel in plan.projection_kernels} == set(
        plan.coordinate_keys
    )
    assert all(kernel.terms for kernel in plan.projection_kernels)
    assert 0 < plan.projected_creation_bytes <= plan.creation_bytes


def test_batched_wedge_reduces_creation_and_acted_contributions() -> None:
    """Mixing acted rows or omitting the contribution reduction must fail."""

    jnp = pytest.importorskip("jax.numpy")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    creation = np.asarray(
        ((1.0 + 0.5j, 2.0 - 1.0j, -0.5j), (-1.0 + 2.0j, 0.25j, 3.0)),
        dtype=np.complex128,
    )
    acted = np.asarray(
        (
            (0.5 - 1.0j, 2.0j, 1.5),
            (4.0, -0.5j, 2.0 + 3.0j),
            (-2.0j, 1.0 + 0.25j, -1.0),
        ),
        dtype=np.complex128,
    )
    rows = np.asarray((2, 0), dtype=np.int64)
    coordinates = np.asarray(((0, 2), (1, 2)), dtype=np.int64)
    expected = np.zeros(2, dtype=np.complex128)
    for term, row in enumerate(rows):
        for position, (left_index, right_index) in enumerate(coordinates):
            expected[position] += (
                creation[term, left_index] * acted[row, right_index]
                - creation[term, right_index] * acted[row, left_index]
            )

    observed = s135_jax._batched_wedge_values(
        jnp,
        jnp.asarray(creation),
        (1, 0, 0, 0),
        jnp.asarray(acted),
        jnp.asarray(rows),
        (1, 0, 0, 0),
        jnp.asarray(coordinates),
    )

    np.testing.assert_allclose(np.asarray(observed), expected, atol=0.0, rtol=0.0)


def test_operator_uses_shape_buckets_instead_of_one_launch_per_term(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """Regressing to thousands of eager term launches must fail."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )

    assert accelerated.kernel_counts == {
        "prefix": len(accelerated.plan.prefix_buckets),
        "projection": len(accelerated.plan.coordinates),
    }


def test_profile_separates_cold_kernel_transfers_and_end_to_end(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """Reporting dispatch time as kernel time or hiding transfers must fail."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )
    rng = np.random.default_rng(20260724)
    vector = rng.normal(size=accelerated.dimension) + 1j * rng.normal(
        size=accelerated.dimension
    )

    profile = accelerated.profile(vector, warm_repeats=2)

    assert profile.first_device_action_seconds > 0.0
    assert len(profile.warm_kernel_seconds) == 2
    assert len(profile.h2d_seconds) == 2
    assert len(profile.d2h_seconds) == 2
    assert len(profile.end_to_end_seconds) == 2
    assert all(value >= 0.0 for value in profile.warm_kernel_seconds)
    assert profile.plan_build_seconds >= 0.0
    assert profile.device_upload_seconds >= 0.0
    assert isinstance(profile.device_memory, dict)


def test_stage_profile_identifies_prefix_and_projection_boundaries(
    small_full_operator: FullS135ResponseOperator,
) -> None:
    """A slow full action must be attributable to one logical device stage."""

    pytest.importorskip("jax")
    s135_jax = importlib.import_module("c3l.cc.s135_jax")
    accelerated = s135_jax.JaxFullS135ResponseOperator.from_cpu(
        small_full_operator,
        platform=os.environ.get("C3L_TEST_JAX_PLATFORM", "cpu"),
        production=False,
    )
    vector = np.linspace(0.1, 1.0, accelerated.dimension).astype(np.complex128)

    observed, stages = accelerated.profile_stages(vector)

    np.testing.assert_allclose(
        observed, small_full_operator.matvec(vector), atol=3e-11, rtol=3e-11
    )
    assert len(stages["prefix_buckets"]) == accelerated.kernel_counts["prefix"]
    assert len(stages["projection_targets"]) == accelerated.kernel_counts[
        "projection"
    ]
    assert {
        tuple(item["target_key"]) for item in stages["projection_targets"]
    } == set(accelerated.plan.coordinate_keys)
    assert all(item["wall_seconds"] >= 0.0 for item in stages["prefix_buckets"])
