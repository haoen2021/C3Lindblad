"""Restore the normal-mode coordinates of a validated CC checkpoint."""

from pathlib import Path
from dataclasses import replace
import numpy as np


def restore_modes(fresh, saved):
    # This repair covers roundoff only, never mode permutations or rotations.
    np.testing.assert_allclose(saved, fresh.V, atol=1e-10, rtol=1e-10)
    inverse = np.linalg.inv(saved)
    condition = float(np.linalg.cond(saved))
    eigen_error = np.linalg.norm(fresh.h @ saved - saved * fresh.eigenvalues)
    eigen_error /= max(np.linalg.norm(fresh.h) * np.linalg.norm(saved), 1e-13)
    # The core constructor checks eigen equations, inverse, normalization,
    # phases, vacuum, rapidities, conditioning and ordering again.
    return replace(
        fresh,
        V=saved,
        V_inv=inverse,
        eigen_residual=float(eigen_error),
        inverse_residual=float(
            np.linalg.norm(inverse @ saved - np.eye(len(saved))) / np.sqrt(len(saved))
        ),
        condition_number=condition,
        condition_roundoff=condition * np.finfo(float).eps,
    )


def saved_reference(node, fock, gamma, checkpoint=None):
    from c3l.ci.reference import full_hartree_fock_reference

    fresh = full_hartree_fock_reference(node, fock, gamma)
    if checkpoint is None:
        raise ValueError("An explicit static checkpoint is required")
    checkpoint = Path(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as saved:
        up = restore_modes(fresh.up, saved["reference_V_up"])
        down = restore_modes(fresh.down, saved["reference_V_down"])
    return replace(fresh, up=up, down=down)


def build_problem(
    physical, fock, gamma, scale=1.0, reference_scale=None, *, checkpoint=None
):
    from c3l.models import homotopy_model
    from c3l.cc.factorized import factorized_cc_problem
    from c3l.cc.component_tensor import ComponentStaticProblem

    node = (
        physical
        if scale == 1 and reference_scale is None
        else homotopy_model(physical, fock[::2, ::2], scale)
    )
    reference = saved_reference(node, fock, gamma, checkpoint)
    problem = factorized_cc_problem(
        node, "CCSD", reference=reference, build_next_rank_masks=False
    )
    return problem, ComponentStaticProblem.compile(problem, fuse_products=True)
