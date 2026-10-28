import pytest

from c3l.green.arnoldi_memory import select_basis_cache


@pytest.mark.parametrize("budget,columns,mode", [(1056, 6, "resident_capacity"), (704, 4, "streaming"), (703, 2, "streaming"), (0, 0, "streaming")])
def test_cache_budget_rounding(budget, columns, mode):
    policy = select_basis_cache(vector_dimension=11, maximum_columns=5, panel_columns=2, budget_bytes=budget)
    assert policy.cache_columns == columns
    assert policy.required_bytes == 1056
    assert policy.budget_bytes == budget
    assert policy.mode == mode


@pytest.mark.parametrize("name,value", [("vector_dimension", 0), ("maximum_columns", -1), ("panel_columns", True), ("budget_bytes", -1), ("budget_bytes", 1.5)])
def test_cache_budget_rejects_invalid_values(name, value):
    arguments = dict(vector_dimension=11, maximum_columns=5, panel_columns=2, budget_bytes=1056)
    arguments[name] = value
    with pytest.raises(ValueError):
        select_basis_cache(**arguments)
