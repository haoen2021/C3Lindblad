"""Small assembly checks for recipes kept outside the installed package."""

import numpy as np

from examples.shared.bath import (
    CoupledPseudomodeFit,
    coupled_pseudomode_dimer_model,
    coupled_pseudomode_model,
)
from c3l import FermionicLindbladModel




def test_saved_fit_recipes_preserve_models():
    fit = CoupledPseudomodeFit(
        N=1, H_lesser=np.array([[-0.7]]), Gamma_lesser=np.array([[0.2]]),
        g_lesser=np.array([0.3j]), H_greater=np.array([[0.6]]),
        Gamma_greater=np.array([[0.4]]), g_greater=np.array([0.5]),

    )
    single = coupled_pseudomode_model(fit, U=2.0)
    dimer = coupled_pseudomode_dimer_model(fit, U=2.0, t12=0.3)
    assert isinstance(single, FermionicLindbladModel)
    assert isinstance(dimer, FermionicLindbladModel)
    np.testing.assert_allclose(np.diag(single.E), [-1.0, -0.7, 0.6])
    np.testing.assert_allclose(dimer.E[:2, :2], [[-1.0, -0.3], [-0.3, -1.0]])
    np.testing.assert_allclose(np.diag(dimer.E)[2:], [-0.7, 0.6, -0.7, 0.6])
    np.testing.assert_array_equal(np.diag(dimer.gamma_gain), [0, 0, 0.2, 0, 0.2, 0])
    np.testing.assert_array_equal(np.diag(dimer.gamma_loss), [0, 0, 0, 0.4, 0, 0.4])
    assert dimer.interaction_sites == (0, 1)
