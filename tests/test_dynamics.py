"""
Test that JAX dynamics produce the same results as the original Numba dynamics.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_dynamics_numerical_match():
    import jax.numpy as jnp
    from f110_jax.dynamics import vehicle_dynamics_ks, vehicle_dynamics_st
    
    # CommonRoad full-scale test params (matching original DynamicsTest)
    mu = 1.0489; C_Sf = 21.92 / 1.0489; C_Sr = 21.92 / 1.0489
    lf = 0.3048 * 3.793293; lr = 0.3048 * 4.667707; h = 0.3048 * 2.01355
    m = 4.4482216152605 / 0.3048 * 74.91452; I = 4.4482216152605 * 0.3048 * 1321.416
    s_min = -1.066; s_max = 1.066; sv_min = -0.4; sv_max = 0.4
    v_min = -13.6; v_max = 50.8; v_switch = 7.319; a_max = 11.5

    f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.15, 5.1464424102339752, 0.2401426578627629]
    f_st_gt = [15.7213512030862397, 0.0925527979719355, 0.15, 5.3536773276413925, 0.0529001056654038, 0.6435589397748606, 0.0313297971641291]

    g = 9.81
    x_ks = jnp.array([3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566, 0.0294717351052816])
    x_st = jnp.array([2.0233348142065677, 0.0041907137716636, 0.0197545248559617, 15.7216236334290116, 0.0025857914776859, 0.0529001056654038, 0.0033012170610298])
    u = jnp.array([0.15, 0.63 * g])

    f_ks = np.array(vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max))
    f_st = np.array(vehicle_dynamics_st(x_st, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max))
    
    assert np.max(np.abs(np.array(f_ks_gt) - f_ks)) < 1e-6, f"KS mismatch: {np.max(np.abs(np.array(f_ks_gt) - f_ks))}"
    assert np.max(np.abs(np.array(f_st_gt) - f_st)) < 1e-6, f"ST mismatch: {np.max(np.abs(np.array(f_st_gt) - f_st))}"
    print("✅ Dynamics numerical match test passed")


def test_default_params():
    from f110_jax.simulator import F110JaxSimulator
    expected = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
    for k, v in expected.items():
        assert abs(F110JaxSimulator.DEFAULT_PARAMS[k] - v) < 1e-10, f"Param '{k}' mismatch"
    print("✅ Default params test passed")


if __name__ == '__main__':
    test_dynamics_numerical_match()
    test_default_params()
    print("🎉 ALL DYNAMICS TESTS PASSED")
