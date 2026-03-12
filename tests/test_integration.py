"""
Integration test: full simulator reset/step with map loading.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _get_map_path():
    """Resolve the Berlin map path from the sibling f1tenth_gym repo."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '../../f1tenth_gym/gym/f110_gym/envs/maps/berlin.yaml'),
        os.path.join(os.path.dirname(__file__), '../../f1tenth_gym/examples/example_map.yaml'),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find any map yaml. Place f1tenth_gym next to f1tenth_gym_jax.")


def test_steering_delay():
    from f110_jax.simulator import F110JaxSimulator, Integrator
    sim = F110JaxSimulator(_get_map_path(), '.png', num_agents=1, integrator=Integrator.RK4)
    sim.reset(np.array([[0.0, 0.0, 0.0]]))
    
    d = sim._apply_steering_delay(np.array([0.5]))
    assert d[0] == 0.0, f"Step1: expected 0, got {d[0]}"
    d = sim._apply_steering_delay(np.array([0.5]))
    assert d[0] == 0.0, f"Step2: expected 0, got {d[0]}"
    d = sim._apply_steering_delay(np.array([0.3]))
    assert d[0] == 0.5, f"Step3: expected 0.5, got {d[0]}"
    print("✅ Steering delay test passed")


def test_reset_step():
    from f110_jax.simulator import F110JaxSimulator, Integrator
    sim = F110JaxSimulator(_get_map_path(), '.png', num_agents=2, integrator=Integrator.RK4)
    
    obs, reward, done, info = sim.reset(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    
    required = ['ego_idx', 'scans', 'poses_x', 'poses_y', 'poses_theta',
                'linear_vels_x', 'linear_vels_y', 'ang_vels_z', 'collisions', 'lap_times', 'lap_counts']
    for k in required:
        assert k in obs, f"Missing key: {k}"
    assert obs['ego_idx'] == 0
    assert len(obs['scans']) == 2
    assert reward == 0.01
    assert 'checkpoint_done' in info
    
    for _ in range(50):
        obs, reward, done, info = sim.step(np.array([[0.1, 3.0], [0.0, 2.0]]))
    
    assert obs['poses_x'][0] != 0.0 or obs['poses_y'][0] != 0.0
    print("✅ Reset/step test passed")


def test_euler():
    from f110_jax.simulator import F110JaxSimulator, Integrator
    sim = F110JaxSimulator(_get_map_path(), '.png', num_agents=1, integrator=Integrator.Euler)
    sim.reset(np.array([[0.0, 0.0, 0.0]]))
    for _ in range(10):
        obs, _, _, _ = sim.step(np.array([[0.0, 2.0]]))
    assert obs['linear_vels_x'][0] != 0.0
    print("✅ Euler integrator test passed")


if __name__ == '__main__':
    test_steering_delay()
    test_reset_step()
    test_euler()
    print("🎉 ALL INTEGRATION TESTS PASSED")
