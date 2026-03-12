import jax
import jax.numpy as jnp

@jax.jit
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints
    """
    pos_limit = jnp.where(vel > v_switch, a_max * v_switch / vel, a_max)
    
    # accl limit reached?
    limit_cond1 = jnp.logical_and(vel <= v_min, accl <= 0.0)
    limit_cond2 = jnp.logical_and(vel >= v_max, accl >= 0.0)
    
    accl = jnp.where(jnp.logical_or(limit_cond1, limit_cond2), 0.0, accl)
    accl = jnp.clip(accl, -a_max, pos_limit)
    return accl

@jax.jit
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints
    """
    limit_cond1 = jnp.logical_and(steering_angle <= s_min, steering_velocity <= 0.0)
    limit_cond2 = jnp.logical_and(steering_angle >= s_max, steering_velocity >= 0.0)
    
    steering_velocity = jnp.where(jnp.logical_or(limit_cond1, limit_cond2), 0.0, steering_velocity)
    steering_velocity = jnp.clip(steering_velocity, sv_min, sv_max)
    return steering_velocity

@jax.jit
def vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics in JAX.
        x: [x, y, steering_angle, velocity, yaw_angle]
        u: [steering_velocity, longitudinal_acceleration]
    """
    lwb = lf + lr

    # constraints
    u0 = steering_constraint(x[2], u[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u[1], v_switch, a_max, v_min, v_max)

    # system dynamics
    f0 = x[3] * jnp.cos(x[4])
    f1 = x[3] * jnp.sin(x[4])
    f2 = u0
    f3 = u1
    f4 = x[3] / lwb * jnp.tan(x[2])
    
    return jnp.array([f0, f1, f2, f3, f4])

@jax.jit
def vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics in JAX.
        x: [x, y, steering_angle, velocity, yaw_angle, yaw_rate, slip_angle]
        u: [steering_velocity, longitudinal_acceleration]
    """
    g = 9.81
    lwb = lf + lr
    
    # constraints
    u0 = steering_constraint(x[2], u[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u[1], v_switch, a_max, v_min, v_max)

    # Kinematic part (fallback for low vel)
    f_ks = vehicle_dynamics_ks(x[:5], jnp.array([u0, u1]), mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    f_ks_st = jnp.array([
        f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4],
        u1 / lwb * jnp.tan(x[2]) + x[3] / (lwb * jnp.cos(x[2])**2) * u0,
        0.0  # slip angle dynamics zeroed out at low speeds
    ])
    
    # Dynamic part
    dyn_f0 = x[3] * jnp.cos(x[6] + x[4])
    dyn_f1 = x[3] * jnp.sin(x[6] + x[4])
    dyn_f2 = u0
    dyn_f3 = u1
    dyn_f4 = x[5]
    
    # Avoiding div by zero by masking low velocities explicitly inside calculations in dynamic regime
    vel_safe = jnp.where(jnp.abs(x[3]) < 1e-4, 1e-4, x[3])
    
    term1_dyn_f5 = -mu * m / (vel_safe * I * (lr + lf)) * (lf**2 * C_Sf * (g * lr - u1 * h) + lr**2 * C_Sr * (g * lf + u1 * h)) * x[5]
    term2_dyn_f5 = mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u1 * h) - lf * C_Sf * (g * lr - u1 * h)) * x[6]
    term3_dyn_f5 = mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u1 * h) * x[2]
    dyn_f5 = term1_dyn_f5 + term2_dyn_f5 + term3_dyn_f5
             
    term1_dyn_f6 = (mu / (vel_safe**2 * (lr + lf)) * (C_Sr * (g * lf + u1 * h) * lr - C_Sf * (g * lr - u1 * h) * lf) - 1.0) * x[5]
    term2_dyn_f6 = -mu / (vel_safe * (lr + lf)) * (C_Sr * (g * lf + u1 * h) + C_Sf * (g * lr - u1 * h)) * x[6]
    term3_dyn_f6 = mu / (vel_safe * (lr + lf)) * (C_Sf * (g * lr - u1 * h)) * x[2]
    dyn_f6 = term1_dyn_f6 + term2_dyn_f6 + term3_dyn_f6
             
    f_st_dyn = jnp.array([dyn_f0, dyn_f1, dyn_f2, dyn_f3, dyn_f4, dyn_f5, dyn_f6])

    # Switch to kinematic model for small velocities
    f = jnp.where(jnp.abs(x[3]) < 0.5, f_ks_st, f_st_dyn)
    
    return f

@jax.jit
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic PID controller: desired speed/steer -> acceleration/steering velocity.
    Translated from the original Numba version to JAX.
    
    Args:
        speed: desired speed
        steer: desired steering angle
        current_speed: current velocity from state
        current_steer: current steering angle from state
        max_sv: maximum steering velocity
        max_a: maximum acceleration
        max_v: maximum velocity
        min_v: minimum velocity
    Returns:
        accl: acceleration command
        sv: steering velocity command
    """
    # steering velocity
    steer_diff = steer - current_steer
    sv = jnp.where(
        jnp.abs(steer_diff) > 1e-4,
        jnp.sign(steer_diff) * max_sv,
        0.0
    )
    
    # acceleration
    vel_diff = speed - current_speed
    
    # Forward: accelerating
    kp_fwd_acc = 10.0 * max_a / max_v
    # Forward: braking
    kp_fwd_brk = 10.0 * max_a / (-min_v)
    # Backward: braking (positive vel_diff)
    kp_bwd_brk = 2.0 * max_a / max_v
    # Backward: accelerating (negative vel_diff)
    kp_bwd_acc = 2.0 * max_a / (-min_v)
    
    # Select kp based on current_speed direction and vel_diff sign
    kp = jnp.where(
        current_speed > 0.0,
        jnp.where(vel_diff > 0, kp_fwd_acc, kp_fwd_brk),
        jnp.where(vel_diff > 0, kp_bwd_brk, kp_bwd_acc)
    )
    accl = kp * vel_diff
    
    return accl, sv
