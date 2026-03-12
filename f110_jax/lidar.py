import jax
import jax.numpy as jnp

@jax.jit
def get_dt(bitmap, resolution):
    """
    Since scipy.ndimage edt cannot be jitted, in a pure JAX implementation
    we expect the user/environment to pre-compute the distance transform
    using scipy and pass it to the JAX simulator as a static array (or device array).
    """
    pass

@jax.jit(static_argnums=(3, 15))
def get_scan(pose, theta_dis, fov, num_beams, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    """
    Perform 2D LiDAR ray casting in parallel using JAX vmap.
    This replaces the while-loop based ray tracing with a vectorized batched approach.
    For true GPU acceleration, we use a fixed max number of steps for ray marching to keep it JIT-compatible.
    """
    
    # Calculate ray angles globally based on vehicle heading
    angle_increment = fov / (num_beams - 1)
    beam_angles_relative = jnp.linspace(-fov/2.0, fov/2.0, num_beams)
    beam_angles_world = pose[2] + beam_angles_relative
    
    ray_sines = jnp.sin(beam_angles_world)
    ray_cosines = jnp.cos(beam_angles_world)

    MAX_STEPS = 1000 # Upper bound iteration limit for JAX while loop

    def trace_single_ray(c, s):
        def cond_fun(val):
            dist, current_x, current_y, step_idx = val
            
            x_trans = current_x - orig_x
            y_trans = current_y - orig_y
            x_rot = x_trans * orig_c + y_trans * orig_s
            y_rot = -x_trans * orig_s + y_trans * orig_c
            
            c_grid = jnp.floor(x_rot / resolution).astype(jnp.int32)
            r_grid = jnp.floor(y_rot / resolution).astype(jnp.int32)
            
            in_bounds = (c_grid >= 0) & (c_grid < width) & (r_grid >= 0) & (r_grid < height)
            
            c_safe = jnp.clip(c_grid, 0, width - 1)
            r_safe = jnp.clip(r_grid, 0, height - 1)
            dist_to_nearest = jnp.where(in_bounds, dt[r_safe, c_safe], 0.0)
            
            not_hit = dist_to_nearest > eps
            not_too_far = dist <= max_range
            not_too_many_steps = step_idx < MAX_STEPS
            return not_hit & not_too_far & not_too_many_steps & in_bounds

        def body_fun(val):
            dist, current_x, current_y, step_idx = val
            
            x_trans = current_x - orig_x
            y_trans = current_y - orig_y
            x_rot = x_trans * orig_c + y_trans * orig_s
            y_rot = -x_trans * orig_s + y_trans * orig_c
            
            c_grid = jnp.floor(x_rot / resolution).astype(jnp.int32)
            r_grid = jnp.floor(y_rot / resolution).astype(jnp.int32)
            
            c_safe = jnp.clip(c_grid, 0, width - 1)
            r_safe = jnp.clip(r_grid, 0, height - 1)
            dist_to_nearest = dt[r_safe, c_safe]
            
            next_x = current_x + dist_to_nearest * c
            next_y = current_y + dist_to_nearest * s
            next_dist = dist + dist_to_nearest
            
            return (next_dist, next_x, next_y, step_idx + 1)

        init_val = (0.0, pose[0], pose[1], 0)
        final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
        
        final_dist = final_val[0]
        return jnp.clip(final_dist, 0.0, max_range)

    # Vectorize computation across all rays simultaneously using vmap
    scan = jax.vmap(trace_single_ray)(ray_cosines, ray_sines)
    
    return scan


# ---- Other-agent ray casting (matching original f110_gym laser_models.py) ----

@jax.jit
def cross_2d(v1, v2):
    """Cross product of two 2-vectors."""
    return v1[0] * v2[1] - v1[1] * v2[0]

@jax.jit
def are_collinear(pt_a, pt_b, pt_c):
    """Checks if three points are collinear in 2D."""
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    return jnp.abs(cross_2d(ba, ca)) < tol

@jax.jit
def get_range(pose, beam_theta, va, vb):
    """
    Get the distance at a beam angle to the vector formed by two vertices of a vehicle.
    Matches original laser_models.get_range.
    
    Args:
        pose: [x, y, theta] of scanning vehicle
        beam_theta: angle of the current beam (world frame)
        va, vb: the two vertices forming an edge [2,]
    Returns:
        distance: smallest distance at beam theta from scanning pose to edge
    """
    o = pose[:2]
    v1 = o - va
    v2 = vb - va
    v3 = jnp.array([jnp.cos(beam_theta + jnp.pi / 2.0), jnp.sin(beam_theta + jnp.pi / 2.0)])
    
    denom = jnp.dot(v2, v3)
    
    # Main case: non-parallel
    d1 = cross_2d(v2, v1) / jnp.where(jnp.abs(denom) > 0.0, denom, 1.0)
    d2 = jnp.dot(v1, v3) / jnp.where(jnp.abs(denom) > 0.0, denom, 1.0)
    valid_main = (jnp.abs(denom) > 0.0) & (d1 >= 0.0) & (d2 >= 0.0) & (d2 <= 1.0)
    dist_main = jnp.where(valid_main, d1, jnp.inf)
    
    # Collinear case
    da = jnp.linalg.norm(va - o)
    db = jnp.linalg.norm(vb - o)
    collinear = are_collinear(o, va, vb) & (jnp.abs(denom) <= 0.0)
    dist_collinear = jnp.where(collinear, jnp.minimum(da, db), jnp.inf)
    
    return jnp.minimum(dist_main, dist_collinear)


@jax.jit
def get_blocked_view_indices(pose, vertices, scan_angles):
    """
    Get the indices of the start and end of blocked FOV in scans by another vehicle.
    Matches original laser_models.get_blocked_view_indices.
    
    Args:
        pose: [x, y, theta] of scanning vehicle
        vertices: [4, 2] four vertices of opponent vehicle
        scan_angles: [num_beams,] beam angles relative to vehicle heading
    Returns:
        min_ind, max_ind: start and end beam indices that are blocked
    """
    # Vectors from pose to all 4 vertices
    vecs = vertices - pose[:2]
    norms = jnp.linalg.norm(vecs, axis=1)
    unit_vecs = vecs / norms[:, None]
    
    # Angle of ego heading
    ego_angle = jnp.arctan2(jnp.sin(pose[2]), jnp.cos(pose[2]))
    
    # Angles of each vertex relative to ego heading
    angles_with_x = jnp.zeros(4)
    vertex_angles = jnp.arctan2(unit_vecs[:, 1], unit_vecs[:, 0])
    raw_angles = -(ego_angle - vertex_angles)
    
    # Wrap to [-pi, pi]
    wrapped = jnp.where(raw_angles > jnp.pi, raw_angles - 2 * jnp.pi, raw_angles)
    wrapped = jnp.where(wrapped < -jnp.pi, wrapped + 2 * jnp.pi, wrapped)
    
    # Find nearest beam index for each vertex angle
    ind1 = jnp.argmin(jnp.abs(scan_angles - wrapped[0]))
    ind2 = jnp.argmin(jnp.abs(scan_angles - wrapped[1]))
    ind3 = jnp.argmin(jnp.abs(scan_angles - wrapped[2]))
    ind4 = jnp.argmin(jnp.abs(scan_angles - wrapped[3]))
    
    all_inds = jnp.array([ind1, ind2, ind3, ind4])
    return jnp.min(all_inds), jnp.max(all_inds)


@jax.jit
def ray_cast_single_edge(pose, beam_theta, va, vb, current_range):
    """
    Check one beam against one edge, return minimum of current range and new range.
    """
    scan_range = get_range(pose, beam_theta, va, vb)
    return jnp.minimum(current_range, scan_range)


def ray_cast_agents(scan, pose, scan_angles, opp_vertices_list):
    """
    Ray cast onto other agents in the env, modify original scan.
    This is the JAX equivalent of RaceCar.ray_cast_agents().
    
    Args:
        scan: [num_beams,] original scan range array
        pose: [x, y, theta] of scanning vehicle
        scan_angles: [num_beams,] relative beam angles
        opp_vertices_list: [num_opponents, 4, 2] vertices of each opponent
    Returns:
        new_scan: [num_beams,] modified scan
    """
    new_scan = scan
    
    # Process each opponent
    for opp_idx in range(opp_vertices_list.shape[0]):
        opp_vertices = opp_vertices_list[opp_idx]
        
        # Get blocked view range
        min_ind, max_ind = get_blocked_view_indices(pose, opp_vertices, scan_angles)
        min_ind = int(min_ind)
        max_ind = int(max_ind)
        
        # Create looped vertices (5 vertices, last = first)
        looped_verts = jnp.vstack([opp_vertices, opp_vertices[0:1]])
        
        # For each beam in the blocked range, check against all 4 edges
        for i in range(min_ind, max_ind + 1):
            beam_theta = pose[2] + scan_angles[i]
            for j in range(4):
                new_scan = new_scan.at[i].set(
                    ray_cast_single_edge(pose, beam_theta, looped_verts[j], looped_verts[j + 1], new_scan[i])
                )
    
    return new_scan
