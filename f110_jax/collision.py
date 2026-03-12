import jax
import jax.numpy as jnp

@jax.jit
def get_trmtx(pose):
    """
    Get transformation matrix of vehicle frame -> global frame
    Args:
        pose: [x, y, theta]
    Returns:
        H: [4, 4] transformation matrix
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = jnp.cos(th)
    sin = jnp.sin(th)
    
    H = jnp.array([
        [cos, -sin, 0., x],
        [sin,  cos, 0., y],
        [0.,   0.,  1., 0.],
        [0.,   0.,  0., 1.]
    ])
    return H

@jax.jit
def get_vertices(pose, length, width):
    """
    Return vertices of the car body given pose and size.
    Args:
        pose: [x, y, theta]
        length: car length
        width: car width
    Returns:
        vertices: [4, 2] points (rl, rr, fr, fl)
    """
    H = get_trmtx(pose)
    
    rl = jnp.dot(H, jnp.array([-length/2, width/2,  0., 1.]))
    rr = jnp.dot(H, jnp.array([-length/2, -width/2, 0., 1.]))
    fl = jnp.dot(H, jnp.array([length/2,  width/2,  0., 1.]))
    fr = jnp.dot(H, jnp.array([length/2,  -width/2, 0., 1.]))
    
    rl = rl[:2] / rl[3]
    rr = rr[:2] / rr[3]
    fl = fl[:2] / fl[3]
    fr = fr[:2] / fr[3]
    
    vertices = jnp.vstack([rl, rr, fr, fl])
    return vertices

@jax.jit
def perpendicular(pt):
    return jnp.array([pt[1], -pt[0]])

@jax.jit
def tripleProduct(a, b, c):
    ac = jnp.dot(a, c)
    bc = jnp.dot(b, c)
    return b*ac - a*bc

@jax.jit
def avgPoint(vertices):
    return jnp.mean(vertices, axis=0)

@jax.jit
def indexOfFurthestPoint(vertices, d):
    return jnp.argmax(jnp.dot(vertices, d))

@jax.jit
def support(vertices1, vertices2, d):
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]

@jax.jit
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap
    Implemented purely in JAX using `jax.lax.while_loop`.
    """
    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2
    # Prevent zero direction
    d = jnp.where(jnp.all(d == 0), jnp.array([1.0, 0.0]), d)

    a = support(vertices1, vertices2, d)
    
    # State tuple for while loop:
    # (iter_count, index, simplex, d, a, is_collision, is_done)
    # simplex shape (3, 2), index is scalar
    simplex = jnp.zeros((3, 2))
    simplex = simplex.at[0].set(a)
    
    init_val = (
        0,                 # iter_count
        0,                 # index
        simplex,           # simplex
        -a,                # d
        a,                 # a
        False,             # is_collision
        jnp.dot(d, a) <= 0 # is_done
    )

    def cond_fun(val):
        iter_count, index, simplex, d, a, is_collision, is_done = val
        return jnp.logical_and(~is_done, iter_count < 100)

    def body_fun(val):
        iter_count, index, simplex, d, a, is_collision, is_done = val
        
        a = support(vertices1, vertices2, d)
        index = index + 1
        simplex = simplex.at[index].set(a)
        
        # If the newly added point does not cross origin, origin is not in Minkowski sum
        early_exit = jnp.dot(d, a) <= 0
        
        ao = -a
        
        def handle_line(val_inner):
            # index < 2 case (line segment)
            idx, smp, d_in, ao_in = val_inner
            b = smp[0]
            ab = b - a
            d_out = tripleProduct(ab, ao_in, ab)
            # handle flat triangle
            d_out = jnp.where(jnp.linalg.norm(d_out) < 1e-10, perpendicular(ab), d_out)
            return d_out, smp, idx
            
        def handle_triangle(val_inner):
            # index >= 2 case (triangle)
            idx, smp, d_in, ao_in = val_inner
            b = smp[1]
            c = smp[0]
            ab = b - a
            ac = c - a
            
            acperp = tripleProduct(ab, ac, ac)
            
            # condition 1: acperp.dot(ao) >= 0
            cond1 = jnp.dot(acperp, ao_in) >= 0
            
            # if cond1 true
            d_out_cond1 = acperp
            
            # if cond1 false
            abperp = tripleProduct(ac, ab, ab)
            cond2 = jnp.dot(abperp, ao_in) < 0
            # if cond2 true (return collision = True)
            
            # if cond2 false
            smp_cond2f = smp.at[0].set(smp[1])
            d_out_cond2f = abperp
            idx_cond2f = idx - 1
            
            # We want to conditionally update `simplex` and `index`
            smp_new = jnp.where(cond1, smp, jnp.where(cond2, smp, smp_cond2f))
            d_new = jnp.where(cond1, d_out_cond1, jnp.where(cond2, d_in, d_out_cond2f))
            idx_new = jnp.where(cond1, idx, jnp.where(cond2, idx, idx_cond2f))
            
            # Re-assign b to c position if we go down from triangle to line (done automatically via smp swap and index-=1)
            # Actually, the original C logic says: `simplex[1, :] = simplex[2, :]`
            smp_final = smp_new.at[1].set(smp_new[2])
            idx_final = idx_new - 1
            
            return d_new, smp_final, idx_final, cond2 # cond2 means we found collision

        # Use jax.lax.cond for the `if index < 2`
        d_next, simplex_next, index_next, found_collision = jax.lax.cond(
            index < 2,
            lambda v: (*handle_line(v), False),
            lambda v: handle_triangle(v),
            (index, simplex, d, ao)
        )
        
        # update states
        is_col_new = jnp.logical_or(is_collision, found_collision) # either we found it now or had it
        is_done_new = jnp.logical_or(early_exit, is_col_new) 
        
        return (iter_count + 1, index_next, simplex_next, d_next, a, is_col_new, is_done_new)

    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    # Return `is_collision`
    return final_val[5]

@jax.jit
def collision_multiple(vertices):
    """
    Batched pairwise collision check for JAX using vmap over all pairs.
    Matches the original f110_gym collision_models.collision_multiple interface.
    Args:
        vertices: [N, 4, 2]
    Returns:
        collisions: [N] float array (1.0 if in collision, 0.0 otherwise)
        collision_idx: [N] float array (index of collision partner, -1.0 if none)
    """
    num_agents = vertices.shape[0]
    
    # Generate all pairs (i, j)
    v1 = jnp.broadcast_to(vertices[:, None, :, :], (num_agents, num_agents, 4, 2))
    v2 = jnp.broadcast_to(vertices[None, :, :, :], (num_agents, num_agents, 4, 2))
    
    # Filter out i == j
    mask = jnp.triu(jnp.ones((num_agents, num_agents), dtype=bool), k=1)
    
    # batched collision function
    batched_gjk = jax.vmap(jax.vmap(collision))
    
    # Check all pairs
    pair_collisions = batched_gjk(v1, v2)
    
    # Zero out diagonal and lower triangle
    valid_collisions = jnp.where(mask, pair_collisions, False)
    
    # Symmetrize: collision(i,j) == collision(j,i)
    symmetric_collisions = jnp.logical_or(valid_collisions, valid_collisions.T)
    
    # Any agent i is in collision if there's a True in its row
    agent_collisions = jnp.any(symmetric_collisions, axis=1).astype(jnp.float32)
    
    # collision_idx: index of last collision partner for each agent, -1 if none
    # Use argmax on reversed array to get the last True index (matching original loop behavior)
    # In the original, later pairs overwrite earlier ones, so we want the last collision partner
    indices = jnp.arange(num_agents)
    # For each agent, find the collision partner with the highest index
    # where symmetric_collisions[i, j] is True
    weighted = jnp.where(symmetric_collisions, indices[None, :], -1)
    collision_idx = jnp.max(weighted, axis=1).astype(jnp.float32)
    
    return agent_collisions, collision_idx

