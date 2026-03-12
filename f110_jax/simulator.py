import jax
import jax.numpy as jnp
import numpy as np
from enum import Enum

# Import JAX physics and LiDAR components
from f110_jax.dynamics import vehicle_dynamics_st, pid
from f110_jax.lidar import get_scan, ray_cast_agents
from f110_jax.collision import collision_multiple, get_vertices

@jax.jit
def update_steer_buffer(buffers, new_steers):
    shifted = buffers.at[:, :-1].set(buffers[:, 1:])
    new_buffers = shifted.at[:, -1].set(new_steers)
    delayed_steers = new_buffers[:, 0]
    return new_buffers, delayed_steers

@jax.jit
def check_ttc_jax(scan, vel, cosines, side_distances, ttc_thresh):
    proj_vel = vel * cosines
    valid_mask = proj_vel > 0.0
    safe_proj_vel = jnp.where(valid_mask, proj_vel, 1.0)
    ttc = (scan - side_distances) / safe_proj_vel
    is_collision = valid_mask & (ttc < ttc_thresh) & (ttc >= 0.0)
    return jnp.any(is_collision)

class Integrator(Enum):
    RK4 = 1
    Euler = 2

class F110JaxSimulator:
    DEFAULT_PARAMS = {
        'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 
        'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 
        'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 
        'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58, 'laser_distance': 0.27,
    }
    
    def __init__(self, map_path, map_ext, num_agents=2, params=None,
                 seed=12345, time_step=0.01, ego_idx=0,
                 integrator=Integrator.RK4, num_beams=1080, fov=4.7,
                 lidar_dist=None):
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.integrator = integrator
        self.map_path = map_path
        self.map_ext = map_ext
        
        if params is not None:
            self.params = dict(self.DEFAULT_PARAMS)
            self.params.update(params)
        else:
            self.params = dict(self.DEFAULT_PARAMS)
        if lidar_dist is not None:
            self.params['laser_distance'] = lidar_dist
            
        self.mu = self.params['mu']
        self.C_Sf = self.params['C_Sf']
        self.C_Sr = self.params['C_Sr']
        self.lf = self.params['lf']
        self.lr = self.params['lr']
        self.h = self.params['h']
        self.m = self.params['m']
        self.I = self.params['I']
        self.s_min = self.params['s_min']
        self.s_max = self.params['s_max']
        self.sv_min = self.params['sv_min']
        self.sv_max = self.params['sv_max']
        self.v_switch = self.params['v_switch']
        self.a_max = self.params['a_max']
        self.v_min = self.params['v_min']
        self.v_max = self.params['v_max']
        self.car_length = self.params['length']
        self.car_width = self.params['width']
        self.lidar_dist = self.params['laser_distance']
        
        self.num_beams = num_beams
        self.fov = fov
        self.eps = 0.0001
        self.theta_dis = 2000
        self.max_range = 30.0
        self.ttc_thresh = 0.005
        
        self.steer_buffer_size = 2
        self.steer_buffers_jax = jnp.zeros((self.num_agents, self.steer_buffer_size))
        
        self._precompute_ttc_tables()
        self.scan_angles = np.linspace(-fov / 2., fov / 2., num_beams)
        self.state = jnp.zeros((self.num_agents, 7))
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        
        self.dt_jax = None
        self.resolution = 0.0
        self.origin = [0.0, 0.0, 0.0]
        self.orig_x = 0.0
        self.orig_y = 0.0
        self.orig_s = 0.0
        self.orig_c = 0.0
        self.height = 0
        self.width = 0
        
        self._load_map(map_path, map_ext)
        self.rng_key = jax.random.PRNGKey(self.seed)
        
        # JIT function definitions
        self._batched_dynamics = jax.jit(jax.vmap(vehicle_dynamics_st, in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)))
        self._batched_scan = jax.jit(jax.vmap(get_scan, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)), static_argnums=(3, 15))
        self._batched_get_vertices = jax.jit(jax.vmap(get_vertices, in_axes=(0, None, None)))
        self._batched_pid = jax.jit(jax.vmap(pid, in_axes=(0, 0, 0, 0, None, None, None, None)))
        self._batched_check_ttc = jax.jit(jax.vmap(check_ttc_jax, in_axes=(0, 0, None, None, None)))
        
        # Build End-to-End step kernel
        self._build_step_kernel()

    def _precompute_ttc_tables(self):
        scan_ang_incr = self.fov / (self.num_beams - 1)
        dist_sides = self.car_width / 2.0
        dist_fr = (self.lf + self.lr) / 2.0
        cosines = np.zeros(self.num_beams)
        side_distances = np.zeros(self.num_beams)
        
        for i in range(self.num_beams):
            angle = -self.fov / 2.0 + i * scan_ang_incr
            cosines[i] = np.cos(angle)
            if angle > 0:
                if angle < np.pi / 2:
                    side_distances[i] = min(dist_sides / np.sin(angle), dist_fr / np.cos(angle))
                else:
                    side_distances[i] = min(dist_sides / np.cos(angle - np.pi / 2.0), dist_fr / np.sin(angle - np.pi / 2.0))
            else:
                if angle > -np.pi / 2:
                    side_distances[i] = min(dist_sides / np.sin(-angle), dist_fr / np.cos(-angle))
                else:
                    side_distances[i] = min(dist_sides / np.cos(-angle - np.pi / 2.0), dist_fr / np.sin(-angle - np.pi / 2.0))
        
        self.cosines_jax = jnp.array(cosines)
        self.side_distances_jax = jnp.array(side_distances)

    def _load_map(self, map_path, map_ext):
        import yaml
        from PIL import Image
        import os
        from scipy.ndimage import distance_transform_edt as edt

        map_img_path = os.path.splitext(map_path)[0] + map_ext
        map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        map_img = map_img.astype(np.float64)
        map_img[map_img <= 128.] = 0.
        map_img[map_img > 128.] = 255.
        
        with open(map_path, 'r') as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.resolution = map_metadata['resolution']
            self.origin = map_metadata['origin']
            
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])
        self.height = map_img.shape[0]
        self.width = map_img.shape[1]
        
        dt = self.resolution * edt(map_img)
        self.dt_jax = jnp.array(dt)
        
        self.current_time = 0.0
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros(self.num_agents)
        
        self.start_xs = np.zeros(self.num_agents)
        self.start_ys = np.zeros(self.num_agents)
        self.start_thetas = np.zeros(self.num_agents)
        self.start_rot = np.eye(2)

    def _build_step_kernel(self):
        """シミュレータの1ステップ全体をJITコンパイルする純粋関数を構築"""
        @jax.jit
        def step_kernel(sim_state, controls):
            # 1. Steering Buffer & PID
            new_buffers, delayed_steers = update_steer_buffer(sim_state['steer_buffers'], controls[:, 0])
            desired_speeds = controls[:, 1]
            
            accls, svs = self._batched_pid(
                desired_speeds, delayed_steers,
                sim_state['state'][:, 3], sim_state['state'][:, 2],
                self.sv_max, self.a_max, self.v_max, self.v_min
            )
            
            active_mask = (sim_state['collisions'] == 0.0)
            accls = jnp.where(active_mask, accls, 0.0)
            svs = jnp.where(active_mask, svs, 0.0)
            u = jnp.stack([svs, accls], axis=1)
            
            # 2. Dynamics
            params = (self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr,
                      self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min,
                      self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
            
            if self.integrator is Integrator.RK4:
                k1 = self._batched_dynamics(sim_state['state'], u, *params)
                k2 = self._batched_dynamics(sim_state['state'] + self.time_step * k1 / 2, u, *params)
                k3 = self._batched_dynamics(sim_state['state'] + self.time_step * k2 / 2, u, *params)
                k4 = self._batched_dynamics(sim_state['state'] + self.time_step * k3, u, *params)
                new_state = sim_state['state'] + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            else:
                f = self._batched_dynamics(sim_state['state'], u, *params)
                new_state = sim_state['state'] + self.time_step * f
                
            new_state = jnp.where(sim_state['collisions'][:, None] > 0., sim_state['state'], new_state)
            
            yaw = new_state[:, 4]
            yaw = jnp.where(yaw > 2 * jnp.pi, yaw - 2 * jnp.pi, yaw)
            yaw = jnp.where(yaw < 0, yaw + 2 * jnp.pi, yaw)
            new_state = new_state.at[:, 4].set(yaw)
            
            # 3. LiDAR Scan & Noise
            laser_x = new_state[:, 0] + self.lidar_dist * jnp.cos(yaw)
            laser_y = new_state[:, 1] + self.lidar_dist * jnp.sin(yaw)
            lidar_poses = jnp.stack([laser_x, laser_y, yaw], axis=1)
            
            scans = self._batched_scan(
                lidar_poses, self.theta_dis, self.fov, self.num_beams,
                None, None, self.eps, self.orig_x, self.orig_y,
                self.orig_c, self.orig_s, self.height, self.width,
                self.resolution, self.dt_jax, self.max_range
            )
            
            rng, subkey = jax.random.split(sim_state['rng_key'])
            noise = jax.random.normal(subkey, shape=(self.num_agents, self.num_beams)) * 0.01
            scans = scans + noise
            
            # 4. GJK Collisions
            agent_poses = jnp.column_stack([new_state[:, 0], new_state[:, 1], new_state[:, 4]])
            all_vertices = self._batched_get_vertices(agent_poses, self.car_length, self.car_width)
            car_collisions, car_collision_idx = collision_multiple(all_vertices)
            
            # 5. Ray Cast Agents
            batched_ray_cast = jax.vmap(ray_cast_agents, in_axes=(0, 0, None, None, 0))
            scans = batched_ray_cast(scans, agent_poses, jnp.array(self.scan_angles), all_vertices, jnp.arange(self.num_agents))
            
            # 6. iTTC Collisions
            vels = new_state[:, 3]
            ttc_collisions = self._batched_check_ttc(scans, vels, self.cosines_jax, self.side_distances_jax, self.ttc_thresh)
            
            stop_mask = ttc_collisions[:, None]
            new_state = jnp.where(stop_mask, new_state.at[:, 3].set(0).at[:, 5].set(0).at[:, 6].set(0), new_state)
            
            final_collisions = jnp.logical_or(car_collisions, ttc_collisions).astype(jnp.float32)
            
            # 7. Laps & Done Check
            new_time = sim_state['current_time'] + self.time_step
            poses_x = new_state[:, 0] - sim_state['start_xs']
            poses_y = new_state[:, 1] - sim_state['start_ys']
            delta_pt = jnp.dot(sim_state['start_rot'], jnp.stack((poses_x, poses_y), axis=0))
            temp_y = delta_pt[1, :]
            
            idx1 = temp_y > 2.0
            idx2 = temp_y < -2.0
            temp_y = jnp.where(idx1, temp_y - 2.0, temp_y)
            temp_y = jnp.where(idx2, -2.0 - temp_y, temp_y)
            temp_y = jnp.where(~(idx1 | idx2), 0.0, temp_y)
            
            dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
            closes = dist2 <= 0.1
            
            cond1 = closes & (~sim_state['near_starts'])
            cond2 = (~closes) & sim_state['near_starts']
            
            new_near_starts = jnp.where(cond1, True, sim_state['near_starts'])
            new_near_starts = jnp.where(cond2, False, new_near_starts)
            
            new_toggle_list = sim_state['toggle_list'] + jnp.where(cond1 | cond2, 1, 0)
            new_lap_counts = new_toggle_list // 2
            new_lap_times = jnp.where(new_toggle_list < 4, new_time, sim_state['lap_times'])
            
            done = jnp.logical_or(final_collisions[self.ego_idx], jnp.all(new_toggle_list >= 4))
            
            # 8. Pack output
            new_sim_state = {
                'state': new_state,
                'collisions': final_collisions,
                'collision_idx': car_collision_idx,
                'steer_buffers': new_buffers,
                'rng_key': rng,
                'current_time': new_time,
                'lap_times': new_lap_times,
                'lap_counts': new_lap_counts,
                'near_starts': new_near_starts,
                'toggle_list': new_toggle_list,
                'start_xs': sim_state['start_xs'],
                'start_ys': sim_state['start_ys'],
                'start_rot': sim_state['start_rot']
            }
            
            obs_jax = {
                'ego_idx': self.ego_idx,
                'scans': scans,
                'poses_x': new_state[:, 0],
                'poses_y': new_state[:, 1],
                'poses_theta': new_state[:, 4],
                'linear_vels_x': new_state[:, 3],
                'linear_vels_y': jnp.zeros_like(new_state[:, 3]),
                'ang_vels_z': new_state[:, 5],
                'collisions': final_collisions,
                'lap_times': new_lap_times,
                'lap_counts': new_lap_counts
            }
            
            return new_sim_state, obs_jax, self.time_step, done, {'checkpoint_done': new_toggle_list >= 4}

        self._compiled_step = step_kernel

    def reset(self, poses):
        assert poses.shape == (self.num_agents, 3), f"Invalid poses shape: expected ({self.num_agents}, 3), got {poses.shape}"
        
        new_state = jnp.zeros((self.num_agents, 7))
        new_state = new_state.at[:, 0:2].set(poses[:, 0:2])
        new_state = new_state.at[:, 4].set(poses[:, 2])
        self.state = new_state
        
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        self.steer_buffers_jax = jnp.zeros((self.num_agents, self.steer_buffer_size))

        self.current_time = 0.0
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros(self.num_agents)
        
        self.start_xs = poses[:, 0].copy()
        self.start_ys = poses[:, 1].copy()
        self.start_thetas = poses[:, 2].copy()
        
        cos_th = np.cos(-self.start_thetas[self.ego_idx])
        sin_th = np.sin(-self.start_thetas[self.ego_idx])
        self.start_rot = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        
        self.rng_key = jax.random.PRNGKey(self.seed)
        
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def step(self, controls):
        # 状態をディクショナリにパッキングしてJAX関数に渡す
        sim_state = {
            'state': self.state,
            'collisions': jnp.array(self.collisions),
            'collision_idx': jnp.array(self.collision_idx),
            'steer_buffers': self.steer_buffers_jax,
            'rng_key': self.rng_key,
            'current_time': jnp.float32(self.current_time),
            'lap_times': jnp.array(self.lap_times),
            'lap_counts': jnp.array(self.lap_counts),
            'near_starts': jnp.array(self.near_starts),
            'toggle_list': jnp.array(self.toggle_list),
            'start_xs': jnp.array(self.start_xs),
            'start_ys': jnp.array(self.start_ys),
            'start_rot': jnp.array(self.start_rot)
        }
        
        # End-to-End JIT実行（ここで1ステップの全計算が完了します）
        new_sim_state, obs_jax, reward, done, info = self._compiled_step(sim_state, jnp.array(controls))
        
        # 新しい状態をクラスの変数にアンパッキング
        self.state = new_sim_state['state']
        self.collisions = np.array(new_sim_state['collisions'])
        self.collision_idx = np.array(new_sim_state['collision_idx'])
        self.steer_buffers_jax = new_sim_state['steer_buffers']
        self.rng_key = new_sim_state['rng_key']
        self.current_time = float(new_sim_state['current_time'])
        self.lap_times = np.array(new_sim_state['lap_times'])
        self.lap_counts = np.array(new_sim_state['lap_counts'])
        self.near_starts = np.array(new_sim_state['near_starts'])
        self.toggle_list = np.array(new_sim_state['toggle_list'])
        
        # オリジナルのf110_gymと互換性を持たせるため、出力をPythonのlist/floatに変換
        obs = {
            'ego_idx': int(obs_jax['ego_idx']),
            'scans': [np.array(s) for s in obs_jax['scans']],
            'poses_x': [float(x) for x in obs_jax['poses_x']],
            'poses_y': [float(y) for y in obs_jax['poses_y']],
            'poses_theta': [float(t) for t in obs_jax['poses_theta']],
            'linear_vels_x': [float(v) for v in obs_jax['linear_vels_x']],
            'linear_vels_y': [float(v) for v in obs_jax['linear_vels_y']],
            'ang_vels_z': [float(w) for w in obs_jax['ang_vels_z']],
            'collisions': self.collisions.copy(),
            'lap_times': self.lap_times.copy(),
            'lap_counts': self.lap_counts.copy()
        }
        
        return obs, float(reward), bool(done), info

    def update_params(self, params, agent_idx=-1):
        self.params.update(params)
        self.mu = self.params['mu']
        self.C_Sf = self.params['C_Sf']
        self.C_Sr = self.params['C_Sr']
        self.lf = self.params['lf']
        self.lr = self.params['lr']
        self.h = self.params['h']
        self.m = self.params['m']
        self.I = self.params['I']
        self.s_min = self.params['s_min']
        self.s_max = self.params['s_max']
        self.sv_min = self.params['sv_min']
        self.sv_max = self.params['sv_max']
        self.v_switch = self.params['v_switch']
        self.a_max = self.params['a_max']
        self.v_min = self.params['v_min']
        self.v_max = self.params['v_max']
        if 'width' in params:
            self.car_width = params['width']
        if 'length' in params:
            self.car_length = params['length']
        if 'laser_distance' in params:
            self.lidar_dist = params['laser_distance']
        
        self._precompute_ttc_tables()
        self._build_step_kernel()  # パラメータが変わったらJITカーネルを再構築

    def update_map(self, map_path, map_ext):
        self._load_map(map_path, map_ext)
        self._build_step_kernel()  # マップが変わったらJITカーネルを再構築