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
    """固定長JAX配列を使ったステアリング遅延バッファの更新"""
    # 1列分左にシフトし、一番右に最新のsteerを入れる
    shifted = buffers.at[:, :-1].set(buffers[:, 1:])
    new_buffers = shifted.at[:, -1].set(new_steers)
    # 遅延されたステアリング値（一番古い値）は列0
    delayed_steers = new_buffers[:, 0]
    return new_buffers, delayed_steers

@jax.jit
def check_ttc_jax(scan, vel, cosines, side_distances, ttc_thresh):
    proj_vel = vel * cosines
    # ゼロ除算回避のため、proj_velが0以下の場合は安全な値(1.0)に置き換え
    valid_mask = proj_vel > 0.0
    safe_proj_vel = jnp.where(valid_mask, proj_vel, 1.0)
    
    ttc = (scan - side_distances) / safe_proj_vel
    
    # 衝突条件: ttcが閾値未満 かつ ttcが0以上 かつ proj_velが正
    is_collision = valid_mask & (ttc < ttc_thresh) & (ttc >= 0.0)
    
    # いずれかのビームで衝突していればTrue
    return jnp.any(is_collision)

class Integrator(Enum):
    RK4 = 1
    Euler = 2


class F110JaxSimulator:
    """
    Core GPU-accelerated simulator class.
    Faithfully replicates the behavior of the original f110_gym (NumPy+Numba)
    environment using JAX for GPU acceleration.
    
    Provides a stand-alone environment step function that maps directly 
    to the inputs and outputs expected by the ROS 2 GymBridge.
    """
    
    # Default 1/10 scale vehicle parameters (matching original f110_gym)
    DEFAULT_PARAMS = {
        'mu': 1.0489,
        'C_Sf': 4.718,
        'C_Sr': 5.4562,
        'lf': 0.15875,
        'lr': 0.17145,
        'h': 0.074,
        'm': 3.74,
        'I': 0.04712,
        's_min': -0.4189,
        's_max': 0.4189,
        'sv_min': -3.2,
        'sv_max': 3.2,
        'v_switch': 7.319,
        'a_max': 9.51,
        'v_min': -5.0,
        'v_max': 20.0,
        'width': 0.31,
        'length': 0.58,
        'laser_distance': 0.27,
    }
    
    def __init__(self, map_path, map_ext, num_agents=2, params=None,
                 seed=12345, time_step=0.01, ego_idx=0,
                 integrator=Integrator.RK4, num_beams=1080, fov=4.7,
                 lidar_dist=None):
        """
        Initialize the JAX simulator with the same interface as F110Env.
        
        Args:
            map_path: path to the map yaml file
            map_ext: extension of the map image file (e.g. '.png')
            num_agents: number of agents in the environment
            params: vehicle parameter dictionary (uses DEFAULT_PARAMS if None)
            seed: random seed for scan noise
            time_step: physics timestep (default 0.01s = 100Hz)
            ego_idx: index of the ego vehicle
            integrator: Integrator.RK4 or Integrator.Euler
            num_beams: number of LiDAR beams
            fov: field of view of LiDAR in radians
            lidar_dist: override for laser_distance parameter
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.integrator = integrator
        self.map_path = map_path
        self.map_ext = map_ext
        
        # Vehicle parameters
        if params is not None:
            self.params = dict(self.DEFAULT_PARAMS)
            self.params.update(params)
        else:
            self.params = dict(self.DEFAULT_PARAMS)
        
        # Override lidar_dist if explicitly provided
        if lidar_dist is not None:
            self.params['laser_distance'] = lidar_dist
        
        # Extract parameters for JAX usage
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
        
        # LiDAR Parameters
        self.num_beams = num_beams
        self.fov = fov
        self.eps = 0.0001
        self.theta_dis = 2000
        self.max_range = 30.0
        
        # Collision threshold for iTTC
        self.ttc_thresh = 0.005
        
        # Steering delay buffer (matching original: steer_buffer_size = 2)
        self.steer_buffer_size = 2
        # Each agent has its own buffer: [num_agents, steer_buffer_size]
        # Initialized empty (will be filled during step)
        self.steer_buffers_jax = jnp.zeros((self.num_agents, self.steer_buffer_size))
        
        # Precompute iTTC side distances and cosines for each beam
        self._precompute_ttc_tables()
        
        # Precompute scan angles for ray casting onto other agents
        self.scan_angles = np.linspace(-fov / 2., fov / 2., num_beams)
        
        # Prepare the state format: [x, y, steering, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = jnp.zeros((self.num_agents, 7))
        
        # Collision state
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        
        # Map data (loaded later)
        self.dt_jax = None
        self.resolution = 0.0
        self.origin = [0.0, 0.0, 0.0]
        self.orig_x = 0.0
        self.orig_y = 0.0
        self.orig_s = 0.0
        self.orig_c = 0.0
        self.height = 0
        self.width = 0
        
        # Load map info
        self._load_map(map_path, map_ext)
        
        # Scan noise RNG
        self.rng_key = jax.random.PRNGKey(self.seed)
        
        # JIT-compile the batched step logic using vmap
        self._batched_dynamics = jax.jit(jax.vmap(
            vehicle_dynamics_st,
            in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        ))
        
        # JIT batched scan (each agent's scan is computed in parallel)
        self._batched_scan = jax.jit(jax.vmap(
            get_scan,
            in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        ), static_argnums=(3, 15))

        self._batched_get_vertices = jax.jit(jax.vmap(
            get_vertices, 
            in_axes=(0, None, None)
        ))

        self._batched_pid = jax.jit(jax.vmap(
            pid,
            in_axes=(0, 0, 0, 0, None, None, None, None)
        ))

        self._batched_check_ttc = jax.jit(jax.vmap(
            check_ttc_jax, 
            in_axes=(0, 0, None, None, None)
        ))

    def _get_lidar_poses(self):
        """
        Compute the LiDAR sensor poses for all agents.
        Original f1tenth_gym offsets the laser from car center by laser_distance in the heading direction.
        Returns: [num_agents, 3] array of [x, y, yaw_angle]
        """
        yaw = self.state[:, 4]
        laser_x = self.state[:, 0] + self.lidar_dist * jnp.cos(yaw)
        laser_y = self.state[:, 1] + self.lidar_dist * jnp.sin(yaw)
        return jnp.stack([laser_x, laser_y, yaw], axis=1)

    def _precompute_ttc_tables(self):
        """Precompute the side_distances and cosines tables for iTTC wall collision check.
        Matches original base_classes.py RaceCar.__init__ logic."""
        scan_ang_incr = self.fov / (self.num_beams - 1)
        dist_sides = self.car_width / 2.0
        dist_fr = (self.lf + self.lr) / 2.0  # Match original: (lf+lr)/2
        
        cosines = np.zeros(self.num_beams)
        side_distances = np.zeros(self.num_beams)
        
        for i in range(self.num_beams):
            angle = -self.fov / 2.0 + i * scan_ang_incr
            cosines[i] = np.cos(angle)
            
            if angle > 0:
                if angle < np.pi / 2:
                    to_side = dist_sides / np.sin(angle)
                    to_fr = dist_fr / np.cos(angle)
                    side_distances[i] = min(to_side, to_fr)
                else:
                    to_side = dist_sides / np.cos(angle - np.pi / 2.0)
                    to_fr = dist_fr / np.sin(angle - np.pi / 2.0)
                    side_distances[i] = min(to_side, to_fr)
            else:
                if angle > -np.pi / 2:
                    to_side = dist_sides / np.sin(-angle)
                    to_fr = dist_fr / np.cos(-angle)
                    side_distances[i] = min(to_side, to_fr)
                else:
                    to_side = dist_sides / np.cos(-angle - np.pi / 2.0)
                    to_fr = dist_fr / np.sin(-angle - np.pi / 2.0)
                    side_distances[i] = min(to_side, to_fr)
        
        self.cosines_jax = jnp.array(cosines)
        self.side_distances_jax = jnp.array(side_distances)

    def _load_map(self, map_path, map_ext):
        import yaml
        from PIL import Image
        import os
        from scipy.ndimage import distance_transform_edt as edt

        # Load image and yaml (must run on CPU/Host for I/O)
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
        
        # Compute DT on CPU with scipy, move to GPU as static JAX device array
        dt = self.resolution * edt(map_img)
        self.dt_jax = jnp.array(dt)
        
        # Lap counting logic variables
        self.current_time = 0.0
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros(self.num_agents)
        
        self.start_xs = np.zeros(self.num_agents)
        self.start_ys = np.zeros(self.num_agents)
        self.start_thetas = np.zeros(self.num_agents)
        self.start_rot = np.eye(2)

    def reset(self, poses):
        """
        Reset the simulator to specific poses.
        Matches F110Env.reset() behavior.
        
        Args:
            poses (np.ndarray): Shape [num_agents, 3] where cols are [x, y, theta]
        Returns:
            obs (dict), reward (float), done (bool), info (dict)
        """
        assert poses.shape == (self.num_agents, 3), \
            f"Invalid poses shape: expected ({self.num_agents}, 3), got {poses.shape}"
        
        # Reset state: [x, y, steering, vel, yaw_angle, yaw_rate, slip_angle]
        new_state = jnp.zeros((self.num_agents, 7))
        new_state = new_state.at[:, 0:2].set(poses[:, 0:2])
        new_state = new_state.at[:, 4].set(poses[:, 2])
        self.state = new_state
        
        # Reset collisions
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        
        # Reset steering delay buffers
        self.steer_buffers_jax = jnp.zeros((self.num_agents, self.steer_buffer_size))

        # Reset lap counters
        self.current_time = 0.0
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros(self.num_agents)
        self.num_toggles = 0
        self.near_start = True
        
        # Set start positions for lap counting
        self.start_xs = poses[:, 0].copy()
        self.start_ys = poses[:, 1].copy()
        self.start_thetas = poses[:, 2].copy()
        
        cos_th = np.cos(-self.start_thetas[self.ego_idx])
        sin_th = np.sin(-self.start_thetas[self.ego_idx])
        self.start_rot = np.array([
            [cos_th, -sin_th],
            [sin_th, cos_th]
        ])
        
        # Reset scan RNG
        self.rng_key = jax.random.PRNGKey(self.seed)
        
        # Get initial observation with zero input (matching original)
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)
        
        return obs, reward, done, info

    def step(self, controls):
        """
        Advance simulator by one step.
        Matches F110Env.step() behavior faithfully.
        
        Args:
            controls (np.ndarray): Shape [num_agents, 2] where cols are [steer, speed]
        Returns:
            obs (dict), reward (float), done (bool), info (dict)
        """
        assert controls.shape == (self.num_agents, 2), \
            f"Invalid controls shape: expected ({self.num_agents}, 2), got {controls.shape}"
        
        # Apply steering delay (matching original)
        new_steers_jax = jnp.array(controls[:, 0])
        self.steer_buffers_jax, delayed_steers_jax = update_steer_buffer(
            self.steer_buffers_jax, new_steers_jax
        )
        delayed_steers = np.array(delayed_steers_jax)
        desired_speeds = controls[:, 1]
        
        # For each agent, compute PID (steer, speed -> sv, accl)
        # Only if not in collision (matching original: if not self.in_collision)
        current_steers = np.array(self.state[:, 2])
        current_speeds = np.array(self.state[:, 3])
        
        accls_jax, svs_jax = self._batched_pid(
            jnp.array(desired_speeds),
            delayed_steers_jax,
            jnp.array(current_speeds),
            jnp.array(current_steers),
            jnp.float32(self.sv_max),
            jnp.float32(self.a_max),
            jnp.float32(self.v_max),
            jnp.float32(self.v_min)
        )
        
        # 衝突していない(collisions == 0.0)エージェントのみ制御入力を有効にする
        active_mask = (jnp.array(self.collisions) == 0.0)
        accls_jax = jnp.where(active_mask, accls_jax, 0.0)
        svs_jax = jnp.where(active_mask, svs_jax, 0.0)
        
        # Stack into dynamics-compatible input [sv, accl]
        u_jax = jnp.stack([svs_jax, accls_jax], axis=1)
        
        # Physics integration (only for non-colliding agents, matching original)
        params = (self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr,
                  self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min,
                  self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        
        if self.integrator is Integrator.RK4:
            k1 = self._batched_dynamics(self.state, u_jax, *params)
            k2 = self._batched_dynamics(self.state + self.time_step * k1 / 2, u_jax, *params)
            k3 = self._batched_dynamics(self.state + self.time_step * k2 / 2, u_jax, *params)
            k4 = self._batched_dynamics(self.state + self.time_step * k3, u_jax, *params)
            new_state = self.state + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        elif self.integrator is Integrator.Euler:
            f = self._batched_dynamics(self.state, u_jax, *params)
            new_state = self.state + self.time_step * f
        else:
            raise SyntaxError(
                f"Invalid Integrator Specified. Provided {self.integrator.name}. "
                "Please choose RK4 or Euler"
            )
        
        # For collided agents, keep their state unchanged (matching original: skip physics update)
        collision_mask = jnp.array(self.collisions)[:, None]  # [num_agents, 1]
        self.state = jnp.where(collision_mask > 0., self.state, new_state)
        
        # Yaw angle wrapping to [0, 2*pi] (matching original base_classes.py)
        yaw = self.state[:, 4]
        yaw = jnp.where(yaw > 2 * jnp.pi, yaw - 2 * jnp.pi, yaw)
        yaw = jnp.where(yaw < 0, yaw + 2 * jnp.pi, yaw)
        self.state = self.state.at[:, 4].set(yaw)
        
        # Get agent poses for collision check
        agent_poses = np.array(jnp.column_stack([
            self.state[:, 0], self.state[:, 1], self.state[:, 4]
        ]))
        
        # Update LiDAR scans
        lidar_poses = self._get_lidar_poses()
        scans_jax = self._batched_scan(
            lidar_poses, self.theta_dis, self.fov, self.num_beams,
            None, None, self.eps, self.orig_x, self.orig_y,
            self.orig_c, self.orig_s, self.height, self.width,
            self.resolution, self.dt_jax, self.max_range
        )
        
        # Convert scans to numpy for per-agent processing
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.normal(subkey, shape=(self.num_agents, self.num_beams)) * 0.01
        scans_jax = scans_jax + noise
        scans_np = np.array(scans_jax) # 後続のNumPy処理用
                
        # Check car-to-car collisions (GJK)
        agent_poses_jax = jnp.array(agent_poses)
        all_vertices_jax = self._batched_get_vertices(agent_poses_jax, self.car_length, self.car_width)
        all_vertices = np.array(all_vertices_jax) # 後続のNumPy処理(ray_cast_agents等)用
        
        car_collisions, car_collision_idx = collision_multiple(all_vertices_jax)

        self.collisions = np.array(car_collisions)
        self.collision_idx = np.array(car_collision_idx)
        
        # Ray cast other agents onto each agent's scan (matching original)
        scan_angles_jnp = jnp.array(self.scan_angles)
        for i in range(self.num_agents):
            # Get opponent poses and vertices
            opp_indices = [j for j in range(self.num_agents) if j != i]
            if len(opp_indices) > 0:
                opp_verts = jnp.array(all_vertices[opp_indices])
                agent_pose = jnp.array([
                    float(self.state[i, 0]),
                    float(self.state[i, 1]),
                    float(self.state[i, 4])
                ])
                scans_np[i] = np.array(
                    ray_cast_agents(
                        jnp.array(scans_np[i]),
                        agent_pose,
                        scan_angles_jnp,
                        opp_verts
                    )
                )
        
        # Check iTTC wall collisions for each agent
        vels_jax = self.state[:, 3]
        # 全エージェントのiTTC壁衝突判定を一括計算
        ttc_collisions = self._batched_check_ttc(
            scans_jax, vels_jax, self.cosines_jax, self.side_distances_jax, self.ttc_thresh
        )
        
        # 衝突したエージェントは速度(3), ヨーレート(5), スリップ角(6)を0にリセット
        # 状態更新用マスク: 形を (num_agents, 1) に拡張
        stop_mask = ttc_collisions[:, None] 
        self.state = jnp.where(
            stop_mask, 
            self.state.at[:, 3].set(0.0).at[:, 5].set(0.0).at[:, 6].set(0.0), 
            self.state
        )
        
        # self.collisions (NumPy配列)の更新
        ttc_collisions_np = np.array(ttc_collisions)
        self.collisions = np.where(ttc_collisions_np, 1.0, self.collisions)
        
        # Update time
        self.current_time += self.time_step
        
        # Check lap completions
        self._check_done()
        
        # Generate observation
        obs = self._generate_obs(scans_np)
        
        # Reward = timestep (matching original)
        reward = self.time_step
        
        # Done condition (matching original)
        done = bool(self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        # Info (matching original)
        info = {'checkpoint_done': self.toggle_list >= 4}
        
        return obs, reward, done, info

    def _check_done(self):
        """
        Check if the current rollout is done (matching original f110_env._check_done).
        Uses numpy for lap counting state management.
        """
        left_t = 2
        right_t = 2
        
        state_cpu = np.array(self.state)
        poses_x = state_cpu[:, 0] - self.start_xs
        poses_y = state_cpu[:, 1] - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :].copy()
        
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0
        
        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

    def _generate_obs(self, scans_np):
        """
        Construct the output observation dictionary.
        Matches original base_classes.py Simulator.step() observation format exactly.
        
        Args:
            scans_np: [num_agents, num_beams] numpy array of scans
        Returns:
            obs (dict)
        """
        state_cpu = np.array(self.state)
        
        obs = {
            'ego_idx': self.ego_idx,
            'scans': [],
            'poses_x': [],
            'poses_y': [],
            'poses_theta': [],
            'linear_vels_x': [],
            'linear_vels_y': [],
            'ang_vels_z': [],
            'collisions': self.collisions.copy(),
            'lap_times': self.lap_times.copy(),
            'lap_counts': self.lap_counts.copy(),
        }
        
        for i in range(self.num_agents):
            obs['scans'].append(scans_np[i])
            obs['poses_x'].append(state_cpu[i, 0])
            obs['poses_y'].append(state_cpu[i, 1])
            obs['poses_theta'].append(state_cpu[i, 4])
            obs['linear_vels_x'].append(state_cpu[i, 3])
            obs['linear_vels_y'].append(0.)  # Match original: always 0
            obs['ang_vels_z'].append(state_cpu[i, 5])
        
        return obs

    def update_params(self, params, agent_idx=-1):
        """
        Updates the parameters used by simulation for vehicles.
        Matches original F110Env.update_params().
        
        Args:
            params: dictionary of parameters
            agent_idx: if >= 0 then only update a specific agent's params
        """
        # For simplicity in JAX version, we update global params
        # (original also uses shared params for all agents)
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
        
        # Recompute iTTC tables if dimensions changed
        self._precompute_ttc_tables()

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation.
        Matches original F110Env.update_map().
        """
        self._load_map(map_path, map_ext)
