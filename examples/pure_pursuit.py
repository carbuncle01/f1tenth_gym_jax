import sys
import os
import time
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import Namespace
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================================================================
# Pure Pursuit Planner (from original f1tenth_gym waypoint_follow.py)
# ==============================================================================
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)
        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c
        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1; first_i = i; first_p = start + t1 * V; break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2; first_i = i; first_p = start + t2 * V; break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1; first_i = i; first_p = start + t1 * V; break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2; first_i = i; first_p = start + t2 * V; break
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start
            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c
            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1; first_i = i; first_p = start + t1 * V; break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2; first_i = i; first_p = start + t2 * V; break
    return first_p, first_i, first_t

def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 is None:
                return None
            current_waypoint = np.empty((3, ))
            current_waypoint[0:2] = wpts[i2, :]
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return 4.0, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        return speed, steering_angle

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend device. Choose from auto, cpu, gpu.'
    )
    args, _ = parser.parse_known_args()

    if args.device != 'auto':
        os.environ['JAX_PLATFORM_NAME'] = args.device

    import jax
    from f110_jax.simulator import F110JaxSimulator, Integrator

    print(f"Using device setting: {args.device} (backend: {jax.default_backend()})")

    # マップディレクトリの解決
    gym_examples_dir = os.environ.get(
        'F1TENTH_MAP_DIR',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples'))
    )
    config_path = os.path.join(gym_examples_dir, 'config_example_map.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        print(f"Set F1TENTH_MAP_DIR to the directory containing config_example_map.yaml")
        sys.exit(1)
    
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    # Resolve relative paths to absolute
    conf_dict['map_path'] = os.path.join(gym_examples_dir, conf_dict['map_path'] + conf_dict['map_ext']).replace(conf_dict['map_ext'], '')
    conf_dict['wpt_path'] = os.path.join(gym_examples_dir, conf_dict['wpt_path'])
    conf = Namespace(**conf_dict)
    
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.8}

    print("Loading Pure Pursuit Planner...")
    planner = PurePursuitPlanner(conf, (0.17145 + 0.15875))
    
    print("Loading JAX Simulator...")
    yaml_path = conf.map_path + '.yaml'
    sim = F110JaxSimulator(yaml_path, conf.map_ext, num_agents=1, integrator=Integrator.RK4)
    
    # Start on the nearest waypoint
    wpt_data = planner.waypoints
    wpt_xy = np.vstack((wpt_data[:, conf.wpt_xind], wpt_data[:, conf.wpt_yind])).T
    start_pos = np.array([conf.sx, conf.sy])
    dists_to_start = np.linalg.norm(wpt_xy - start_pos, axis=1)
    nearest_wpt_idx = np.argmin(dists_to_start)
    next_idx = (nearest_wpt_idx + 1) % len(wpt_xy)
    dx = wpt_xy[next_idx, 0] - wpt_xy[nearest_wpt_idx, 0]
    dy = wpt_xy[next_idx, 1] - wpt_xy[nearest_wpt_idx, 1]
    start_theta = np.arctan2(dy, dx)
    
    poses = np.array([[wpt_xy[nearest_wpt_idx, 0], wpt_xy[nearest_wpt_idx, 1], start_theta]])
    print(f"Start pose: x={poses[0,0]:.3f}, y={poses[0,1]:.3f}, theta={np.degrees(start_theta):.1f}deg")
    obs, _, done, _ = sim.reset(poses)
    
    # ---- 1. ウォームアップ ----
    print("JAX Warming up...")
    dummy_action = np.array([[0.0, 0.0]])
    obs, _, _, _ = sim.step(dummy_action)
    
    # JAX配列（GPUメモリ）として出力されるため、block_until_readyで待機
    if hasattr(obs['scans'], 'block_until_ready'):
        obs['scans'].block_until_ready()
    
    # ---- 2. 本番シミュレーション ----
    # JAX配列からPython標準の float や NumPy配列 に明示的にキャストして保存する
    traj_x = [float(obs['poses_x'][0])]
    traj_y = [float(obs['poses_y'][0])]
    traj_theta = [float(obs['poses_theta'][0])]
    traj_scans = [np.array(obs['scans'][0])]
    
    laptime = 0.0
    steps = 0
    max_steps = 3000
    
    print("Simulating...")
    start = time.time()
    
    # doneもJAX配列になっている可能性があるため、bool()で評価
    while not bool(done) and steps < max_steps:
        # Plannerへの入力時に float() で CPUへ持ってくる
        px = float(obs['poses_x'][0])
        py = float(obs['poses_y'][0])
        ptheta = float(obs['poses_theta'][0])
        
        speed, steer = planner.plan(px, py, ptheta, work['tlad'], work['vgain'])
        
        # SimulatorはJAX環境だが、NumPyを渡せば自動でJAX配列に変換される
        obs, step_reward, done, info = sim.step(np.array([[steer, speed]]))
        
        # 結果をCPU(NumPy/float)に戻してリストに追加
        traj_x.append(float(obs['poses_x'][0]))
        traj_y.append(float(obs['poses_y'][0]))
        traj_theta.append(float(obs['poses_theta'][0]))
        traj_scans.append(np.array(obs['scans'][0]))
        
        laptime += sim.time_step
        steps += 1
        
    # 最後にもう一度待機
    if hasattr(obs['scans'], 'block_until_ready'):
        obs['scans'].block_until_ready()
        
    elapsed = time.time() - start
    
    print(f"Done: {steps} steps, {laptime:.1f}s sim, {elapsed:.1f}s real, {steps/elapsed:.0f} FPS")
    
    # --- 以下レンダリング処理（変更なし） ---
    print("Rendering video with LiDAR...")
    with open(yaml_path, 'r') as f:
        map_meta = yaml.safe_load(f)
    origin = map_meta['origin']
    resolution = map_meta['resolution']
    
    img = Image.open(conf.map_path + conf.map_ext).transpose(Image.FLIP_TOP_BOTTOM)
    img_array = np.array(img)
    height, width = img_array.shape
    
    extent = [
        origin[0], origin[0] + width * resolution,
        origin[1], origin[1] + height * resolution
    ]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array, cmap='gray', origin='lower', extent=extent, vmax=255, vmin=-50)
    
    wpt_x = wpt_data[:, conf.wpt_xind]
    wpt_y = wpt_data[:, conf.wpt_yind]
    ax.plot(wpt_x, wpt_y, 'g--', linewidth=1.5, alpha=0.7, label='Centerline')
    
    car_line, = ax.plot([], [], 'b-', linewidth=2.5, alpha=0.8, label='Trajectory')
    car_pos, = ax.plot([], [], 'ro', markersize=8, zorder=5, label='Car')
    lidar_pts = ax.scatter([], [], s=2, c='orange', alpha=0.6, zorder=4, label='LiDAR Hits')
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('JAX Pure Pursuit + LiDAR 2D Scan')
    
    frame_skip = 3 
    frames = steps // frame_skip
    
    scan_angles = np.array(sim.scan_angles)
    
    def init():
        car_line.set_data([], [])
        car_pos.set_data([], [])
        lidar_pts.set_offsets(np.empty((0, 2)))
        return car_line, car_pos, lidar_pts
    
    def animate(i):
        idx = min(i * frame_skip, len(traj_x) - 1)
        
        car_line.set_data(traj_x[:idx+1], traj_y[:idx+1])
        car_pos.set_data([traj_x[idx]], [traj_y[idx]])
        
        current_x = traj_x[idx]
        current_y = traj_y[idx]
        current_theta = traj_theta[idx]
        current_scan = traj_scans[idx]
        
        global_angles = current_theta + scan_angles
        pts_x = current_x + current_scan * np.cos(global_angles)
        pts_y = current_y + current_scan * np.sin(global_angles)
        
        lidar_pts.set_offsets(np.c_[pts_x, pts_y])
        return car_line, car_pos, lidar_pts
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=33, blit=True)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pure_pursuit_lidar.mp4')
    
    try:
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        ani.save(out_path, writer=writer)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"MP4の保存に失敗しました: {e}")
        alt_path = out_path.replace('.mp4', '.gif')
        ani.save(alt_path, writer='pillow', fps=30)

    plt.close()

if __name__ == '__main__':
    main()
