"""
Pure Pursuit走行テスト

f1tenth_gym のexample_mapとwaypointsを使って、
JAXシミュレータでcenter line追従走行し、結果を動画で出力する。

Usage:
    python examples/pure_pursuit.py

マップファイルのデフォルトパスは兄弟ディレクトリの f1tenth_gym を参照。
環境変数 F1TENTH_MAP_DIR で任意のマップディレクトリを指定可能。
"""
import sys
import os
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import Namespace
from PIL import Image

# パッケージがインストールされていない場合のフォールバック
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from f110_jax.simulator import F110JaxSimulator, Integrator


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
    # マップディレクトリの解決
    gym_examples_dir = os.environ.get(
        'F1TENTH_MAP_DIR',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../f1tenth_gym/examples'))
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
    sim = F110JaxSimulator(yaml_path, conf.map_ext, num_agents=1)
    
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
    
    traj_x, traj_y = [], []
    laptime = 0.0
    start = time.time()
    steps = 0
    max_steps = 3000
    
    print("Simulating...")
    while not done and steps < max_steps:
        speed, steer = planner.plan(
            obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
            work['tlad'], work['vgain']
        )
        obs, step_reward, done, info = sim.step(np.array([[steer, speed]]))
        traj_x.append(obs['poses_x'][0])
        traj_y.append(obs['poses_y'][0])
        laptime += 0.01
        steps += 1
    
    elapsed = time.time() - start
    print(f"Done: {steps} steps, {laptime:.1f}s sim, {elapsed:.1f}s real, {steps/elapsed:.0f} FPS")
    print(f"Collisions: {obs['collisions']}, Laps: {obs['lap_counts']}")
    
    # ---- Render video ----
    print("Rendering video...")
    with open(yaml_path, 'r') as f:
        map_meta = yaml.safe_load(f)
    origin = map_meta['origin']
    resolution = map_meta['resolution']
    img = Image.open(conf.map_path + conf.map_ext)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', origin='lower', extent=[
        origin[0], origin[0] + img.size[0] * resolution,
        origin[1], origin[1] + img.size[1] * resolution
    ])
    wpt_x = wpt_data[:, conf.wpt_xind]
    wpt_y = wpt_data[:, conf.wpt_yind]
    ax.plot(wpt_x, wpt_y, 'g--', linewidth=1, label='Centerline')
    car_line, = ax.plot([], [], 'r-', linewidth=2, label='Trajectory')
    car_pos, = ax.plot([], [], 'bo', markersize=6, label='Car')
    ax.legend()
    ax.set_title('JAX Pure Pursuit Tracking')
    
    frame_skip = max(1, steps // 500)
    frames = steps // frame_skip
    
    def init():
        car_line.set_data([], [])
        car_pos.set_data([], [])
        return car_line, car_pos
    
    def animate(i):
        idx = i * frame_skip
        car_line.set_data(traj_x[:idx], traj_y[:idx])
        car_pos.set_data([traj_x[idx]], [traj_y[idx]])
        return car_line, car_pos
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=33, blit=True)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pure_pursuit.mp4')
    ani.save(out_path, writer=animation.FFMpegWriter(fps=30))
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
