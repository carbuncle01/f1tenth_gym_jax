import os
import sys
import time
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
from jax import tree_util
from argparse import Namespace

from f110_jax.simulator import F110JaxSimulator, Integrator
from pure_pursuit import PurePursuitPlanner

def pack_sim_state(sim):
    """
    SimulatorインスタンスからJAX関数に渡すための状態辞書を抽出するヘルパー関数
    """
    return {
        'state': sim.state,
        'collisions': jnp.array(sim.collisions),
        'collision_idx': jnp.array(sim.collision_idx),
        'steer_buffers': sim.steer_buffers_jax,
        'rng_key': sim.rng_key,
        'current_time': jnp.float32(sim.current_time),
        'lap_times': jnp.array(sim.lap_times),
        'lap_counts': jnp.array(sim.lap_counts),
        'near_starts': jnp.array(sim.near_starts),
        'toggle_list': jnp.array(sim.toggle_list),
        'start_xs': jnp.array(sim.start_xs),
        'start_ys': jnp.array(sim.start_ys),
        'start_rot': jnp.array(sim.start_rot)
    }

def main():
    num_envs = 10  # テストする環境の数
    max_steps = 300

    # マップと設定の読み込み
    gym_examples_dir = os.environ.get(
        'F1TENTH_MAP_DIR',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples'))
    )
    config_path = os.path.join(gym_examples_dir, 'config_example_map.yaml')
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    conf_dict['map_path'] = os.path.join(gym_examples_dir, conf_dict['map_path'] + conf_dict['map_ext']).replace(conf_dict['map_ext'], '')
    conf_dict['wpt_path'] = os.path.join(gym_examples_dir, conf_dict['wpt_path'])
    conf = Namespace(**conf_dict)
    
    planner = PurePursuitPlanner(conf, (0.17145 + 0.15875))
    yaml_path = conf.map_path + '.yaml'
    
    # シミュレータのインスタンスは「1つ」だけ作成する（これがマスターとなる）
    print("Loading JAX Simulator (Master)...")
    sim = F110JaxSimulator(yaml_path, conf.map_ext, num_agents=1, integrator=Integrator.RK4)
    
    wpt_xy = np.vstack((planner.waypoints[:, conf.wpt_xind], planner.waypoints[:, conf.wpt_yind])).T
    
    # --- 1. 複数環境分の初期状態（バッチ）を作成する ---
    states_list = []
    
    # num_envs に応じて、スタート地点をウェイポイント上に均等配置する
    start_indices = np.linspace(0, len(wpt_xy) - 1, num_envs, dtype=int).tolist()
    
    print(f"\n{num_envs}つの環境の初期化を開始:")
    for i, wpt_idx in enumerate(start_indices):
        next_idx = (wpt_idx + 1) % len(wpt_xy)
        start_theta = np.arctan2(wpt_xy[next_idx, 1] - wpt_xy[wpt_idx, 1], 
                                 wpt_xy[next_idx, 0] - wpt_xy[wpt_idx, 0])
        pose = np.array([[wpt_xy[wpt_idx, 0], wpt_xy[wpt_idx, 1], start_theta]])
        
        # 一旦マスターシミュレータをリセットして状態を作る
        sim.reset(pose)
        
        # 状態辞書を抽出してリストに保存
        states_list.append(pack_sim_state(sim))
        print(f"  Env {i}: Start at WP {wpt_idx} (x={pose[0,0]:.2f}, y={pose[0,1]:.2f})")

    # jax.tree_map を使って、num_envs 個の状態辞書を「1つのバッチ化された辞書」に合成（スタック）する
    # これにより、shape が (1, ...) から (num_envs, 1, ...) になります
    batched_state = tree_util.tree_map(lambda *xs: jnp.stack(xs), *states_list)

    # --- 2. ステップ関数のベクトル化 (vmap) ---
    print("\nJITコンパイル中 (Vectorized Step)...")
    # in_axes=(0, 0) は、「sim_stateとcontrolsの先頭次元(num_envs)を並列化してね」という指示です
    batched_step_fn = jax.jit(jax.vmap(sim._compiled_step, in_axes=(0, 0)))

    # ウォームアップ (ダミーアクションで初回コンパイルを済ませる)
    dummy_actions = jnp.zeros((num_envs, 1, 2)) # (num_envs環境, 1エージェント, [steer, speed])
    _ = batched_step_fn(batched_state, dummy_actions)
    jax.block_until_ready(batched_state['state'])
    print("ウォームアップ完了！\n")

    # --- 3. 並列シミュレーションループ ---
    work = {'tlad': 0.8246, 'vgain': 0.8}

    # 環境ごとの位置履歴を保存して動画化に使う
    history_x = [np.array(batched_state['state'][:, 0, 0])]
    history_y = [np.array(batched_state['state'][:, 0, 1])]
    
    start_time = time.time()
    for step in range(max_steps):
        actions = np.zeros((num_envs, 1, 2))
        
        # 各環境ごとにPure Pursuitでアクションを計算 (ここはPythonのforループ)
        # ※実際のRLなどでは、ニューラルネットで全環境分を一括推論します
        for i in range(num_envs):
            px = float(batched_state['state'][i, 0, 0])
            py = float(batched_state['state'][i, 0, 1])
            pth = float(batched_state['state'][i, 0, 4])
            
            speed, steer = planner.plan(px, py, pth, work['tlad'], work['vgain'])
            actions[i, 0, 0] = steer
            actions[i, 0, 1] = speed
            
        # num_envs 環境分のシミュレーションを「GPU/CPUのベクトル演算」で一撃で進める！
        batched_state, batched_obs, batched_rewards, batched_dones, batched_infos = batched_step_fn(
            batched_state, jnp.array(actions)
        )

        # JAX配列をNumPyに変換して全環境の履歴を保存
        history_x.append(np.array(batched_obs['poses_x'][:, 0]))
        history_y.append(np.array(batched_obs['poses_y'][:, 0]))
        
        # 10ステップごとに状況をプリント
        if step % 10 == 0:
            print(f"--- Step {step} ---")
            for i in range(num_envs):
                px = batched_obs['poses_x'][i, 0]
                py = batched_obs['poses_y'][i, 0]
                is_done = batched_dones[i]
                print(f"  Env {i}: pos=({px:.2f}, {py:.2f}) | done={is_done}")
                
    jax.block_until_ready(batched_state['state'])
    elapsed = time.time() - start_time
    print(f"\n{max_steps}ステップの並列実行完了: {elapsed:.3f}秒 ({(max_steps * num_envs)/elapsed:.1f} FPS)")

    # --- 4. 全環境の軌跡を同時に動画化 ---
    print("Rendering multi-env video...")
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

    x_hist = np.stack(history_x, axis=0)
    y_hist = np.stack(history_y, axis=0)
    total_frames = x_hist.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array, cmap='gray', origin='lower', extent=extent, vmax=255, vmin=-50)

    wpt_x = planner.waypoints[:, conf.wpt_xind]
    wpt_y = planner.waypoints[:, conf.wpt_yind]
    ax.plot(wpt_x, wpt_y, 'g--', linewidth=1.2, alpha=0.6, label='Centerline')

    cmap = plt.get_cmap('tab20')
    env_colors = [cmap(i % 20) for i in range(num_envs)]
    env_lines = []
    env_points = []
    for i in range(num_envs):
        line, = ax.plot([], [], linewidth=2.0, color=env_colors[i], alpha=0.85, label=f'Env {i}')
        point, = ax.plot([], [], 'o', markersize=5, color=env_colors[i], zorder=5)
        env_lines.append(line)
        env_points.append(point)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title(f'Parallel Multi-Env Trajectories (num_envs={num_envs})')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    frame_skip = 2
    render_frames = max(1, total_frames // frame_skip)

    def init_anim():
        artists = []
        for i in range(num_envs):
            env_lines[i].set_data([], [])
            env_points[i].set_data([], [])
            artists.extend((env_lines[i], env_points[i]))
        return artists

    def animate(frame_idx):
        idx = min(frame_idx * frame_skip, total_frames - 1)
        artists = []
        for i in range(num_envs):
            env_lines[i].set_data(x_hist[:idx + 1, i], y_hist[:idx + 1, i])
            env_points[i].set_data([x_hist[idx, i]], [y_hist[idx, i]])
            artists.extend((env_lines[i], env_points[i]))
        return artists

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_anim,
        frames=render_frames,
        interval=33,
        blit=True
    )

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'multi_env_{num_envs}.mp4')

    try:
        writer = animation.FFMpegWriter(fps=30, bitrate=2500)
        ani.save(out_path, writer=writer)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"MP4の保存に失敗しました: {e}")
        alt_path = out_path.replace('.mp4', '.gif')
        ani.save(alt_path, writer='pillow', fps=20)
        print(f"Saved GIF: {alt_path}")
    finally:
        plt.close(fig)

if __name__ == '__main__':
    main()