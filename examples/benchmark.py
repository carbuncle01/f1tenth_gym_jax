import time
import os
import argparse
import numpy as np
import jax
import matplotlib.pyplot as plt

from f110_jax.simulator import F110JaxSimulator, Integrator

def generate_initial_poses(num_agents):
    """初期姿勢を生成する。車両同士の重なりを避けるために y 方向へ等間隔で配置する。"""
    poses = np.zeros((num_agents, 3), dtype=np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = np.linspace(0.0, 0.4 * max(num_agents - 1, 0), num_agents)
    poses[:, 2] = 0.0
    return poses


def run_benchmark(sim, num_agents, num_steps=1000, speed=5.0):
    # テスト用のダミー入力 (直進): [steer, speed]
    actions = np.zeros((num_agents, 2), dtype=np.float32)
    actions[:, 1] = speed

    print("--- 1. ウォームアップ開始 (JITコンパイル) ---")
    start_warmup = time.time()
    # 初回のstep呼び出し時にJITコンパイルが走ります
    sim.step(actions)
    # 完了を保証するため非同期実行をブロック (JAX特有の作法)
    jax.block_until_ready(sim.state) 
    warmup_time = time.time() - start_warmup
    print(f"ウォームアップ完了: {warmup_time:.3f} 秒")

    print(f"\n--- 2. ベンチマーク開始 ({num_steps} ステップ) ---")
    start_time = time.time()
    
    for _ in range(num_steps):
        sim.step(actions)
        
    # 全ての計算が終わるのを待つ
    jax.block_until_ready(sim.state)
    elapsed = time.time() - start_time

    fps = num_steps / elapsed
    env_fps = (num_steps * num_agents) / elapsed
    print(f"合計実行時間: {elapsed:.3f} 秒")
    print(f"FPS (シミュレータ step/秒): {fps:.1f}")
    print(f"Throughput (env-step/秒): {env_fps:.1f}")
    return {
        'num_agents': num_agents,
        'num_steps': num_steps,
        'elapsed_sec': elapsed,
        'fps_step_per_sec': fps,
        'throughput_env_step_per_sec': env_fps,
        'warmup_sec': warmup_time,
    }


def save_results(results, out_csv_path):
    header = (
        "num_agents,num_steps,elapsed_sec,fps_step_per_sec,"
        "throughput_env_step_per_sec,warmup_sec\n"
    )
    with open(out_csv_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for r in results:
            f.write(
                f"{r['num_agents']},{r['num_steps']},{r['elapsed_sec']:.6f},"
                f"{r['fps_step_per_sec']:.6f},{r['throughput_env_step_per_sec']:.6f},"
                f"{r['warmup_sec']:.6f}\n"
            )


def plot_results(results, out_png_path):
    num_agents = [r['num_agents'] for r in results]
    fps = [r['fps_step_per_sec'] for r in results]
    throughput = [r['throughput_env_step_per_sec'] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(num_agents, fps, 'o-', color='tab:blue', linewidth=2, label='Step FPS')
    ax1.set_xlabel('Number of Environments (num_agents)')
    ax1.set_ylabel('Step FPS [steps/s]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(num_agents, throughput, 's--', color='tab:red', linewidth=2, label='Env Throughput')
    ax2.set_ylabel('Throughput [env-steps/s]', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    ax1.set_title('F110 JAX Benchmark: Environments vs FPS')

    plt.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def parse_env_counts(env_counts_str):
    return [int(x.strip()) for x in env_counts_str.split(',') if x.strip()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark FPS vs number of environments.')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of benchmark steps per env count.')
    parser.add_argument('--speed', type=float, default=5.0, help='Constant speed used for dummy action.')
    parser.add_argument(
        '--env-counts',
        type=str,
        default='1,2,4,8,16,32',
        help='Comma-separated list of environment counts. Example: 1,2,4,8,16'
    )
    args = parser.parse_args()

    env_counts = parse_env_counts(args.env_counts)
    if any(c <= 0 for c in env_counts):
        raise ValueError('All env counts must be positive integers.')

    base_dir = os.path.dirname(__file__)
    map_path = os.path.join(base_dir, 'example_map.yaml')
    map_ext = '.png'
    out_dir = os.path.join(base_dir, '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_csv_path = os.path.join(out_dir, 'benchmark_env_vs_fps.csv')
    out_png_path = os.path.join(out_dir, 'benchmark_env_vs_fps.png')
    results = []
    
    try:
        print(f"ベンチマーク対象の環境数: {env_counts}")
        for num_agents in env_counts:
            print(f"\n================ num_agents={num_agents} ================")
            print("シミュレータを初期化中...")
            sim = F110JaxSimulator(
                map_path=map_path,
                map_ext=map_ext,
                num_agents=num_agents,
                integrator=Integrator.RK4
            )

            poses = generate_initial_poses(num_agents)
            sim.reset(poses)

            result = run_benchmark(
                sim,
                num_agents=num_agents,
                num_steps=args.num_steps,
                speed=args.speed
            )
            results.append(result)

        save_results(results, out_csv_path)
        plot_results(results, out_png_path)

        print("\n==== ベンチマーク完了 ====")
        print(f"CSV: {out_csv_path}")
        print(f"Plot: {out_png_path}")
        
    except FileNotFoundError:
        print(f"エラー: マップファイルが見つかりません。パスを確認してください: {map_path}")