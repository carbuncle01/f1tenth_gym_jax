import time
import numpy as np
import jax

from f110_jax.simulator import F110JaxSimulator, Integrator

def run_benchmark(sim, num_steps=1000):
    # テスト用のダミー入力 (直進)
    # エージェント数2のアクション: [steer, speed]
    actions = np.array([[0.0, 5.0], [0.0, 5.0]])

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
    print(f"合計実行時間: {elapsed:.3f} 秒")
    print(f"FPS (ステップ/秒): {fps:.1f} FPS")

if __name__ == '__main__':
    # 兄弟ディレクトリの f1tenth_gym からマップを読み込む想定
    map_path = '..//examples/example_map.yaml'
    map_ext = '.png'
    
    try:
        print("シミュレータを初期化中...")
        sim = F110JaxSimulator(
            map_path=map_path, 
            map_ext=map_ext, 
            num_agents=2, 
            integrator=Integrator.RK4
        )
        
        # エージェントを少しずらして初期化
        poses = np.array([
            [0.0, 0.0, 0.0], 
            [1.0, 1.0, 0.0]
        ])
        sim.reset(poses)
        
        run_benchmark(sim, num_steps=1000)
        
    except FileNotFoundError:
        print(f"エラー: マップファイルが見つかりません。パスを確認してください: {map_path}")