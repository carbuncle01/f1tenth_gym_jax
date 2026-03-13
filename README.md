# f1tenth_gym_jax

JAX (GPU対応) による F1TENTH シミュレータ。  
オリジナルの [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) (NumPy + Numba) の挙動を JAX で完全に再現しています。

## 構成

```
f1tenth_gym_jax/
├── setup.py                 # pip install -e . で利用可能
├── README.md
│
├── f110_jax/                # コアライブラリ
│   ├── __init__.py
│   ├── simulator.py         # メインシミュレータ (F110JaxSimulator)
│   ├── dynamics.py          # 車両ダイナミクス (ST/KS + PID)
│   ├── lidar.py             # LiDARレイキャスト + 他車両レイキャスト
│   └── collision.py         # GJK衝突検出
│
├── examples/
│   ├── pure_pursuit.py      # center line追従走行
│   └── gym_bridge.py        # ROS2 GymBridge連携
│
└── tests/
    ├── test_dynamics.py     # ダイナミクス数値一致テスト
    └── test_integration.py  # 統合テスト (reset/step)
```

## インストール

```bash
cd f1tenth_gym_jax
pip install -e .
```

GPU利用時は、環境に合わせた `jaxlib` を別途インストールしてください：
```bash
pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 使い方

```python
import numpy as np
from f110_jax import F110JaxSimulator, Integrator

sim = F110JaxSimulator(
    map_path='path/to/map.yaml',
    map_ext='.png',
    num_agents=2,
    integrator=Integrator.RK4,
)

poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
obs, reward, done, info = sim.reset(poses)

while not done:
    action = np.array([[steer, speed], [steer2, speed2]])
    obs, reward, done, info = sim.step(action)
```

## オリジナルとの互換性

- 車両パラメータ: 1/10スケール (オリジナルと同一デフォルト)
- ステアリング遅延バッファ: size=2
- 積分方式: RK4 / Euler 選択可
- 衝突検出: GJK + iTTC (衝突時速度リセット)
- LiDAR: レイマーチング + 他車両レイキャスト + ガウシアンノイズ
- Observation形式: オリジナルと完全一致

## テスト

```bash
python -m pytest tests/ -v
```
