# 拼图强化学习系统 — 三阶段实施计划

## Context

现有项目已有完整的拼图还原系统（图像切分、边缘匹配、求解算法、iOS自动化），但通过手机截图训练RL不现实。需要：(1) 网页版拼图游戏作为RL训练环境，(2) RL训练框架，(3) 模型迁移到移动端。目标是先跑通端到端流程，再迭代优化。

---

## Phase 1: 网页拼图游戏

### 目标
纯HTML/CSS/JS实现6×6/7×7/8×8拼图游戏，支持上传自定义图片，拖动交换拼图块。

### 目录结构
```
web/
├── static/
│   ├── index.html          # 单页应用：画布+控制栏
│   ├── style.css           # 样式
│   └── puzzle_game.js      # 核心游戏逻辑（单文件，无构建工具）
└── server.py               # Python HTTP服务器
```

### 核心实现

**`web/static/puzzle_game.js`** — `PuzzleGame` 类：
- `grid[row][col] = pieceIndex`（与Python端numpy grid编码一致）
- `loadImage(img)`: 将图片按grid大小切分为pieces
- `swapPieces(posA, posB)`: 交换两块，更新grid
- `getGrid()` / `setGrid(grid)`: 状态读写（供RL控制）
- `isComplete()`: 检查是否还原
- Canvas渲染 + 点击/拖拽交换交互
- 视觉反馈：选中高亮（蓝色）、正确位置（绿色）、错误位置（红色）

**`web/server.py`** — `PuzzleHTTPHandler`：
- 静态文件服务
- API: `GET /api/image?seed=42` 生成图片，`POST /api/upload` 上传图片
- 用 `python3 -m web.server` 启动

### 复用现有代码
- 切分逻辑：参考 `jigsaw/splitter.py` 的 `PuzzleSplitter.split()` 在JS中实现等效逻辑
- 图片生成：`jigsaw/utils.py::generate_natural_like_image()` 作为默认图片来源

### 验证
```bash
conda activate jigsaw
python3 -m web.server
# 浏览器打开 http://localhost:8080，选择6x6，上传图片或用默认图片，拖拽交换拼图块
```

---

## Phase 2: RL训练框架

### 关键设计决策
- **Headless Python模拟**：不通过浏览器训练（太慢），用纯Python实现等效游戏逻辑
- **复用现有模块**：`jigsaw/splitter.py`、`jigsaw/matcher.py`、`jigsaw/evaluator.py`
- **Action space**: `Discrete(n*(n-1)//2)`，单步离散动作，先跑通再调整
- **奖励函数**: 基于cost差值（improvement-based），解决奖励+步数惩罚

### 目录结构
```
rl/
├── __init__.py
├── envs/
│   ├── __init__.py
│   └── puzzle_env.py       # Gymnasium环境 PuzzleSwapEnv
├── training/
│   ├── __init__.py
│   ├── train.py            # PPO训练脚本
│   ├── config.py           # 超参数配置
│   └── callbacks.py        # 评估回调
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py         # 模型评估
└── export/
    ├── __init__.py
    └── export_model.py     # ONNX导出
```

### 核心实现

**`rl/envs/puzzle_env.py`** — `PuzzleSwapEnv(gym.Env)`:

```python
# Observation Space:
obs = {
    "grid": Box(0, n-1, shape=(n,), dtype=int32),        # 扁平化的拼图块索引
    "edge_costs": Box(0, inf, shape=(n, n, 2), float32),  # 预计算代价矩阵
}

# Action Space:
action_space = Discrete(n*(n-1)//2)  # 编码所有有效交换对

# Reward:
reward = (prev_cost - new_cost)          # 主信号：代价改善
       + 10.0 * (new_acc - prev_acc)     # 准确率提升奖励
       + 100.0 if solved else 0          # 完成奖励
       - 0.01                            # 步数惩罚

# Episode:
#   reset: 随机打乱拼图块
#   terminated: 所有块位置正确
#   truncated: 达到max_steps(默认500)
```

**`rl/training/train.py`**:
- 使用Stable Baselines3的PPO算法
- `MultiInputPolicy`处理Dict observation space
- TensorBoard日志
- 从6×6开始训练

**`rl/training/config.py`**:
```python
@dataclass
class TrainConfig:
    grid_size: int = 6
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    max_episode_steps: int = 500
    seed: int = 42
```

**`rl/evaluation/evaluate.py`**:
- 加载训练好的模型，在测试集上评估
- 指标：position_accuracy, neighbor_accuracy, 平均步数, 解决率

**`rl/export/export_model.py`**:
- PyTorch模型 → ONNX格式
- 验证导出模型输出一致性

### 新增依赖
```
# requirements.txt 追加
gymnasium>=0.29.0
stable-baselines3>=2.1.0
torch>=2.0.0
onnxruntime>=1.16.0
tensorboard>=2.14.0
```

### 验证
```bash
conda activate jigsaw
# 测试环境
python3 -m pytest tests/test_rl_env.py -v
# 训练
python3 -m rl.training.train --grid-size 6 --timesteps 100000
# 评估
python3 -m rl.evaluation.evaluate --model rl_models/final_model --grid-size 6
```

---

## Phase 3: 移动端迁移

### 目录结构
```
mobile_transfer/
├── __init__.py
├── state_extractor.py      # 截图 → RL observation
├── model_runner.py         # ONNX推理
└── executor.py             # 执行交换动作（桥接ios_auto）
```

### 核心实现

**`mobile_transfer/state_extractor.py`** — `ScreenStateExtractor`:
1. 复用 `jigsaw/roi_color.py` 提取拼图区域
2. 复用 `jigsaw/gap_splitter.py` 带间隙切分
3. 复用 `jigsaw/matcher.py` 构建代价矩阵
4. 输出与训练时一致的observation格式

**`mobile_transfer/model_runner.py`** — `MobileModelRunner`:
- ONNX Runtime推理
- 输入observation，输出action index

**`mobile_transfer/executor.py`** — `MobileExecutor`:
- 截图 → 提取状态 → 推理 → 执行交换
- 复用 `ios_auto/gesture.py::Gesture.swap_pieces()` 执行拖拽
- 主循环：capture → extract → predict → execute → sleep

### 域差距处理
- 训练时使用 `jigsaw/utils.py::degrade_observation()` 增加噪声/模糊/色彩偏移
- 使用 `gap_splitter.py` 处理真实截图中的间隙

### 验证
```bash
# 离线测试（不需要真机）
python3 -m mobile_transfer.executor --test-offline examples/IMG_1230.png --grid 6x6
```

---

## 实施顺序

1. **Phase 1**: `web/static/` (index.html, style.css, puzzle_game.js) + `web/server.py`
2. **Phase 2a**: `rl/envs/puzzle_env.py` + `tests/test_rl_env.py`
3. **Phase 2b**: `rl/training/train.py` + `config.py` + `callbacks.py`
4. **Phase 2c**: `rl/evaluation/evaluate.py` + `rl/export/export_model.py`
5. **Phase 3**: `mobile_transfer/` (state_extractor, model_runner, executor)

## 测试计划

| 阶段 | 测试内容 | 命令 |
|------|---------|------|
| Phase 1 | 浏览器手动玩6x6拼图 | `python3 -m web.server` |
| Phase 2 | 环境单元测试 | `python3 -m pytest tests/test_rl_env.py -v` |
| Phase 2 | 训练冒烟测试 | `python3 -m rl.training.train --grid-size 6 --timesteps 1000` |
| Phase 3 | 离线截图测试 | `python3 -m mobile_transfer.executor --test-offline` |
