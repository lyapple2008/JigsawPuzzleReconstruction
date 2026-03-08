# 真实图像 Benchmark 使用指南

> 文档更新时间：2026-03-08

---

## 概述

`benchmarks/run_real_benchmark.py` 脚本用于在真实图像数据集上评估拼图还原算法的性能。与合成图像 benchmark 不同，该脚本支持从任意图像目录加载图片，并提供详细的统计分析。

---

## 测试指标

### 1. 位置准确率 (Position Accuracy)

衡量拼图块还原到原始位置的正确比例：

```
position_accuracy = 正确位置的块数 / 总块数
```

- **范围**：0% ~ 100%
- **越高越好**：100% 表示所有拼图块都回到了正确位置

### 2. 邻居匹配准确率 (Neighbor Accuracy)

衡量相邻拼图块匹配的正确比例：

```
neighbor_accuracy = 正确匹配的邻居对数 / 总邻居对数
```

- **范围**：0% ~ 100%
- **越高越好**：100% 表示所有相邻块都正确匹配

### 3. 总匹配代价 (Total Cost)

所有相邻块边缘差异的总和（L2 距离）：

```
total_cost = Σ D(patch[i].edge, patch[j].edge)
```

- **范围**：≥ 0
- **越低越好**：表示边缘匹配更紧密

### 4. 运行时间 (Runtime)

拼图还原算法执行的耗时：

- **单位**：秒
- **越低越好**：表示算法效率更高

---

## 统计量

对于多个图像的测试，脚本会计算以下统计量：

| 统计量 | 说明 |
|--------|------|
| Mean | 平均值 |
| Std | 标准差 |
| Min | 最小值 |
| Max | 最大值 |

---

## 使用方法

### 基本用法

```bash
# 激活 conda 环境
conda activate jigsaw

# 运行 benchmark
python benchmarks/run_real_benchmark.py \
    --dataset-path datasets/val2017-small \
    --grid-sizes 3 5 \
    --num-images 10
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset-path` | str | **必需** | 图像目录路径 |
| `--grid-sizes` | int[] | [3, 5, 8, 10] | 要测试的 grid 大小列表 |
| `--num-images` | int | 50 | 测试图像数量 |
| `--skip-images` | int | 0 | 跳过前 N 张图像 |
| `--image-size` | int | 300 | 图像 resize 大小 |
| `--solver` | str | default | 求解器选择 (default/gaps) |
| `--local-opt-iters` | int | 1000 | 局部优化迭代次数 |
| `--use-position-prior` | flag | False | 启用位置先验 |
| `--auto-position-prior` | flag | False | 自动启用位置先验 |
| `--seed` | int | 42 | 随机种子 |
| `--output-json` | str | None | JSON 报告输出路径 |
| `--output-csv` | str | None | CSV 报告输出路径 |
| `--verbose` | flag | False | 打印详细进度 |

---

## 示例

### 示例 1：测试 3x3 grid

```bash
python benchmarks/run_real_benchmark.py \
    --dataset-path datasets/val2017-small \
    --grid-sizes 3 \
    --num-images 5
```

输出：

```
============================================================
Real Image Benchmark Results
============================================================
Dataset: datasets/val2017-small
Images per grid: 5

Grid      Images   PosAcc(%)   NbrAcc(%)          Cost   Time(s)
----------------------------------------------------------------
3x3          5      100.00      100.00     693542.81     0.011

=== Statistics ===
Mean Position Accuracy: 100.00% ± 0.00%
Mean Neighbor Accuracy: 100.00% ± 0.00%
```

### 示例 2：测试多个 grid 大小

```bash
python benchmarks/run_real_benchmark.py \
    --dataset-path datasets/val2017-small \
    --grid-sizes 3 5 8 \
    --num-images 10 \
    --verbose
```

### 示例 3：保存报告

```bash
python benchmarks/run_real_benchmark.py \
    --dataset-path datasets/val2017-small \
    --grid-sizes 3 5 \
    --num-images 10 \
    --output-json results.json \
    --output-csv results.csv
```

### 示例 4：使用自己的图像目录

```bash
python benchmarks/run_real_benchmark.py \
    --dataset-path ./my_images \
    --grid-sizes 5 \
    --num-images 20 \
    --verbose
```

---

## 输出结果

### 终端输出

终端会打印一个格式化的表格，包含：
- Grid：grid 大小
- Images：测试图像数量
- PosAcc(%)：位置准确率
- NbrAcc(%)：邻居匹配准确率
- Cost：总匹配代价
- Time(s)：运行时间

### JSON 输出

```json
{
  "timestamp": "2026-03-08T16:25:55.546691",
  "dataset": {
    "path": "datasets/val2017-small",
    "num_images": 5
  },
  "config": {
    "grid_sizes": [3, 5],
    "num_images": 5,
    "image_size": 300,
    "solver": "default",
    "local_opt_iters": 1000
  },
  "summary": {
    "mean_position_accuracy": 0.8,
    "mean_neighbor_accuracy": 0.925,
    "mean_runtime": 0.029
  },
  "rows": [
    {
      "grid_size": 3,
      "num_images": 5,
      "position_accuracy_mean": 1.0,
      "position_accuracy_std": 0.0,
      "position_accuracy_min": 1.0,
      "position_accuracy_max": 1.0,
      "neighbor_accuracy_mean": 1.0,
      "neighbor_accuracy_std": 0.0,
      "total_cost_mean": 693542.81,
      "runtime_mean": 0.011
    }
  ],
  "image_results": [
    {
      "image_index": 0,
      "image_name": "000000000139.jpg",
      "grid_size": 3,
      "position_accuracy": 1.0,
      "neighbor_accuracy": 1.0,
      "total_cost": 251232.13,
      "runtime_seconds": 0.0109
    }
  ]
}
```

### CSV 输出

```csv
image_index,image_name,grid_size,position_accuracy,neighbor_accuracy,total_cost,runtime_seconds
0,000000000139.jpg,3,1.000000,1.000000,251232.13,0.0109
1,000000000285.jpg,3,1.000000,1.000000,544786.63,0.0109
...
```

---

## 数据集准备

### 使用 COCO 数据集

COCO val2017 是标准的测试数据集，约 5000 张图像。

#### 方式一：手动下载

```bash
# 下载 COCO val2017
python benchmarks/download_dataset.py --output datasets/coco
```

参数：
- `--output`: 输出目录（默认：datasets/coco）
- `--skip-annotations`: 跳过 annotations 下载（默认：True）
- `--verify`: 下载后验证数据集
- `--force`: 强制重新下载

#### 方式二：使用已有数据集

将图像放入一个目录即可：

```
datasets/
├── my_images/
│   ├── image1.jpg
│   ├── image2.png
│   └── image3.jpg
```

### 支持的图像格式

- JPG / JPEG
- PNG
- BMP
- WebP

---

## 性能参考

以下是在 COCO val2017-small（10 张图像）上的参考性能：

| Grid | 位置准确率 | 邻居准确率 | 运行时间 |
|------|------------|------------|----------|
| 3x3  | ~100%      | ~100%      | ~0.01s   |
| 5x5  | ~60-100%   | ~85-100%   | ~0.05s   |
| 8x8  | ~40-80%    | ~60-80%    | ~0.2s    |
| 10x10| ~30-60%    | ~50-70%    | ~0.5s    |

---

## 故障排除

### 问题 1：No images found

```
ValueError: No images found in /path/to/directory
```

**解决方案**：确保目录中存在支持的图像格式（jpg, png, bmp, webp）。

### 问题 2：图像数量不足

```
Warning: Only X images available, but requested Y
```

**解决方案**：减少 `--num-images` 或 `--skip-images` 参数值。

### 问题 3：内存不足

**解决方案**：减小 `--image-size` 参数（如 300 -> 150）或减少 `--num-images`。

---

## 相关文件

- `benchmarks/run_real_benchmark.py` - 主脚本
- `benchmarks/image_loader.py` - 图像加载器
- `benchmarks/download_dataset.py` - 数据集下载工具
- `benchmarks/run_benchmark.py` - 合成图像 benchmark（参考）
