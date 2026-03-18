# Jigsaw Puzzle Reconstruction

基于边缘相似度的规则拼图还原系统，支持本地图像还原和 iOS 设备自动化操作。

## 功能特性

- **图像切分**: 将图像均匀分割为 M×N 拼图块
- **边缘匹配**: 基于 L2 距离计算相邻图块的匹配代价
- **多种求解器**: 支持贪心求解、遗传算法求解、带间隙感知的求解
- **准确率评估**: 计算位置准确率和邻居匹配准确率
- **iOS 自动化**: 通过 WDA (WebDriverAgent) 控制 iOS 设备自动完成拼图

## 项目结构

```
.
├── jigsaw/                      # 核心拼图还原模块
│   ├── splitter.py              # 图像切分
│   ├── matcher.py              # 边缘匹配计算
│   ├── solver/                 # 求解器
│   │   ├── default_solver.py   # 贪心求解器
│   │   ├── gaps_solver.py      # 带间隙感知求解器
│   │   └── factory.py          # 求解器工厂
│   ├── evaluator.py            # 准确率评估
│   ├── roi_color.py            # 基于颜色的ROI提取
│   └── puzzle_roi.py           # 拼图区域检测
│
├── ios_auto/                   # iOS自动化模块
│   ├── connector.py            # WDA设备连接
│   ├── screenshot.py           # 截图获取
│   ├── gesture.py              # 触控操作
│   ├── planner.py              # 移动规划
│   └── automation.py           # 主自动化流程
│
├── demo.py                      # 演示脚本
├── reconstruct.py              # 拼图还原脚本
└── tests/                       # 测试用例
```

## 环境配置

```bash
conda create -n jigsaw python=3.10
conda activate jigsaw
pip install -r requirements.txt
```

## 快速开始

### 本地图像还原

```bash
conda activate jigsaw
python3 demo.py --image examples/IMG_0970.PNG --extract-roi
```

### 离线测试（无需设备）

```bash
python3 -m ios_auto.automation --test-offline examples/IMG_1230.png --grid 8x8
```

### iOS 设备自动化

1. 安装 WDA 环境（参考 `docs/ios-wda环境安装指南.md`）

2. 连接设备并运行自动化：

```bash
# 方式A: 使用 tidevice（推荐）
tidevice wdaproxy -p 8100 &
python3 -m ios_auto.automation --grid 8x8 --udid <YOUR_UDID>

# 方式B: 使用 facebook-wda + iproxy
iproxy 8100 8100 &
python3 -m ios_auto.automation --grid 8x8 --url http://localhost:8100
```

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--grid` | 网格大小 (如 8x8) | 8x8 |
| `--solver` | 求解器 (default/gaps) | gaps |
| `--border-width` | 边缘像素宽度 | 10 |
| `--robust-method` | 鲁棒方法 (mse/median/percentile/huber) | median |
| `--max-time` | 最大运行时间（秒） | 210 |
| `--output` | 输出目录 | ios_auto_output |

## 算法原理

### 边缘匹配

使用 L2 距离计算相邻图块的边缘相似度：

```
D(A.right, B.left) = sum((A - B)^2)
```

### 贪心求解

1. 固定随机种子（42）
2. 随机选择左上角起始块
3. 横向贪心扩展第一行
4. 逐行向下填充

### 遗传优化（可选）

- 构建代价矩阵
- 使用遗传算法寻找最优排列
- 支持局部交换优化

## 测试

```bash
conda activate jigsaw
python3 -m pytest tests/ -v
```

## 性能

- 10×10 拼图：< 5 秒
- 内存占用：< 1GB
- 支持 3×3 到 10×10 网格

## 技术栈

- Python 3.10+
- NumPy
- OpenCV
- Matplotlib
- facebook-wda / tidevice
- WebDriverAgent (iOS)
