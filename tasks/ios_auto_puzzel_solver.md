# iOS自动拼图还原 - 方案设计

## 需求概述

- **目标**：实现一个在电脑端运行，操控iOS手机，自动完成拼图还原小游戏的程序
- **平台**：iOS (使用WDA通过USB控制)
- **游戏类型**：拼图还原（Jigsaw），拖动图块到目标位置，与目标位置图块交换
- **时间限制**：3分半
- **网格大小**：8x8
- **流程**：截图 → reconstruct.py还原 → 根据还原位置移动图块

---

## 已有的实现

- `reconstruct.py`: 从打乱图像还原拼图，输出solved_grid
- `jigsaw/` 模块: 包含splitter, solver, matcher, puzzle_roi等

---

## 功能模块设计

### 1. WDA环境安装与连接

#### 1.1 安装依赖
```bash
conda activate jigsaw
pip install facebook-wda
pip install tidevice  # USB直连需要
```

#### 1.2 USB直连方式
使用 `tidevice` 或 `iproxy` + WDA组合：
- **方案A**: tidevice（推荐，更简单）
- **方案B**: WDA + iproxy

### 2. 手机控制模块 (ios_auto/)

```
ios_auto/
├── __init__.py
├── connector.py     # USB连接管理 (WDA)
├── screenshot.py    # 截图获取
├── gesture.py       # 拖拽操作
├── planner.py       # 移动规划
└── automation.py    # 主自动化流程
```

#### 2.1 连接管理
- 使用tidevice连接iOS设备
- 启动WDA服务
- 维持session

#### 2.2 截图
- 通过WDA获取屏幕截图
- 保存到临时文件供reconstruct.py使用

#### 2.3 触控操作
- 拖拽操作：gesture.swipe
- 坐标转换：屏幕坐标 ↔ 图像坐标

### 3. 移动规划模块 (planner.py)

#### 3.1 问题建模
- 当前状态：位置(i,j)上的图块索引
- 目标状态：solved_grid（每个位置的正确图块）
- 操作：交换两个位置的内容

#### 3.2 贪心移动策略
```
移动序列生成（从还原目标倒序）:
1. 从最后一个位置开始向前遍历
2. 如果当前位置的块不对，将其移动到正确位置
3. 正确位置的块被交换了出来，继续处理
4. 重复直到完成
```

### 4. 自动化流程 (automation.py)

```
主流程:
1. 连接iOS设备
2. 截图获取当前拼图状态
3. 调用reconstruct.py还原（输出solved_grid）
4. 计算移动序列
5. 执行移动（倒序执行）
6. 重复直到还原完成或超时（3分半）
```

---

## WDA安装指南

### 方式一：使用 tidevice（推荐USB直连）

```bash
# 1. 安装tidevice
conda activate jigsaw
pip install tidevice

# 2. 查看已连接设备
tidevice list

# 3. 启动WDA（后台运行）
tidevice wda &
# 或
tidevice devicelist  # 查看设备UDID
tidevice -u <UDID> wdaproxy -t 8100 &
```

### 方式二：使用 WDA + iproxy

```bash
# 1. 安装facebook-wda
conda activate jigsaw
pip install facebook-wda

# 2. 安装iproxy（需要先安装Node.js）
npm install -g iproxy

# 3. 在iOS设备上安装WebDriverAgent
# （需要Xcode自行编译WDA）

# 4. 启动iproxy转发
iproxy 8100 8100

# 5. 连接WDA
python3 -c "import wda; c = wda.Client('http://localhost:8100')"
```

### 验证WDA可用

```bash
python3 -c "
import wda
c = wda.Client('http://localhost:8100')  # 或使用USB: wda.USBClient(udid='...')
print(c.status())
screenshot = c.screenshot()
screenshot.save('test.png')
print('Screenshot saved!')
"
```

---

## 使用方法

### 测试离线模式（无需设备）

```bash
conda activate jigsaw
python3 -m ios_auto.automation --test-offline examples/IMG_1230.png --grid 8x8
```

### 运行自动化（需要设备）

```bash
# 启动WDA服务
tidevice wdaproxy -t 8100 &

# 运行自动化
python3 -m ios_auto.automation --grid 8x8 --max-time 210
```

### 可选参数

- `--grid`: 网格大小 (默认 8x8)
- `--solver`: 求解器 (default 或 gaps)
- `--max-time`: 最大运行时间（秒，默认210）
- `--interval`: 检查间隔（秒，默认5）
- `--output`: 输出目录

---

## 实施步骤

### Step 1: 安装WDA依赖（推荐tidevice）
```bash
conda activate jigsaw
pip install tidevice
```

### Step 2: 测试设备连接
```bash
tidevice list
```

### Step 3: 启动WDA服务
```bash
# 方式A: 使用tidevice内置的wdaproxy
tidevice wdaproxy -t 8100 &
```

### Step 4: 实现截图功能
- 测试WDA截图

### Step 5: 实现拖拽功能

### Step 6: 实现移动规划

### Step 7: 集成测试

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `reconstruct.py` | 现有，还原算法 |
| `jigsaw/puzzle_roi.py` | 现有，ROI提取 |
| `ios_auto/` | 新增，设备控制 |

---

## 实现状态

✅ 已完成:
- ios_auto/connector.py - 设备连接
- ios_auto/screenshot.py - 截图
- ios_auto/gesture.py - 触控操作
- ios_auto/planner.py - 移动规划
- ios_auto/automation.py - 主流程
- 离线测试通过
