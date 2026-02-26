# 项目名称：规则拼图还原（Square Jigsaw Puzzle Reconstruction）

---

# 🎯 项目目标

实现一个完整的规则拼图还原系统。

系统必须完成以下流程：

1. 读取输入图像
2. 将图像切分为 M×N 等大小方块
3. 随机打乱方块顺序
4. 基于边缘相似度进行拼图重排
5. 计算还原准确率
6. 可视化展示：
   - 原始图
   - 打乱图
   - 重建图

系统必须：

- 可重复（固定随机种子）
- 模块化
- 可测试
- 可扩展

---

# 📁 项目目录结构

必须严格按照以下结构实现：

project_root/
│
├── jigsaw/
│   ├── __init__.py
│   ├── splitter.py      # 图像切分
│   ├── matcher.py       # 边缘匹配计算
│   ├── solver.py        # 拼图求解算法
│   ├── evaluator.py     # 准确率计算
│   └── utils.py         # 工具函数
│
├── tests/
│   ├── test_solver.py
│   └── test_accuracy.py
│
├── demo.py
└── requirements.txt

---

# 🧠 算法要求

---

## 1️⃣ Patch 数据结构

每个拼图块必须包含：

- image: 图像 numpy 数组
- original_index: 原始位置索引
- edges:
  - top
  - bottom
  - left
  - right

边缘必须使用 numpy array 表示。

---

## 2️⃣ 边缘相似度计算

使用 L2 距离：

D(A.right, B.left) = sum((A - B)^2)

必须实现两个方向：

- RIGHT：A 在左，B 在右
- DOWN：A 在上，B 在下

可选：实现归一化版本。

---

## 3️⃣ 构建代价矩阵

构建：

cost[i][j][direction]

direction ∈ {RIGHT, DOWN}

必须预先计算完整代价矩阵。

---

## 4️⃣ 拼图求解策略

分两阶段实现：

---

### 第一阶段：贪心构建初始解

步骤：

1. 固定随机种子（42）
2. 选择一个随机拼图块作为左上角
3. 横向扩展填充第一行：
   - 每次选择 RIGHT 代价最小且未使用的块
4. 填充后续行：
   - 使用 DOWN 匹配约束

---

### 第二阶段：局部交换优化（可选但推荐实现）

实现局部优化：

- 随机交换两个拼图块
- 如果总代价下降则接受
- 迭代次数可配置（默认 1000 次）

---

# 📊 评估指标

必须实现：

1. 位置准确率

accuracy = 正确位置的块数 / 总块数

2. 邻居匹配准确率

统计相邻块是否与原图一致。

---

# 🧪 测试要求

---

## tests/test_solver.py

测试：

- 3x3 拼图
- 5x5 拼图

要求：

- 程序正常运行
- accuracy > 0.8（自然图像）

---

## tests/test_accuracy.py

必须测试：

1. 随机图像
2. 渐变图像
3. 自然图像

输出：

- accuracy
- 运行时间
- 总匹配代价

---

# 🖼 demo.py 要求

demo.py 必须实现：

1. 加载图像
2. 切分 5x5
3. 打乱
4. 重建
5. 输出准确率
6. 并排展示三张图

使用 matplotlib 可视化。

---

# ⚙️ 技术要求

- Python 3.10+
- 使用库：
  - numpy
  - opencv-python
  - matplotlib
  - pytest

禁止：

- 使用全局变量
- 写单文件混乱代码
- 硬编码参数

必须：

- 所有函数有 docstring
- 所有模块可单独测试
- 使用面向对象设计

---

# 🚀 性能要求

- 10x10 拼图运行时间 < 5 秒
- 内存 < 1GB

---

# 🔁 可复现性要求

- 固定 random seed = 42
- 所有随机行为必须可控

---

# 📌 扩展任务（加分项）

如时间允许，可实现：

1. Sobel 梯度边缘匹配
2. Beam Search 求解
3. 模拟退火优化
4. 不同算法性能对比

---

# 🔎 实现顺序（必须严格遵守）

Claude Code 必须按以下顺序实现：

1. splitter.py
2. matcher.py
3. solver.py
4. evaluator.py
5. demo.py
6. tests
7. 优化模块

每完成一个模块必须进行简单自测。

---

# ✅ 项目完成标准

满足以下条件才算完成：

- demo.py 可正常运行
- 所有 pytest 测试通过
- 5x5 拼图准确率 > 85%
- 代码结构清晰
- 无冗余代码
- 可读性良好