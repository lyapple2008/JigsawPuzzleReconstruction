# 规则拼图还原（Squared Jigsaw Puzzle）解决方案调研

> 调研时间：2026-03-06
> 场景：矩形拼图块，大小一致，无旋转，位置打乱

---

## 问题定义

### 1.1 问题描述

**规则拼图还原**是计算机视觉领域的经典问题，具体定义为：

- **输入**：一张被分割成 M×N 个等大小矩形块的图像，这些矩形块的位置被打乱（但没有旋转）
- **输出**：恢复原始图像的排列顺序，重新拼接成完整的图像

### 1.2 数学形式化

设原始图像被分割为 M×N 个 patches，每个 patch 记为 P_i（i = 1, 2, ..., M×N），每个 patch 具有以下属性：
- `image`: 图像 numpy 数组
- `original_index`: 原始位置索引
- `edges`: 包含 top, bottom, left, right 四个方向的边缘特征

目标是找到一个排列 π，使得重新排列后的图像与原始图像一致。

### 1.3 关键挑战

1. **边缘相似度计算**：如何衡量两个 patch 边缘的匹配程度（L2 距离是最常用的方法）
2. **全局优化**：拼图是一个组合优化问题，需要找到全局最优解
3. **计算效率**：随着拼图块数量增加，搜索空间呈指数增长
4. **无旋转假设**：本项目假设拼图块没有旋转，简化了问题

### 1.4 评估指标

1. **位置准确率** (Position Accuracy)
   ```
   accuracy = 正确位置的块数 / 总块数
   ```

2. **邻居匹配准确率** (Neighbor Matching Accuracy)
   - 统计相邻块是否与原图一致

3. **总匹配代价** (Total Matching Cost)
   - 所有相邻块边缘差异的总和

---

## 1. Deepzzle（2020）

**基本原理**：使用深度学习预测碎片位置，然后通过图优化和最短路径算法进行全局重建。核心思想是将拼图重建问题转化为图论问题，利用神经网络预测相邻碎片之间的关系，再通过最短路径优化得到最终解。

**论文**：IEEE Trans Image Process 2020
**链接**：https://www.storkapp.me/pubpaper/31944956

---

## 2. JigsawGAN（2021）

**基本原理**：将GAN（生成对抗网络）用于拼图求解。通过生成器学习图像块的特征表示，判别器评估重建质量，实现端到端的拼图还原。

**论文**：arXiv:2101.07555
**链接**：https://arxiv.org/pdf/2101.07555.pdf

---

## 3. Prim最小生成树算法

**基本原理**：将每个图像块视为图节点，边缘相似度作为边权重。使用Prim算法构建最小生成树，从一个随机块开始，逐步选择与当前集合代价最小的相邻块进行扩展。

**特点**：
- 时间复杂度 O(N²)
- 实现简单，运行速度快

**GitHub**：
- https://github.com/hj2choi/fast_jigsaw_puzzle_solver
- https://github.com/bminaiev/jigsaw-puzzle-solver

---

## 4. 贪心算法 + 局部交换优化

**基本原理**：
- **阶段1（贪心构建）**：随机选择左上角块，横向依次选择RIGHT代价最小的块，纵向依次选择DOWN代价最小的块
- **阶段2（局部优化）**：随机交换两个块的位置，若总代价下降则接受

**GitHub**：https://github.com/prolleytroblems/jigsolving

---

## 5. CNN特征匹配方法

**基本原理**：使用卷积神经网络提取每个图像块的特征，计算块之间的特征相似度来确定正确的拼接顺序。

**GitHub**：https://github.com/shivaverma/Jigsaw-Solver

---

## 6. 图神经网络（GNN）方法

**基本原理**：将拼图块建模为图的节点，块之间的关系建模为边。使用GNN进行消息传递，学习块之间的空间关系，预测正确的排列顺序。

**相关论文**：
- Jigsaw Clustering for Unsupervised Visual Representation Learning (2021)
- https://arxiv.org/pdf/2104.00323.pdf

---

## 7. 自监督学习拼图求解

**基本原理**：将打乱的图像块作为输入，训练CNN预测正确的排列顺序。引入Context-Free Network (CFN)处理每个图像块独立学习特征。

**论文**：Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (ECCV 2016)

**GitHub**：https://github.com/kvananth/JigsawPuzzle-Mnist

---

## 8. 遗传算法/模拟退火优化

**基本原理**：使用进化算法优化拼图排列。通过选择、交叉、变异等操作迭代搜索最优解，配合代价函数评估重建质量。

**相关参考**：
- https://github.com/RePAIRProject/starter-pack/blob/main/papers.md

---

## 9. Harris角点 + 边缘匹配

**基本原理**：使用Harris角点检测提取图像块边缘特征，通过边缘匹配计算相似度，寻找最佳拼接位置。

**GitHub**：https://github.com/Iirana/jigsaw-puzzle-solver

---

## 10. Beam Search（束搜索）

**基本原理**：在每一步保留多个最优候选解，而不是只保留一个。通过限制搜索宽度来平衡计算效率和求解质量。

---

## 11. 积分不变量方法（2022）

**基本原理**：使用积分面积不变量进行形状匹配，结合优化过程聚合形状信息，适合无图像的非图形拼图。

**论文**：Application of Integral Invariants to Apictorial Jigsaw Puzzle Assembly (JMIV 2022)
**链接**：https://link.springer.com/10.1007/s10851-022-01120-z

---

## 12. 多体弹簧-质点系统（2024）

**基本原理**：将拼图块建模为质点，相邻关系建模为弹簧，使用层次化循环约束和分层重建过程进行求解。

**论文**：Pictorial and Apictorial Polygonal Jigsaw Puzzles (IJCV 2024)
**链接**：https://link.springer.com/article/10.1007/s11263-024-02033-7

---

## 综述论文

- **Jigsaw puzzle solving techniques and applications: a survey** (The Visual Computer, 2022)
- https://link.springer.com/10.1007/s00371-022-02598-9

---

## 方法对比

| 方法 | 复杂度 | 优点 | 缺点 |
|------|--------|------|------|
| Prim MST | O(N²) | 快速、简单 | 可能陷入局部最优 |
| 贪心+局部优化 | O(N²) | 实现简单 | 需多次迭代 |
| 深度学习 | 训练耗时 | 准确率高 | 需大量数据 |
| 遗传/模拟退火 | 可调 | 全局搜索能力 | 收敛较慢 |
| CNN特征匹配 | O(N²) | 特征鲁棒 | 需训练模型 |
| GNN | O(N²) | 关系建模强 | 实现复杂 |

---

## 推荐方案

对于本项目（5x5矩形拼图，无旋转），推荐以下方案：

1. **首选**：贪心算法 + 局部交换优化
   - 实现简单，效果良好
   - 符合CLAUDE.md中的要求

2. **备选**：Prim最小生成树
   - 速度快，适合大规模拼图

3. **进阶**：可尝试深度学习方法提升准确率
