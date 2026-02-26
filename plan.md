# 拼图还原算法优化计划（面向 8x8 / 10x10）

## 1. 问题现状
- 当前方法在 `5x5` 正确率尚可。
- 提升到 `8x8` 或 `10x10` 后正确率明显下降。
- 根因：组合空间快速增大，`边缘L2 + 贪心 + 随机交换` 容易陷入局部最优。

## 2. 最新相关算法方向（到 2025）
1. 生成式全局先验 + 一对一分配
- 代表：GANzzle (2022/2025)
- 思路：先估计整图全局布局，再做匈牙利分配，提升大规模拼图稳定性。

2. 扩散/Transformer 位置生成
- 代表：JPDVT (CVPR 2024)
- 思路：直接生成 patch 位置表示，可支持缺块；文中含带缝隙设定。

3. 坐标回归替代排列分类
- 代表：FCViT (2025)
- 思路：回归每块 `(x,y)` 位置，缓解 `N!` 级别排列复杂度。

4. 全局一致性约束（经典强基线）
- 代表：Hierarchical Loop Constraints (TPAMI 2019), Paikin & Tal (CVPR 2015)
- 思路：通过环一致性与组件生长增强大拼图全局正确性。

## 3. 面向当前项目的分阶段优化路线

### P0：建立可靠评测基线（1天）
- 固定随机种子。
- 建立 `5x5 / 8x8 / 10x10` 三档 benchmark。
- 统一记录：
  - `position accuracy`
  - `neighbor accuracy`
  - `runtime`

目标：确保后续每次优化都有可比较、可复现实验结果。

### P1：无训练快速增益（2-4天）
1. 升级匹配代价（`matcher`）
- 单行/单列边缘 -> 多像素条带（2-4 像素）
- L2 颜色差 + 梯度差（可引入 MGC 思路）
- 引入 best-buddy 置信机制

2. 升级搜索策略（`solver`）
- 单次贪心 -> 多起点 beam search（保留 top-k）
- 随机交换 -> 2-opt/3-opt + 模拟退火或 tabu

预期：对 `8x8` 成功率提升最明显。

### P2：图优化与全局结构一致性（4-7天）
- 基于高置信邻接构图并形成组件。
- 组件级合并，避免局部错误快速扩散。
- 引入 loop consistency 过滤冲突边。

预期：显著改善 `10x10` 场景的稳定性。

### P3：轻量学习先验（1-2周）
- 训练轻量 ViT/CNN 坐标回归头，预测每块位置先验。
- 目标函数融合：`edge_cost + λ * position_prior`。

预期：进一步缩小搜索空间，提升大规模拼图成功率上限。

### P4：面向“有缝隙”图像的专项增强（可并行）
- 数据增强：随机 gap 宽度。
- 匹配时忽略边缘 1-2 像素，降低缝隙残留干扰。

## 4. 建议的近期执行顺序（最小闭环）
1. 先完成 P1（无需训练，收益快）。
2. 在 `tests` 中新增 `8x8 / 10x10` 回归测试。
3. 每次提交输出统一 benchmark 表格，持续追踪收益。

## 5. 参考资料
- JPDVT, CVPR 2024
  - https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.html
  - https://arxiv.org/abs/2404.07292
- FCViT, Expert Systems with Applications 2025
  - https://doi.org/10.1016/j.eswa.2025.126776
- GANzzle, Pattern Recognition Letters 2025
  - https://doi.org/10.1016/j.patrec.2024.11.010
- Hierarchical Loop Constraints, TPAMI 2019
  - https://pubmed.ncbi.nlm.nih.gov/30028692/
- Paikin & Tal, CVPR 2015
  - https://openaccess.thecvf.com/content_cvpr_2015/html/Paikin_Solving_Multiple_Square_2015_CVPR_paper.html
- Cho et al., CVPR 2010
  - https://people.csail.mit.edu/taegsang/JigsawPuzzle.html
