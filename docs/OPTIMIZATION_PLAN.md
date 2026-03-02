# 拼图还原算法优化计划（基于 SOTA 调研）

本文档基于 2024–2025 年文献与开源工作，梳理当前项目与 SOTA 的差距，并给出分阶段优化计划。在 conda 环境 `jigsaw` 下执行相关实验与测试。

---

## 一、当前项目状态简要

| 模块 | 现状 |
|------|------|
| **ROI 提取** | 支持背景颜色（默认）与网格线两种方式，已集成 `jigsaw.roi_color` |
| **边缘匹配** | L2 距离 + 条带/梯度/归一化加权（`EdgeMatcher`），无学习 |
| **求解器** | 贪心 + 多起点/Beam 初始解 + 局部交换 + 可选模拟退火、位置先验（`JigsawSolver`） |
| **评估** | 位置准确率、邻居准确率、总代价 |

---

## 二、SOTA 与主流方向（2024–2025）

### 2.1 学习式兼容度（替代手工 L2）

- **DNN-Buddies**：用 CNN 以两条边为输入，预测是否应为邻居；仅像素信息、无手工特征，显著提升求解精度。
- **启示**：在现有 `EdgeMatcher` 上增加「学习式兼容度」分支（或替换 L2），用成对边数据训练二分类/回归模型，再接入当前 solver。

### 2.2 Vision Transformer 与全局建模

- **ViT + 边缘编码**：将拼图视为「片段判别 + 放置」两步；边缘编码器 + Transformer 做排列学习，对**边界腐蚀（eroded boundaries）**鲁棒。
- **启示**：中长期可做「基于 Transformer 的片段级表示 + 放置网络」，适合有腐蚀、大尺寸、需要全局一致性的场景。

### 2.3 生成式与“心理图像”

- **GANzzle++**：用 GAN / Slot Attention / ViT 生成完整图的近似（“心理图像”），再通过 1-to-1 分配（如 Hungarian + attention）做片段到位置的匹配，局部到全局。
- **启示**：可作为「全局一致性」的另一条路线：先估整体图再反推布局，与当前「局部兼容度 + 搜索」互补。

### 2.4 强化学习与搜索

- **Alphazzle**：深度强化学习 + 蒙特卡洛树搜索，在指数级解空间中做有导向搜索。
- **SD²RL**：Siamese 判别网络 + DQN，建模片段间成对关系（水平/垂直），在**大腐蚀缝隙**上表现好。
- **启示**：现有 solver 已有 beam、多起点、局部交换；可进一步引入 MCTS 或 RL 策略，用于引导扩展顺序或交换策略。

### 2.5 传统优化与鲁棒性

- **Sequential Monte Carlo (SMC)**：最大权子图 + 约束，自适应排列顺序，相比 loopy belief propagation 等有数倍准确率提升。
- **混合框架**：CNN 兼容度 + 遗传算法等元启发式，在**真实破损拼图（如葡萄牙瓷砖）**上达到 SOTA。
- **启示**：在保留当前图结构（成对代价 + 网格）前提下，可引入 SMC、遗传算法或更复杂的 MCMC，提升对噪声/腐蚀的鲁棒性。

### 2.6 多模态与语义

- **VLHSA（Vision-Language）**：文本描述 + 视觉特征，层次化语义对齐，在带腐蚀缝隙的拼图上比纯视觉方法提升约 14.2 个百分点。
- **启示**：若有标题/描述等文本，可增加「语义一致性」项，用于约束放置或重排。

---

## 三、优化计划（分阶段、可落地）

### 阶段 1：无/轻学习，提升现有管线（优先）

目标：不引入深度学习依赖，在现有 `matcher` + `solver` 上提效。

| 序号 | 内容 | 说明 |
|------|------|------|
| 1.1 | **Sobel/梯度边缘兼容度** | 在 `EdgeMatcher` 中已有梯度权重；可单独暴露「仅梯度」或「梯度+颜色」组合，并做消融与参数搜索。 |
| 1.2 | **模拟退火超参与调度** | 当前已有 `sa_initial_temp_ratio`、`sa_cooling`；系统化做退火调度（线性/指数/自适应）与迭代预算，并写进 benchmark。 |
| 1.3 | **Beam 宽度与候选池** | 对 5×5、7×9、10×10 做 `beam_width`、`beam_candidate_pool` 的网格搜索，记录准确率与耗时。 |
| 1.4 | **腐蚀/缝隙鲁棒性** | 与 `gap_splitter` 配合，在「带缝隙/腐蚀」的合成或真实数据上测当前 matcher+solver，建立 baseline 指标（位置/邻居准确率）。 |
| 1.5 | **多起点与早停** | 已有 `use_multi_start`、`max_start_pieces`；可加早停（如连续 N 次无改进则停）与时间上限，便于大拼图。 |

交付：配置与脚本可复现、文档记录各策略对准确率/时间的影响。

---

### 阶段 2：学习式兼容度（替代/增强 L2）

目标：用「成对边 → 兼容度」的小模型替代或加权现有 L2，保持与现有 solver 接口一致。

| 序号 | 内容 | 说明 |
|------|------|------|
| 2.1 | **数据管线** | 从正确拼好的图中切块、打乱，生成「边 A–边 B + 是否邻居」数据集；支持 4 个方向与数据增强。 |
| 2.2 | **轻量 CNN 二分类器** | 输入：两条边的小图（或 1D 特征）；输出：是否应为邻居（或兼容度标量）。与 DNN-Buddies 思路一致，便于复现。 |
| 2.3 | **与 EdgeMatcher 集成** | 新增 `LearnedMatcher` 或 `EdgeMatcher(use_learned=True)`：内部调用小模型得到兼容度，可与现有 L2 加权融合（如 `alpha * L2 + (1-alpha) * learned`）。 |
| 2.4 | **评估与消融** | 在同一数据集上对比：仅 L2、仅学习、L2+学习；报告位置/邻居准确率与推理时间。 |

交付：训练脚本、模型格式、与现有 solver 的对接方式及简要实验报告。

---

### 阶段 3：搜索与全局优化增强（可选）

目标：在现有图结构上加强搜索与全局一致性，不改变「网格 + 成对代价」的设定。

| 序号 | 内容 | 说明 |
|------|------|------|
| 3.1 | **遗传算法 / 进化策略** | 种群 = 若干完整排列；适应度 = 总兼容度 + 结构惩罚；交叉、变异、选择；与当前局部交换或 SA 结合做后处理。 |
| 3.2 | **MCTS 引导扩展** | 在 beam 或贪心扩展时，用轻量 MCTS 选择「下一步放哪块」以提升长期回报（如预测整行/整图代价）。 |
| 3.3 | **SMC 式排列采样** | 参考 SMC 思路：维护若干排列粒子，按兼容度加权、重采样、局部扰动，迭代逼近最大权合法排列。 |

交付：可选开关（如 `--solver genetic`）、benchmark 对比与可复现脚本。

---

### 阶段 4：深度全局模型（中长期）

目标：面向「高精度、大尺寸、强腐蚀」场景，引入 ViT/生成式等 SOTA 思路。

| 序号 | 内容 | 说明 |
|------|------|------|
| 4.1 | **片段级 ViT 编码器** | 每个 patch 用 ViT 编码为向量；用边缘 crop 或专用边缘编码器增强对缝隙的鲁棒性。 |
| 4.2 | **放置网络或 Hungarian 分配** | 基于编码相似度或「位置–片段」得分矩阵，用 Hungarian 或小网络做 1-to-1 分配。 |
| 4.3 | **生成式辅助（可选）** | 若资源允许，可尝试 Slot Attention / 轻量 GAN 生成「完整图」再反推布局，与现有 solver 做集成或对比。 |

交付：模块化设计（编码器 / 分配器 / 数据格式）、训练与推理脚本、与现有 pipeline 的对接文档。

---

## 四、实施优先级建议

1. **先做阶段 1**：不增加新依赖，直接提升现有 matcher/solver 的鲁棒性与可调性，并建立稳定 benchmark（含带缝隙/腐蚀数据）。
2. **再做阶段 2**：学习式兼容度与现有 L2 融合，收益/实现成本比高，且与现有架构兼容。
3. **阶段 3、4**：按需求与资源选择；阶段 3 可作为阶段 2 的「搜索侧」补充，阶段 4 适合作为独立研究方向或新分支。

---

## 五、参考文献与链接（概要）

- Vision Transformers for jigsaw with edge encoding, robustness to erosion (e.g. Pattern Analysis and Applications, 2025).
- GANzzle++: Generative approaches, local-to-global assignment (Durham Repository).
- DNN-Buddies: CNN-based compatibility metric (Springer LNCS; Bar-Ilan).
- Hybrid CNN + genetic algorithms for real-world reconstruction (e.g. IEEE, arxiv).
- VLHSA: Vision-language hierarchical semantic alignment (2025).
- Alphazzle: Deep MCTS for jigsaw (paperswithcode).
- Sequential Monte Carlo for maximum weight subgraphs (IJCV / Springer).
- SD²RL: Siamese-Discriminant DRL for large eroded gaps (AAAI).

（具体引用格式与 URL 可在定稿时按项目规范补全。）

---

## 六、与现有代码的对应关系

| 计划项 | 当前代码位置 |
|--------|----------------|
| 梯度/Sobel、退火、Beam | `jigsaw/matcher.py`（strip/权重）, `jigsaw/solver.py`（SolverConfig, _local_swap_optimize, beam） |
| 腐蚀/缝隙 | `jigsaw/gap_splitter.py`, `reconstruct.py` |
| 学习式兼容度 | 新增 `jigsaw/learned_matcher.py` 或扩展现有 `EdgeMatcher` |
| 评估与 benchmark | `jigsaw/evaluator.py`, `benchmarks/run_benchmark.py`, `tests/test_accuracy.py` |

所有新实验与脚本均在 conda 环境 `jigsaw` 下运行，并在文档或脚本内注明依赖与复现步骤。
