# fitness优化

## 目前fitness计算存在的问题
目前fitness计算方式是，计算两个块相邻边的均方差来度量fitness，具体计算方式在 thirdparty/gaps/docs/fitness.md。
在实际场景中分块的边缘不是完全理想的状态，可能会有侵蚀，或者边缘附带着其它的内容，这种情况下会导致fitness的准确性大大下降。

## 目标优化实际场景下fitness的鲁棒性

目前想到的一种方案是，排除边缘一定宽度的内容，同时计算多条边缘均值的均方差，这样可以一定程度避免前面提到的实际场景下的一些问题。

---

## 备选优化方案

### 方案 A：统计鲁棒性改进（推荐优先实现）
- **中值代替均值**：对差值使用中位数，减少异常值影响
- **百分位数**：使用 25%-75% 百分位区间，忽略极端值
- **M-估计器**：使用类似 Huber 损失的鲁棒代价函数

### 方案 B：多尺度边缘匹配
- 使用不同宽度的 strip（当前 `strip_width`），取加权平均
- 自适应选择最佳宽度：尝试多个宽度，选择一致性最好的

### 方案 C：特征增强
- **梯度一致性**：除了颜色匹配，还匹配边缘处的梯度方向
- **纹理描述符**：使用 LBP（局部二值模式）描述边缘纹理
- **频域分析**：在频域中比较边缘的频率特征

### 方案 D：预处理/后处理
- **形态学去噪**：在匹配前对边缘进行形态学操作
- **一致性验证**：使用置信度机制，标记低置信度匹配

---

## 实现计划

### Phase 1: 方案 A - 统计鲁棒性改进
1. 修改 `jigsaw/matcher.py` 中的代价计算函数
2. 将 MSE 改为中值或百分位数
3. 添加参数选择（可配置使用哪种度量）

> 2026.03.17 测试median搭配border-width效果最好
``` python
python3 reconstruct.py --image examples/IMG_1230.png --grid 8x8 --extract-roi --solver gaps --robust-method median --border-width 10
```

### Phase 2: 方案 B - 多尺度边缘匹配（如 Phase 1 效果不足）
1. 实现多宽度 strip 计算
2. 添加自适应宽度选择逻辑

### Phase 3: 方案 C - 特征增强（如效果仍不足）
1. 添加梯度一致性损失
2. 添加 LBP 纹理描述符

---

## 相关文件
- `jigsaw/matcher.py` - 主要修改目标
- `thirdparty/gaps/gaps/fitness.py` - gaps 库的 fitness 实现
- `thirdparty/gaps/docs/fitness.md` - fitness 文档
