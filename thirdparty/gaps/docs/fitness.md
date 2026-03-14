# gaps/fitness.py 模块文档

## 概述

该模块实现了拼图碎片之间的**不相似度度量（Dissimilarity Measure）**。通过计算相邻碎片边缘的颜色差异，评估两个碎片是否应该拼接在一起。

算法基于以下假设：原始图像中相邻的拼图块在边缘处具有相似的颜色，因此边缘像素的颜色差异应该最小化。

---

## dissimilarity_measure 函数

### 函数签名

```python
def dissimilarity_measure(first_piece, second_piece, orientation="LR", border_width=1):
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `first_piece` | Piece | 必填 | 第一个拼图块 |
| `second_piece` | Piece | 必填 | 第二个拼图块 |
| `orientation` | str | `"LR"` | 碎片排列方向 |
| `border_width` | int | `1` | 边缘像素宽度 |

#### orientation 参数

| 值 | 说明 | 图示 |
|----|------|------|
| `"LR"` | 左-右方向，first 在左，second 在右 | `| L | - | R |` |
| `"TD"` | 上-下方向，first 在上，second 在下 | `| T |` <br>`| D |` |

#### border_width 参数

| 值 | 说明 |
|----|------|
| `1` | 使用最外边缘的 1 像素（默认行为） |
| `>1` | 使用向内 N 像素的边缘 |

**注意**：`border_width` 会被限制在 `[1, min(rows, columns)]` 范围内。

---

### 算法原理

```
┌─────────────────────────────────────────────────────────────┐
│              dissimilarity_measure 计算流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: first_piece, second_piece, orientation, border_width │
│                                                             │
│  1. 获取碎片尺寸                                             │
│     rows, columns = piece.shape()                           │
│                                                             │
│  2. 根据 orientation 提取边缘像素                            │
│                                                             │
│     LR (左右):                                              │
│     ┌─────────┐     ┌─────────┐                            │
│     │         │     │         │                            │
│     │  first  │────▶│ second  │                            │
│     │         │     │         │                            │
│     └─────────┘     └─────────┘                            │
│  right_edge_of_   left_edge_of_                            │
│  first [:,-N:,:]  second [:,:N,:]                          │
│                                                             │
│     TD (上下):                                              │
│     ┌─────────┐                                             │
│     │  first  │                                             │
│     │         │                                             │
│     └─────────┘                                             │
│     bottom    ┌─────────┐                                   │
│     edge      │ second  │                                   │
│     [-N:, :]  │         │                                   │
│               └─────────┘                                   │
│                                                             │
│  3. 计算颜色差异                                             │
│     diff = left_edge - right_edge                           │
│                                                             │
│  4. 归一化和平方                                             │
│     squared = (diff / 255.0) ^ 2                            │
│                                                             │
│  5. 汇总所有通道和像素                                       │
│     total = sum(squared)                                    │
│                                                             │
│  6. 计算最终度量值                                           │
│     value = sqrt(total)                                     │
│                                                             │
│  输出: value (越小表示越相似)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 数学公式

对于两个相邻碎片 A 和 B，边缘宽度为 W（border_width），不相似度定义为：

$$D(A, B) = \sqrt{\sum_{c \in \{R,G,B\}} \sum_{i=1}^{W} \sum_{j=1}^{N} \left( \frac{A_{c}[i,j] - B_{c}[i,j]}{255} \right)^2 }$$

其中：
- $N$ 为边缘像素数量（行或列的像素数）
- $W$ 为 border_width
- $c$ 为颜色通道（R, G, B）

---

### 使用示例

#### 基本用法

```python
from gaps.fitness import dissimilarity_measure
from gaps.piece import Piece

# 加载碎片
p1 = Piece("piece_1.png")
p2 = Piece("piece_2.png")

# 计算左右方向的不相似度
score_lr = dissimilarity_measure(p1, p2, orientation="LR")

# 计算上下方向的不相似度
score_td = dissimilarity_measure(p1, p2, orientation="TD")

print(f"LR 不相似度: {score_lr}")
print(f"TD 不相似度: {score_td}")
```

#### 使用自定义边缘宽度

```python
# 使用最外边缘 1 像素（默认）
score_1 = dissimilarity_measure(p1, p2, orientation="LR", border_width=1)

# 使用向内 3 像素的边缘
score_3 = dissimilarity_measure(p1, p2, orientation="LR", border_width=3)

# 使用向内 5 像素的边缘
score_5 = dissimilarity_measure(p1, p2, orientation="LR", border_width=5)
```

---

### 边缘宽度选择建议

| 场景 | 推荐 border_width |
|------|-------------------|
| 碎片较小 (< 50px) | 1 |
| 碎片中等 (50-100px) | 1-3 |
| 碎片较大 (> 100px) | 3-5 |
| 边缘有明显噪点 | 增大 border_width 以平滑噪声 |
| 边缘有精细纹理 | 减小 border_width 以保持细节 |

---

### 性能考虑

- 时间复杂度：$O(N \times W)$，其中 N 是边缘像素总数，W 是 border_width
- 空间复杂度：$O(1)$
- 建议使用 `ImageAnalysis.analyze_image()` 预计算所有碎片对的不相似度，避免重复计算

---

## 相关模块

- [image_analysis.md](./image_analysis.md) - 不相似度缓存和最佳匹配查找
- [genetic_algorithm.md](./genetic_algorithm.md) - 使用不相似度进行拼图求解
- [individual.md](./individual.md) - 使用不相似度计算个体适应度
