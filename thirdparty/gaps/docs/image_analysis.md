# gaps/image_analysis.py 模块文档

## 概述

该模块实现了图像拼图的**不相似度缓存机制**。通过预计算所有碎片之间的匹配程度，避免在遗传算法迭代过程中重复计算，大幅提升性能。

---

## ImageAnalysis 类

### 类属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `dissimilarity_measures` | Dict[Tuple, Dict[str, float]] | 缓存碎片对之间的不相似度值 |
| `best_match_table` | Dict[int, Dict[str, List[Tuple[int, float]]]] | 每个碎片的四个边缘最佳匹配列表 |

---

## analyze_image 函数

### 函数签名

```python
@classmethod
def analyze_image(cls, pieces):
```

### 作用

**预计算并缓存**所有碎片之间的不相似度度量，并为每个碎片的四个边缘（上下左右）建立最佳匹配查找表。

### 处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                   analyze_image 流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 初始化 best_match_table                                 │
│     ┌─────────────────────────────────────────────────┐     │
│     │ piece.id -> {"T": [], "R": [], "D": [], "L": []}│    │
│     └─────────────────────────────────────────────────┘     │
│     为每个碎片初始化四个方向的空列表                        │
│                                                             │
│  2. 双重循环遍历所有碎片对                                  │
│     ┌─────────────────────────────────────────────────┐     │
│     │ for first in range(n):                         │     │
│     │   for second in range(first+1, n):             │     │
│     │     # 处理 LR (左右) 方向                       │     │
│     │     # 处理 TD (上下) 方向                       │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  3. 计算不相似度                                            │
│     - 调用 dissimilarity_measure() 计算两个碎片之间的差异   │
│     - 存储到 dissimilarity_measures 缓存                   │
│     - 更新 best_match_table                                 │
│                                                             │
│  4. 排序最佳匹配                                            │
│     - 对每个碎片的四个方向列表按相似度排序                   │
│     - 相似度越低（差异越小）排在越前面                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 详细步骤

#### 步骤 1: 初始化查找表

```python
for piece in pieces:
    cls.best_match_table[piece.id] = {"T": [], "R": [], "D": [], "L": []}
```

为每个碎片初始化四个方向的列表：
- **T** (Top): 上边缘最佳匹配
- **R** (Right): 右边缘最佳匹配
- **D** (Down): 下边缘最佳匹配
- **L** (Left): 左边缘最佳匹配

#### 步骤 2: 遍历碎片对

```python
for first in range(iterations):
    for second in range(first + 1, len(pieces)):
        for orientation in ["LR", "TD"]:
            update_best_match_table(...)
```

- 外层循环：选择第一个碎片
- 内层循环：选择第二个碎片（避免重复计算）
- 方向：`LR`（左右关系）、`TD`（上下关系）

#### 步骤 3: 计算并缓存不相似度

```python
measure = dissimilarity_measure(first_piece, second_piece, orientation)
cls.put_dissimilarity((first_id, second_id), orientation, measure)
```

调用 `dissimilarity_measure()` 计算两个碎片在指定方向上的不相似度，然后：
1. 存入 `dissimilarity_measures` 缓存
2. 更新 `best_match_table`

#### 步骤 4: 排序

```python
for piece in pieces:
    for orientation in ["T", "L", "R", "D"]:
        cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])
```

对每个方向的匹配列表按不相似度排序，差异最小的排在第一位。

---

## 其他函数

### put_dissimilarity

```python
@classmethod
def put_dissimilarity(cls, ids, orientation, value):
```

将计算出的不相似度值存入缓存字典。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ids` | Tuple | 两个碎片的 ID 元组 |
| `orientation` | str | 方向，"LR" 或 "TD" |
| `value` | float | 不相似度值 |

### get_dissimilarity

```python
@classmethod
def get_dissimilarity(cls, ids, orientation):
```

从缓存中获取不相似度值，避免重复计算。

### best_match

```python
@classmethod
def best_match(cls, piece, orientation):
```

返回指定碎片在指定方向上的最佳匹配碎片的 ID。

---

## 性能优化

该类的核心作用是**空间换时间**：

| 优化项 | 说明 |
|--------|------|
| 缓存机制 | 避免每次迭代都计算不相似度 |
| 预排序 | best_match_table 已排序，快速获取最佳匹配 |
| 双向存储 | LR 和 TD 方向信息双向存储，查询方便 |

---

## 不相似度度量

`dissimilarity_measure()` 函数的具体实现需参考 `gaps/fitness.py`，通常基于：
- 相邻碎片边缘的颜色差异
- 边缘像素的 RGB 值对比
- 差异越小，表示越可能是正确匹配
