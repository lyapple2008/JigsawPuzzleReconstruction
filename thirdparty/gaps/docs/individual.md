# gaps/individual.py 模块文档

## 概述

`Individual` 类表示遗传算法中的**一个个体**（即拼图的一种可能排列方式）。每个个体代表拼图碎片的**一种随机排布**，是遗传算法进化的基本单位。

---

## Individual 类

### 函数签名

```python
class Individual(object):
    def __init__(self, pieces, rows, columns, shuffle=True):
```

### 作用

将打乱的拼图碎片封装为一个个体，个体包含：
- 碎片的排列顺序
- 适应度计算
- 与其他个体交互的接口

---

## 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `pieces` | List[Piece] | 碎片列表（打乱后的顺序） |
| `rows` | int | 行数 |
| `columns` | int | 列数 |
| `_fitness` | float | 适应度缓存（懒计算） |
| `_piece_mapping` | Dict | 碎片ID到索引的映射 |

---

## 关键方法

### __init__ - 初始化

```python
def __init__(self, pieces, rows, columns, shuffle=True):
```

- 如果 `shuffle=True`，随机打碎碎片顺序
- 构建 `_piece_mapping` 字典，快速通过ID查找碎片位置

```python
# 映射示例
_piece_mapping = {
    0: 5,   # ID=0 的碎片在第5个位置
    1: 2,   # ID=1 的碎片在第2个位置
    ...
}
```

### __getitem__ - 索引访问

```python
def __getitem__(self, key):
    return self.pieces[key * self.columns : (key + 1) * self.columns]
```

按**行**获取碎片，支持 `individual[row]` 语法。

```python
# 获取第2行的所有碎片
row_2 = individual[1]  # index=1 表示第2行
```

### fitness - 适应度属性

```python
@property
def fitness(self):
```

**核心评估函数**，计算个体的适应度值。

#### 计算逻辑

```
适应度 = FITNESS_FACTOR / (1/FITNESS_FACTOR + 所有相邻碎片不相似度之和)
```

具体步骤：
1. 遍历所有**水平相邻**的碎片对，计算 "LR" 方向不相似度
2. 遍历所有**垂直相邻**的碎片对，计算 "TD" 方向不相似度
3. 返回适应度值

#### 适应度含义

| 情况 | 适应度 |
|------|--------|
| 相邻碎片完全不匹配 | 接近 0 |
| 相邻碎片完全匹配 | 接近 FITNESS_FACTOR (1000) |

适应度越高，表示解越接近正确。

#### 懒计算

```python
if self._fitness is None:
    # 计算...
    self._fitness = ...
return self._fitness
```

首次访问时计算并缓存，避免重复计算。

### piece_by_id - 按ID获取碎片

```python
def piece_by_id(self, identifier):
    return self.pieces[self._piece_mapping[identifier]]
```

根据碎片ID快速获取该碎片对象。

```python
# 获取ID为5的碎片
piece = individual.piece_by_id(5)
```

### to_image - 转换为图像

```python
def to_image(self):
    pieces = [piece.image for piece in self.pieces]
    return utils.assemble_image(pieces, self.rows, self.columns)
```

将个体的碎片排列**转换为可显示的图像**，用于可视化结果。

### edge - 获取相邻碎片ID

```python
def edge(self, piece_id, orientation):
```

获取指定碎片在指定方向上的**相邻碎片ID**。

| 参数 | 说明 |
|------|------|
| `piece_id` | 目标碎片的ID |
| `orientation` | 方向："T"(上)、"R"(右)、"D"(下)、"L"(左) |

```python
# 获取碎片5的右边相邻碎片ID
right_neighbor = individual.edge(5, "R")
```

#### 边界检查

方法内部会检查边界：
- 如果碎片在最左边，`L` 返回 None
- 如果碎片在最右边，`R` 返回 None
- 以此类推...

---

## 使用示例

```python
from gaps.individual import Individual
from gaps.utils import flatten_image

# 分割图像
pieces, rows, columns = flatten_image(image, piece_size=32, indexed=True)

# 创建个体（自动打乱）
individual = Individual(pieces, rows, columns)

# 获取适应度
score = individual.fitness  # 自动计算并缓存

# 查看第2行
row_2 = individual[1]

# 获取碎片5的右边相邻碎片
right = individual.edge(5, "R")

# 转换为图像查看
img = individual.to_image()
```

---

## 在遗传算法中的作用

| 阶段 | 作用 |
|------|------|
| 初始化 | 创建随机排列的个体种群 |
| 选择 | 根据 fitness 选优秀的个体 |
| 交叉 | 组合两个个体的碎片顺序 |
| 变异 | 随机打乱部分碎片顺序 |
| 评估 | fitness 作为选择依据 |
