# gaps/selection.py 模块文档

## 概述

该模块实现了**轮盘赌选择算法**（Roulette Wheel Selection），用于遗传算法中的**父母选择**阶段。优秀个体有更高概率被选为父母，类似于自然界"适者生存"的机制。

---

## roulette_selection 函数

### 函数签名

```python
def roulette_selection(population, elites=4):
```

### 作用

从种群中选择父母个体对，用于后续的交叉操作。适应度越高的个体，被选中的概率越大。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `population` | List[Individual] | 必填 | 种群列表 |
| `elites` | int | 4 | 精英个体数量（不参与选择，直接保留） |

### 返回值

返回列表，每个元素是一个元组 `(first_parent, second_parent)`，表示一对父母。

```python
# 返回值示例
[(parent1_a, parent1_b), (parent2_a, parent2_b), ...]
```

---

## 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│              roulette_selection 流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 提取适应度值                                            │
│     fitness_values = [ind1.fitness, ind2.fitness, ...]     │
│                                                             │
│  2. 构建累积概率区间                                         │
│     ┌─────────────────────────────────────────────────┐     │
│     │ fitness:    [10,  30,  50,  10]                 │     │
│     │ cumulative: [10,  40,  90, 100]  ← 区间边界    │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  3. 轮盘赌选择 (select_individual)                          │
│     - 随机生成 [0, 总和] 之间的数                           │
│     - 用二分查找找到落在哪个区间                             │
│     - 该区间对应的个体被选中                                 │
│                                                             │
│  4. 重复选择，生成父母对                                     │
│     - 选择 (种群大小 - elites) 对父母                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 详细实现

### 步骤 1: 提取适应度

```python
fitness_values = [individual.fitness for individual in population]
```

获取每个个体的适应度值。

### 步骤 2: 构建累积概率区间

```python
probability_intervals = [
    sum(fitness_values[: i + 1]) for i in range(len(fitness_values))
]
```

将适应度转换为累积区间：

```
个体索引:     0     1     2     3
适应度:      10    30    50    10
            ┌────┬────┬────┬────┐
            │ 10 │ 30 │ 50 │ 10 │
            └────┴────┴────┴────┘
累积区间:    [0,10) [10,40) [40,90) [90,100)
```

### 步骤 3: 选择个体

```python
def select_individual():
    random_select = random.uniform(0, probability_intervals[-1])
    selected_index = bisect.bisect_left(probability_intervals, random_select)
    return population[selected_index]
```

- 随机生成一个数
- 用二分查找快速找到该数落在哪个区间
- 该区间对应的个体被选中

### 步骤 4: 生成父母对

```python
selected = []
for i in range(len(population) - elites):
    first, second = select_individual(), select_individual()
    selected.append((first, second))
```

选择 `种群大小 - elites` 对父母（因为精英个体不参与选择，直接保留）。

---

## 二分查找优化

使用 `bisect.bisect_left` 实现 O(log n) 的查找，比线性遍历快。

```python
import bisect

# 示例
intervals = [10, 40, 90, 100]
bisect.bisect_left(intervals, 35)  # 返回 1（落在 [10, 40) 区间）
```

---

## 概率分析

假设种群有 4 个个体，适应度分别为：

| 个体 | 适应度 | 被选概率 |
|------|--------|----------|
| A | 10 | 10% |
| B | 30 | 30% |
| C | 50 | 50% |
| D | 10 | 10% |

适应度越高的个体（C），在轮盘上占的区域越大，被选中的概率越高。

---

## 使用示例

```python
from gaps.selection import roulette_selection

# 从100个个体中选择父母
parents = roulette_selection(population, elites=2)

# 返回格式：[(父1, 母1), (父2, 母2), ...]
for father, mother in parents:
    # 进行交叉操作
    pass
```

---

## 与精英保留的关系

注意 `elites` 参数的作用：
- `elites=4` 表示前 4 优秀的个体不参与轮盘赌选择
- 这些精英个体已经在 `_get_elite_individuals` 中被保留到下一代
- 这里只选择剩余的个体作为父母

这样设计保证了：
1. 精英个体一定保留
2. 其他个体按适应度概率被选择
