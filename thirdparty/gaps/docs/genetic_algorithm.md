# gaps/genetic_algorithm.py 模块文档

## 概述

该模块实现了图像拼图求解的遗传算法（Genetic Algorithm）。通过模拟自然选择、交叉、变异等过程，逐步进化出最优的图像拼图解。

---

## GeneticAlgorithm 类

### 函数签名

```python
class GeneticAlgorithm(object):
    def __init__(self, image, piece_size, population_size, generations, elite_size=2):
```

### 作用

初始化遗传算法对象，将图像分割成碎片并生成初始种群。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入图像（打乱后的拼图图像） |
| `piece_size` | int | 必填 | 单个碎片的尺寸（像素） |
| `population_size` | int | 必填 | 种群大小（个体数量） |
| `generations` | int | 必填 | 最大迭代代数 |
| `elite_size` | int | 2 | 精英个体数量（保留的最优个体） |

### 初始化流程

1. 调用 `utils.flatten_image(image, piece_size, indexed=True)` 将图像分割成带索引的碎片
2. 创建 `population_size` 个 `Individual` 对象作为初始种群
3. 每个个体代表一种碎片的随机排列方式

### 使用示例

```python
from gaps.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(
    image=shuffled_image,  # 打乱后的图像
    piece_size=32,          # 碎片大小
    population_size=200,   # 种群大小
    generations=50,        # 迭代代数
    elite_size=2           # 精英数量
)
result = ga.start_evolution(verbose=True)
solved_image = result.to_image()
```

---

## start_evolution 函数

### 函数签名

```python
def start_evolution(self, verbose):
```

### 作用

启动遗传算法进化过程，逐步迭代直到找到最优解或达到终止条件。

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `verbose` | bool | 是否显示实时进度和图像 |

### 返回值

返回最优的 `Individual` 对象，可通过 `to_image()` 方法获取求解后的图像。

---

## 算法流程详解

```
┌─────────────────────────────────────────────────────────┐
│                   start_evolution                       │
├─────────────────────────────────────────────────────────┤
│  1. 图像分析 (ImageAnalysis.analyze_image)              │
│     - 预计算碎片边缘特征                                │
│     - 准备适应度评估                                    │
│                                                         │
│  2. 迭代进化 (for generation in range(generations))     │
│     ┌───────────────────────────────────────────────┐  │
│     │  a) 精英保留 (Elitism)                         │  │
│     │     - 从当前种群选取 top elite_size 个体      │  │
│     │     - 直接复制到下一代                         │  │
│     │                                               │  │
│     │  b) 父母选择 (Roulette Selection)             │  │
│     │     - 轮盘赌选择法                             │  │
│     │     - 适应度越高的个体被选中的概率越大         │  │
│     │                                               │  │
│     │  c) 交叉 (Crossover)                          │  │
│     │     - 两个父母染色体交叉                      │  │
│     │     - 生成子代个体                             │  │
│     │                                               │  │
│     │  d) 评估适应度                                 │  │
│     │     - 计算每个个体的适应度分数                 │  │
│     │                                               │  │
│     │  e) 终止条件检查                               │  │
│     │     - 连续 10 代无改进则终止                   │  │
│     └───────────────────────────────────────────────┘  │
│                                                         │
│  3. 返回最优个体                                        │
└─────────────────────────────────────────────────────────┘
```

### 详细步骤

#### 步骤 1: 图像分析
```python
ImageAnalysis.analyze_image(self._pieces)
```
预计算每个碎片的边缘特征，用于后续的适应度计算。

#### 步骤 2: 精英保留 (Elitism)
```python
elite = self._get_elite_individuals(elites=self._elite_size)
new_population.extend(elite)
```
保留上一代中最优秀的 `elite_size` 个个体，直接复制到下一代，确保最优解不会丢失。

#### 步骤 3: 父母选择
```python
selected_parents = roulette_selection(self._population, elites=self._elite_size)
```
使用轮盘赌选择法，根据适应度概率选择父母个体。

#### 步骤 4: 交叉生成子代
```python
for first_parent, second_parent in selected_parents:
    crossover = Crossover(first_parent, second_parent)
    crossover.run()
    child = crossover.child()
    new_population.append(child)
```
对选中的父母进行交叉操作，生成子代加入新种群。

#### 步骤 5: 终止条件检查
```python
if termination_counter == self.TERMINATION_THRESHOLD:
    # 连续 10 代无改进，终止算法
```
如果连续 10 代最优适应度没有提升，提前终止算法。

---

## 关键参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `population_size` | 100-500 | 种群越大，找到最优解概率越高，但计算时间更长 |
| `generations` | 30-100 | 迭代代数越多，解的质量越高 |
| `elite_size` | 2-5 | 精英数量不宜过多，避免陷入局部最优 |
| `TERMINATION_THRESHOLD` | 10 | 提前终止的耐心值 |

---

## 适应度函数

适应度评估基于相邻碎片之间的颜色相似度：
- 越相邻的两个碎片边缘颜色越接近，适应度越高
- 完整还原的图像适应度最高

---

## 注意事项

- 输入的 `image` 应该是打乱后的拼图图像，而非原始图像
- `indexed=True` 会在碎片中保留原始位置信息，用于计算适应度
- `verbose=True` 会实时显示当前最优解的图像
