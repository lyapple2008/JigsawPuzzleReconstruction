# gaps/crossover.py 模块文档

## 概述

该模块实现了遗传算法中的**交叉（Crossover）操作**。不同于传统的单点交叉，这里使用的是一种**基于"核（Kernel）"生长的交叉算法**，类似于拼图求解的过程：从一个碎片开始，逐步选择最佳匹配的碎片填充到正确位置。

---

## Crossover 类

### 函数签名

```python
class Crossover(object):
    def __init__(self, first_parent, second_parent):
```

### 作用

将两个父代个体的碎片组合，生成一个子代个体。核心思想是：
1. 随机选择一个"根碎片"作为起点
2. 使用优先级队列，逐步选择最佳匹配的碎片填入
3. 优先使用两个父代共同的信息，其次使用最佳匹配表

---

## 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_parents` | Tuple | 父代个体对 |
| `_kernel` | Dict | 已放置的碎片映射 {piece_id: (row, col)} |
| `_taken_positions` | Set | 已占用的位置集合 |
| `_candidate_pieces` | List | 优先级队列，待放置的候选碎片 |
| `_min_row`, `_max_row` | int | kernel 的行边界 |
| `_min_column`, `_max_column` | int | kernel 的列边界 |

---

## 优先级常数

```python
SHARED_PIECE_PRIORITY = -10   # 最高优先级：两个父代共同放置的碎片
BUDDY_PIECE_PRIORITY = -1     # 次优先级："伙伴"碎片
```

负数越小优先级越高（在 heapq 中）。

---

## 主要方法

### run - 执行交叉

```python
def run(self):
    self._initialize_kernel()

    while len(self._candidate_pieces) > 0:
        _, (position, piece_id), relative_piece = heapq.heappop(...)

        # 跳过已占用的位置
        if position in self._taken_positions:
            continue

        # 如果碎片已放置，找新的候选
        if piece_id in self._kernel:
            self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
            continue

        # 放置碎片到 kernel
        self._put_piece_to_kernel(piece_id, position)
```

### child - 获取子代

```python
def child(self):
    pieces = [None] * self._pieces_length

    for piece, (row, column) in self._kernel.items():
        index = (row - self._min_row) * self._child_columns + (column - self._min_column)
        pieces[index] = self._parents[0].piece_by_id(piece)

    return Individual(pieces, ..., shuffle=False)
```

将 kernel 转换为 Individual 对象返回。

---

## 碎片选择策略

这是算法的核心，按优先级从高到低：

```
┌─────────────────────────────────────────────────────────────┐
│                  碎片选择优先级                               │
├─────────────────────────────────────────────────────────────┤
│  1. 共享碎片 (SHARED_PIECE_PRIORITY = -10)  ← 最高       │
│  2. 伙伴碎片 (BUDDY_PIECE_PRIORITY = -1)    ← 次高       │
│  3. 最佳匹配 (priority = 不相似度分数)        ← 最低     │
└─────────────────────────────────────────────────────────────┘
```

### add_piece_candidate - 候选碎片添加主调度

```python
def add_piece_candidate(self, piece_id, orientation, position):
    # 策略1: 尝试共享碎片
    shared_piece = self._get_shared_piece(piece_id, orientation)
    if self._is_valid_piece(shared_piece):
        self._add_shared_piece_candidate(shared_piece, position, (piece_id, orientation))
        return

    # 策略2: 尝试伙伴碎片
    buddy_piece = self._get_buddy_piece(piece_id, orientation)
    if self._is_valid_piece(buddy_piece):
        self._add_buddy_piece_candidate(buddy_piece, position, (piece_id, orientation))
        return

    # 策略3: 尝试最佳匹配
    best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
    if self._is_valid_piece(best_match_piece):
        self._add_best_match_piece_candidate(best_match_piece, position, priority, (piece_id, orientation))
        return
```

按优先级顺序尝试三种策略，一旦成功添加则返回。

---

### 1. Shared Piece（共享碎片）- 优先级最高 (-10)

**来源**：两个父代的共同信息

```python
def _get_shared_piece(piece_id, orientation):
    first_parent_edge = first_parent.edge(piece_id, orientation)
    second_parent_edge = second_parent.edge(piece_id, orientation)

    if first_parent_edge == second_parent_edge:
        return first_parent_edge
```

**含义**：两个父代在相同位置、相同方向放置了**相同的碎片**，说明这个放置是**高度可信**的。

**示例**：
```
父1:  ┌───┬───┐      父2:  ┌───┬───┐
      │ A │ B │            │ A │ B │   ← 相同！
      └───┴───┘            └───┴───┘

碎片A的右邻居: 父1认为是B，父2也认为是B → B是共享碎片
```

**可信度**：⭐⭐⭐ 最高

---

### 2. Buddy Piece（伙伴碎片）- 优先级次高 (-1)

**来源**：图像互相匹配

```python
def _get_buddy_piece(piece_id, orientation):
    first_buddy = ImageAnalysis.best_match(piece_id, orientation)
    second_buddy = ImageAnalysis.best_match(
        first_buddy, complementary_orientation(orientation)
    )

    if second_buddy == piece_id:
        for edge in [parent.edge(piece_id, orientation) for parent in self._parents]:
            if edge == first_buddy:
                return first_buddy
```

**含义**：碎片A认为碎片B是自己在某方向的最佳匹配，碎片B也认为碎片A是自己在相反方向的最佳匹配 → **互相认定为伙伴**。

**示例**：
```
碎片A的右边缘图像  →  最佳匹配是碎片B
碎片B的左边缘图像  →  最佳匹配也是碎片A
→ A和B是"伙伴关系"，优先级提升
```

**可信度**：⭐⭐ 中等

---

### 3. Best Match Piece（最佳匹配）- 优先级最低

**来源**：图像相似度预计算表

```python
def _get_best_match_piece(self, piece_id, orientation):
    for piece, dissimilarity_measure in ImageAnalysis.best_match_table[piece_id][orientation]:
        if self._is_valid_piece(piece):
            return piece, dissimilarity_measure
```

**含义**：前两种策略都失败后，从预计算的**图像相似度表**中找到最佳匹配的碎片。

- `dissimilarity_measure`（不相似度）作为优先级
- 不相似度越低，越优先被选中

**可信度**：⭐ 最低

---

### 三种策略对比

| 类型 | 来源 | 优先级 | 可信度 | 代码方法 |
|------|------|--------|--------|----------|
| **Shared Piece** | 两父代共同信息 | -10 (最高) | ⭐⭐⭐ | `_get_shared_piece` |
| **Buddy Piece** | 图像互相匹配 | -1 | ⭐⭐ | `_get_buddy_piece` |
| **Best Match** | 图像相似度表 | = 不相似度分数 | ⭐ | `_get_best_match_piece` |

---

### 选择流程图

```
add_piece_candidate(piece_id, orientation, position)
                    │
                    ▼
        ┌───────────────────────┐
        │ 1. 获取 shared_piece  │
        │ (两父代共同邻居)      │
        └───────────────────────┘
                    │
           ┌────────┴────────┐
           │  存在且有效?    │
           └────────┬────────┘
              Yes   │   No
           ┌────────┴──────────────────┐
           ▼                          ▼
    入队(优先级-10)           ┌───────────────────────┐
           │                   │ 2. 获取 buddy_piece  │
           │                   │ (互相认为最佳匹配)   │
           │                   └───────────────────────┘
           │                              │
           │                     ┌────────┴────────┐
           │                     │  存在且有效?    │
           │                     └────────┬────────┘
           │                        Yes   │   No
           │                     ┌────────┴────────┐
           │                     ▼                 ▼
           │              入队(优先级-1)  ┌───────────────────────┐
           │                              │ 3. 获取 best_match  │
           │                              │ (从相似度表)        │
           │                              └───────────────────────┘
           │                                        │
           │                               ┌────────┴────────┐
           │                               │  存在且有效?   │
           │                               └────────┬────────┘
           │                                 Yes   │   No
           │                          ┌────────────┴──────────┐
           │                          ▼
           │                   入队(优先级=不相似度)
           │
           ▼
    (三种都没有→不添加候选)
```

---

## 核生长完整流程

### run() 主循环

```python
def run(self):
    self._initialize_kernel()  # 初始化：随机选择根碎片放到(0,0)

    while len(self._candidate_pieces) > 0:  # 候选队列不为空时循环
        _, (position, piece_id), relative_piece = heapq.heappop(...)

        # 跳过已占用的位置
        if position in self._taken_positions:
            continue

        # 如果碎片已放置，找新的候选放回队列
        if piece_id in self._kernel:
            self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
            continue

        # 放置碎片到 kernel
        self._put_piece_to_kernel(piece_id, position)
```

### 流程步骤

```
┌─────────────────────────────────────────────────────────────────┐
│                      核生长主循环                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 初始化 kernel                                              │
│     └─ 随机选一个根碎片，放到 (0,0)                            │
│                                                                 │
│  2. 进入 while 循环 (候选队列不为空)                            │
│     │                                                          │
│     ▼                                                          │
│  3. 从优先级队列弹出一个候选                                    │
│     │  heapq.heappop() - 取出优先级最高的候选                  │
│     ▼                                                          │
│  4. 检查位置是否已被占用?                                       │
│     │  position in _taken_positions                           │
│     ├─ Yes → continue (跳过，取下一个)                        │
│     └─ No → 继续                                               │
│        ▼                                                       │
│  5. 检查碎片是否已放置?                                         │
│     │  piece_id in _kernel                                    │
│     │                                                          │
│     ├─ Yes → 找新的候选放回队列，continue                     │
│     │                                                          │
│     └─ No → 继续                                               │
│        ▼                                                       │
│  6. _put_piece_to_kernel(piece_id, position)                  │
│     │                                                          │
│     ├─ 记录碎片位置到 _kernel                                  │
│     ├─ 标记位置已占用                                          │
│     └─ 更新候选碎片 ──────────────────────────────┐            │
│                                                    │            │
│  7. _update_candidate_pieces(piece_id, position)  │            │
│     │                                              │            │
│     ├─ 获取该位置的四个边界 (上/下/左/右)          │            │
│     │                                              │            │
│     └─ 对每个边界调用 add_piece_candidate()        │            │
│         (尝试添加共享/伙伴/最佳匹配碎片到队列)     │            │
│                                                    │            │
│     ◄──────────────────────────────────────────────┘            │
│     │                                                          │
│     ▼                                                          │
│  8. 回到步骤 3，继续循环                                        │
│     │                                                          │
│     ▼                                                          │
│  9. 队列为空 或 kernel 已满 → 退出                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 生长过程示例

```
初始: 随机选择碎片A作为根
   ┌───┐
   │ A │    ← 放入(0,0)
   └───┘

   候选队列: [B在(0,1), D在(1,0), ...]

步骤1: 取出优先级最高的候选(比如B在位置(0,1))
       放入B到(0,1)
       检查B的四个边界，添加新候选

   ┌───┬───┐
   │ A │ B │    ← kernel = {A:(0,0), B:(0,1)}
   └───┴───┘

步骤2: 取出下一个候选(比如D在位置(1,0))
       放入D到(1,0)

   ┌───┬───┐
   │ A │ B │
   ├───┼───┤
   │ D │   │
   └───┴───┘

... 继续直到填满
```

### 关键点

1. **每放置一个碎片** → 检查它的四个边界位置
2. **为每个边界位置** → 调用 `add_piece_candidate()` 尝试添加候选
3. **继续从队列取下一个** → 重复直到完成
4. **不是"找完上下左右就停止"**，而是循环执行直到队列为空

---

## 优先级队列机制

使用 Python 的 `heapq` 实现最小堆，优先级低的先出列：

```python
# 入队
piece_candidate = (priority, (position, piece_id), relative_piece)
heapq.heappush(self._candidate_pieces, piece_candidate)

# 出队
priority, (position, piece_id), relative_piece = heapq.heappop(self._candidate_pieces)
```

### 候选碎片入队时机

1. 初始化 kernel 后
2. 每放置一个新碎片后，检测其四个边界
3. 碎片已被放置但位置不合适时，找新的候选

---

## 边界检测

```python
def _available_boundaries(self, position):
    positions = {
        "T": (row - 1, column),  # 上
        "R": (row, column + 1),  # 右
        "D": (row + 1, column),  # 下
        "L": (row, column - 1),  # 左
    }
    # 返回未占用且在范围内的边界位置
```

---

## 辅助方法详解

### 1. _initialize_kernel - 初始化核

```python
def _initialize_kernel(self):
    root_piece = self._parents[0].pieces[
        int(random.uniform(0, self._pieces_length))
    ]
    self._put_piece_to_kernel(root_piece.id, (0, 0))
```

随机从第一个父代中选择一个碎片作为"根碎片"，放置在位置 (0, 0)。

---

### 2. _put_piece_to_kernel - 放置碎片到核

```python
def _put_piece_to_kernel(self, piece_id, position):
    self._kernel[piece_id] = position          # 记录碎片位置
    self._taken_positions.add(position)        # 标记位置已占用
    self._update_candidate_pieces(piece_id, position)  # 更新候选队列
```

将碎片放入 kernel，并触发候选碎片更新。

---

### 3. _update_candidate_pieces - 更新候选碎片

```python
def _update_candidate_pieces(self, piece_id, position):
    available_boundaries = self._available_boundaries(position)

    for orientation, position in available_boundaries:
        self.add_piece_candidate(piece_id, orientation, position)
```

当碎片放置后，检查其四个边界，为每个可用的边界位置添加候选碎片。

---

### 4. add_piece_candidate - 添加候选碎片（主调度）

```python
def add_piece_candidate(self, piece_id, orientation, position):
    # 策略1: 尝试获取共享碎片
    shared_piece = self._get_shared_piece(piece_id, orientation)
    if self._is_valid_piece(shared_piece):
        self._add_shared_piece_candidate(shared_piece, position, (piece_id, orientation))
        return

    # 策略2: 尝试获取伙伴碎片
    buddy_piece = self._get_buddy_piece(piece_id, orientation)
    if self._is_valid_piece(buddy_piece):
        self._add_buddy_piece_candidate(buddy_piece, position, (piece_id, orientation))
        return

    # 策略3: 获取最佳匹配碎片
    best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
    if self._is_valid_piece(best_match_piece):
        self._add_best_match_piece_candidate(best_match_piece, position, priority, (piece_id, orientation))
        return
```

按优先级顺序尝试三种策略，将有效的候选碎片加入队列。

---

### 5. 候选入队方法

#### _add_shared_piece_candidate

```python
def _add_shared_piece_candidate(self, piece_id, position, relative_piece):
    piece_candidate = (SHARED_PIECE_PRIORITY, (position, piece_id), relative_piece)
    heapq.heappush(self._candidate_pieces, piece_candidate)
```

优先级 -10，最高优先级入队。

#### _add_buddy_piece_candidate

```python
def _add_buddy_piece_candidate(self, piece_id, position, relative_piece):
    piece_candidate = (BUDDY_PIECE_PRIORITY, (position, piece_id), relative_piece)
    heapq.heappush(self._candidate_pieces, piece_candidate)
```

优先级 -1，次高优先级入队。

#### _add_best_match_piece_candidate

```python
def _add_best_match_piece_candidate(self, piece_id, position, priority, relative_piece):
    piece_candidate = (priority, (position, piece_id), relative_piece)
    heapq.heappush(self._candidate_pieces, piece_candidate)
```

使用不相似度分数作为优先级入队。

---

### 6. 范围检测方法

#### _is_kernel_full - 检查核是否已满

```python
def _is_kernel_full(self):
    return len(self._kernel) == self._pieces_length
```

当已放置碎片数量等于总碎片数时，核已满。

#### _is_in_range - 检查位置是否在范围内

```python
def _is_in_range(self, row_and_column):
    (row, column) = row_and_column
    return self._is_row_in_range(row) and self._is_column_in_range(column)
```

同时检查行和列是否在允许范围内。

#### _is_row_in_range / _is_column_in_range

```python
def _is_row_in_range(self, row):
    current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
    return current_rows < self._child_rows

def _is_column_in_range(self, column):
    current_columns = abs(min(self._min_column, column)) + abs(
        max(self._max_column, column)
    )
    return current_columns < self._child_columns
```

动态计算当前 kernel 的跨度，确保不超过子代的行/列数。

---

### 7. _update_kernel_boundaries - 更新核边界

```python
def _update_kernel_boundaries(self, row_and_column):
    (row, column) = row_and_column
    self._min_row = min(self._min_row, row)
    self._max_row = max(self._max_row, row)
    self._min_column = min(self._min_column, column)
    self._max_column = max(self._max_column, column)
```

当新位置加入时，更新 kernel 的边界记录。

---

### 8. _is_valid_piece - 检查碎片是否有效

```python
def _is_valid_piece(self, piece_id):
    return piece_id is not None and piece_id not in self._kernel
```

有效碎片需满足：
- 碎片ID不为 None
- 碎片尚未被放置到 kernel 中

---

### 9. complementary_orientation - 互补方向（顶层函数）

```python
def complementary_orientation(orientation):
    return {"T": "D", "R": "L", "D": "T", "L": "R"}.get(orientation, None)
```

返回方向的互补方向：
- 上(T) ↔ 下(D)
- 左(L) ↔ 右(R)

用于伙伴碎片检测：若A在B的右边，则B在A的左边。

---

## 核生长过程示例

```
初始: 随机选择碎片A作为根
   ┌───┐
   │ A │
   └───┘

第1步: A的右邻居最佳匹配B
   ┌───┬───┐
   │ A │ B │
   └───┴───┘

第2步: B的下邻居是C（共享碎片）
   ┌───┬───┐
   │ A │ B │
   ├───┼───┤
   │   │ C │
   └───┴───┘

第3步: A的上邻居是D，C的右邻居是E
   ┌───┬───┐
   │   │ D │
   ├───┼───┤
   │ A │ B │
   ├───┼───┤
   │   │ C │ E
   └───┴───┘

... 继续生长直到填满
```

---

## 使用示例

```python
from gaps.crossover import Crossover

# 创建交叉对象
crossover = Crossover(parent1, parent2)

# 执行交叉
crossover.run()

# 获取子代
child = crossover.child()
```

---

## 与传统交叉的区别

| 传统交叉 | 本算法 |
|----------|--------|
| 随机切分染色体 | 从根碎片开始生长 |
| 可能产生无效解 | 保证解的有效性 |
| 不利用问题知识 | 利用图像匹配信息 |

这种算法专门为拼图问题设计，充分利用了碎片相似度信息。
