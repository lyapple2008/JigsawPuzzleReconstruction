# gaps/utils.py 模块文档

## 概述

该模块提供图像处理工具函数，用于将图像切分成方块碎片，以及将碎片重新组装成图像。

---

## flatten_image

### 函数签名

```python
def flatten_image(image, piece_size, indexed=False):
```

### 作用

将输入图像分割成指定大小的正方形碎片，并将这些碎片展平成一个列表。每个碎片是一个 `piece_size x piece_size x 3` 的三维数组（RGB 图像）。

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | numpy.ndarray | 输入图像 |
| `piece_size` | int | 单个方块碎片的尺寸（像素） |
| `indexed` | bool | 可选参数。如果为 `True`，返回带索引的 `Piece` 对象列表；否则返回 numpy 数组碎片列表 |

### 返回值

返回元组 `(pieces, rows, columns)`：

- **pieces**: 图像碎片列表，每个元素是 `piece_size x piece_size x 3` 的 numpy 数组
- **rows**: 图像在垂直方向上分割的行数 = `image.shape[0] // piece_size`
- **columns**: 图像在水平方向上分割的列数 = `image.shape[1] // piece_size`

### 使用示例

```python
from gaps.utils import flatten_image

# 将图像分割成 32x32 像素的碎片
pieces, rows, columns = flatten_image(image, 32)
# 假设图像 640x480，分割后得到 rows=15, columns=20，共 300 个碎片
```

---

## assemble_image

### 函数签名

```python
def assemble_image(pieces, rows, columns):
```

### 作用

将图像碎片列表重新组装成完整的图像。根据指定的行数和列数，将碎片按顺序排列并堆叠成原始图像。

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `pieces` | list | 图像碎片列表 |
| `rows` | int | 目标图像的行数（碎片网格的行数） |
| `columns` | int | 目标图像的列数（碎片网格的列数） |

### 返回值

返回组装后的图像，类型为 `numpy.ndarray`，形状为 `(rows * piece_size, columns * piece_size, 3)`，数据类型为 `uint8`。

### 使用示例

```python
from gaps.utils import flatten_image, assemble_image

# 分割图像
pieces, rows, columns = flatten_image(image, 32)

# 重新组装图像
reassembled = assemble_image(pieces, rows, columns)
```

### 内部实现逻辑

1. 先按行分组，每行使用 `np.hstack()` 水平拼接
2. 再使用 `np.vstack()` 垂直拼接各行
3. 最终结果转换为 `uint8` 类型

---

## 注意事项

- `flatten_image` 中分割的行数和列数是通过整数除法计算的，可能会有边缘裁剪
- 碎片在列表中的顺序是行优先（row-major）：从左到右、从上到下
- 重新组装时，碎片的索引计算方式为 `i * columns + j`（i 为行索引，j 为列索引）
