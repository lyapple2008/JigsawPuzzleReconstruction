# 错误分析报告

## 执行命令
```bash
python3 reconstruct.py --image examples/IMG_0970.PNG --grid 8x8 --extract-roi --solver gaps
```

## 报错信息
```
Extracted puzzle ROI: no regular grid detected, using full image
=== Pieces:      96

=== Analyzing image: ██████████████████████████████████████████████████ 100.0%
=== Solving puzzle:  █████████████████████████████--------------------- 57.9%

=== GA terminated
=== There was no improvement for 10 generations
Traceback (most recent call last):
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/reconstruct.py", line 219, in <module>
    main()
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/reconstruct.py", line 176, in main
    solve_result = solver.solve(patches, original_image=original_for_solver, cost_matrix=cost_matrix)
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 103, in solve
    grid = self._individual_to_grid(result_individual, patches)
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 134, in _individual_to_grid
    raise ValueError(
ValueError: Piece count mismatch: gaps returned 96 pieces (12x8), but we have 64 patches (requested 8x8). Try using a different piece_size or use the 'default' solver instead.
```

## 报错原因分析

### 核心问题
网格尺寸不匹配：
- 用户指定 `--grid 8x8`，期望 64 个拼图块
- 但 `split_with_gap_aware` 按 8x8 切分产生了 64 个 patches
- gaps 求解器根据图像尺寸自行计算 `piece_size`，实际产生了 96 个块（12×8）

### 问题链路

1. **ROI 提取失败**
   - 使用了 `--extract-roi`，但检测失败：`Extracted puzzle ROI: no regular grid detected, using full image`
   - 系统回退到使用完整图像

2. **piece_size 计算差异**
   - 在 `gaps_solver.py:82-90`，当传入 `piece_size` 时使用传入值，否则自动计算：
     ```python
     if self._piece_size is not None:
         piece_size = self._piece_size
     else:
         height, width = original_image.shape[:2]
         piece_size = min(height // self.rows, width // self.cols)
     ```
   - `reconstruct.py` 中确实传入了 `piece_size`：
     ```python
     patch_h, patch_w = sorted_patches[0].image.shape[:2]
     solver_kwargs["piece_size"] = min(patch_h, patch_w)
     ```

3. **尺寸不匹配原因**
   - patches 是通过 `split_with_gap_aware(image, rows=8, cols=8)` 得到的
   - 传入 gaps 的是 `original_image`（切割后的图像）
   - gaps 内部会根据 `original_image` 重新切分，导致块数量与 patches 不一致

### 具体数值估算
- 假设 ROI 提取后的图像约为 800×600 像素
- patch 大小约为 100×75（8×8=64 块）
- gaps 内部 `piece_size = min(600//8, 800//8) = 75`
- gaps 实际切分：600/75 × 800/75 ≈ 8×10.67 → 实际得到 12×8 = 96 块

## 解决方案思路

1. **方案一**：使用 `default` 求解器代替 `gaps`
2. **方案二**：修改 gaps 求解器，使其接受外部传入的 patches 而非重新切分图像
3. **方案三**：确保传入正确的 `piece_size`，使 gaps 切分出的块数与 patches 一致
