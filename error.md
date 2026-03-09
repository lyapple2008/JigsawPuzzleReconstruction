1. **[P1] `gaps` 路径的随机性没有被 `seed` 控制，结果不可复现**
- 证据：脚本为每张图构造了 `seed` 并传给求解器配置（[run_real_benchmark.py:263](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:263), [run_real_benchmark.py:169](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:169)），但 `GapsSolver.__init__` 明确忽略额外参数（[gaps_solver.py:38](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py:38), [gaps_solver.py:52](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py:52)），GA 初始化也没有接收种子（[gaps_solver.py:101](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py:101)）。
- 影响：`--solver gaps` 下同一命令重复运行可能得到不同结果，benchmark 可比性受影响。

2. **[P1] 缺少 `--num-images` / `--skip-images` 下界校验，可能导致“错误采样”或空统计**
- 证据：`skip_images` 直接参与 range 起点（[run_real_benchmark.py:251](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:251)）。当 `--skip-images` 为负时，会触发 Python 负索引语义，实际加载到末尾图片。`--num-images` 允许 0 或负数时，`stats_list` 可能为空，后续均值统计走空数组（[run_real_benchmark.py:366](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:366), [run_real_benchmark.py:439](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:439)）。
- 影响：可能生成 `nan` 统计或隐性错误采样，报告不可信。

3. **[P2] 未校验 grid 与图像尺寸关系，超大 grid 会产生 0 尺寸 patch 并在切分时崩溃**
- 证据：`patch_height = h // rows`, `patch_width = w // cols`（[run_real_benchmark.py:140](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/benchmarks/run_real_benchmark.py:140)）。若 `rows > h` 或 `cols > w`，patch 尺寸为 0；随后 `Patch.from_image` 访问 `image[0, ...]` 会越界（[splitter.py:23](/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/splitter.py:23)）。
- 影响：输入参数合法但运行时异常，错误信息不够友好。

