(jigsaw) liuxiaoyang@liuxiaoyangdeMac-mini PingLeHaiPingByCodex % python3 reconstruct.py --image examples/IMG_0970.PNG  --grid 8x8 --extract-roi --solver gaps
Starting reconstruct...
Extracted puzzle ROI: bbox=(31, 455, 1181, 2174), grid inferred from --grid argument
=== Pieces:      60

=== Analyzing image: ██████████████████████████████████████████████████ 100.0%
=== Solving puzzle:  █████████████████████████████--------------------- 57.9%

=== GA terminated
=== There was no improvement for 10 generations
求解器错误: Piece count mismatch: gaps returned 60 pieces (12x5), but we have 64 patches (requested 8x8). Try using a different piece_size or use the 'default' solver instead.
建议: 使用 --solver default 或检查 --grid 与图像尺寸是否匹配。
Traceback (most recent call last):
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/reconstruct.py", line 258, in <module>
    main()
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/reconstruct.py", line 184, in main
    solve_result = solver.solve(patches, original_image=original_for_solver, cost_matrix=cost_matrix)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 111, in solve
    grid = self._individual_to_grid(result_individual, patches)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 142, in _individual_to_grid
    raise ValueError(
ValueError: Piece count mismatch: gaps returned 60 pieces (12x5), but we have 64 patches (requested 8x8). Try using a different piece_size or use the 'default' solver instead.