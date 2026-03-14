(jigsaw) liuxiaoyang@liuxiaoyangdeMac-mini PingLeHaiPingByCodex % python3 demo.py --image thirdparty/gaps/images/lion.jpg --grid 8x8 --solver gaps
Input image size: 736x1308 (WxH)
Note: Image cropped from 1308x736 to 1304x736 for grid 8x8
Processed image size: 736x1304 (WxH)
Patch size: 92x163 (WxH)
=== Pieces:      56

=== Analyzing image: ██████████████████████████████████████████████████ 100.0%
=== Solving puzzle:  ████████████████████████████████------------------ 63.2%

=== GA terminated
=== There was no improvement for 10 generations
Traceback (most recent call last):
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/demo.py", line 170, in <module>
    run_demo(
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/demo.py", line 112, in run_demo
    solve_result = solver.solve(shuffled_patches)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 110, in solve
    grid = self._individual_to_grid(result_individual, patches)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Volumes/tiger/Workspace/side-projects/2026/PingLeHaiPingByCodex/jigsaw/solver/gaps_solver.py", line 141, in _individual_to_grid
    raise ValueError(
ValueError: Piece count mismatch: gaps returned 56 pieces (14x4), but we have 64 patches (requested 8x8). Try using a different piece_size or use the 'default' solver instead.