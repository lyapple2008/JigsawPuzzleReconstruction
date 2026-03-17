(jigsaw) liuxiaoyang@liuxiaoyangdeMac-mini JigsawPuzzleReconstruction % python3 -m ios_auto.automation --grid 8x8 --udid 00008150-00152C6A117A401C

<frozen runpy>:128: RuntimeWarning: 'ios_auto.automation' found in sys.modules after import of package 'ios_auto', but prior to execution of 'ios_auto.automation'; this may result in unpredictable behaviour
==================================================
iOS Jigsaw Puzzle Solver - Automation Started
==================================================

[1/6] Connecting to iOS device...
Connected to device: 00008150-00152C6A117A401C

[2/6] Capturing initial puzzle state...
=== Pieces:      64

=== Analyzing image: ██████████████████████████████████████████████████ 100.0%
=== Solving puzzle:  ██████████████████████████████████---------------- 68.4%

=== GA terminated
=== There was no improvement for 10 generations
    Puzzle bbox: (31, 455, 1181, 2174)
    Solved grid:
[[ 0  1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14 15]
 [16 17 18 19 20 21 22 23]
 [24 25 26 27 28 29 30 45]
 [32 33 34 35 36 37 38 63]
 [40 41 42 43 44 47 46 55]
 [48 49 50 51 52 53 54 31]
 [56 57 58 59 60 61 62 39]]

[3/6] Starting solve loop...

--- Iteration 1 (elapsed: 3.1s) ---
=== Pieces:      64

=== Analyzing image: ██████████████████████████████████████████████████ 100.0%
=== Solving puzzle:  ██████████████████████████████████---------------- 68.4%

=== GA terminated
=== There was no improvement for 10 generations
    Planned 6 moves
    Executing moves...
    Move 1/6: SwapMove((2, 7) -> (5, 7))

[6/6] Disconnecting device...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/liuxiaoyang/workspace/JigsawPuzzleReconstruction/ios_auto/automation.py", line 372, in <module>
    run_automation(
  File "/Users/liuxiaoyang/workspace/JigsawPuzzleReconstruction/ios_auto/automation.py", line 223, in run_automation
    gesture.swap_pieces(move.from_pos, move.to_pos)
  File "/Users/liuxiaoyang/workspace/JigsawPuzzleReconstruction/ios_auto/gesture.py", line 186, in swap_pieces
    self.drag(from_pos, to_pos, duration)
  File "/Users/liuxiaoyang/workspace/JigsawPuzzleReconstruction/ios_auto/gesture.py", line 162, in drag
    session.swipe(
  File "/Users/liuxiaoyang/miniconda3/envs/jigsaw/lib/python3.11/site-packages/wda/__init__.py", line 902, in swipe
    x1, y1 = self._percent2pos(x1, y1, size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/liuxiaoyang/miniconda3/envs/jigsaw/lib/python3.11/site-packages/wda/__init__.py", line 857, in _percent2pos
    assert w >= x >= 0
           ^^^^^^^^^^^
AssertionError