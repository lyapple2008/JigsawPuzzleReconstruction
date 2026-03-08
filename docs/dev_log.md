# Dev Log

## 2026-03-08

### Benchmark Results on COCO val2017

Dataset: COCO val2017 (50 images per grid)
Image size: 300x300

#### Default Solver

| Grid | Patch Size | Images | PosAcc(%) | NbrAcc(%) | Cost | Time(s) |
|------|------------|--------|------------|------------|------|---------|
| 3x5 | 100x60 | 50 | 91.73 | 97.00 | 917665.41 | 0.021 |
| 5x5 | 60x60 | 50 | 66.16 | 85.85 | 2083123.78 | 0.046 |
| 8x8 | 37x37 | 50 | 18.81 | 55.88 | 6611717.70 | 0.242 |

**Summary:**
- Mean Position Accuracy: 58.90% ± 30.21%
- Mean Neighbor Accuracy: 79.58% ± 17.37%

#### Gaps Solver

| Grid | Patch Size | Images | PosAcc(%) | NbrAcc(%) | Cost | Time(s) |
|------|------------|--------|------------|------------|------|---------|
| 3x5 | 100x60 | 50 | 95.33 | 98.00 | 918379.65 | 0.154 |
| 5x5 | 60x60 | 50 | 95.20 | 98.10 | 1245341.00 | 0.279 |
| 8x8 | 37x37 | 50 | 87.66 | 97.16 | 2036890.81 | 0.808 |

**Summary:**
- Mean Position Accuracy: 92.73% ± 3.59%
- Mean Neighbor Accuracy: 97.75% ± 0.42%

### Observations

1. **Gaps solver significantly outperforms default solver** on all grid sizes
2. **Gaps solver is more stable** with lower standard deviation
3. **Gaps solver is slower** (~5-10x slower than default)
4. Both solvers show decreasing accuracy as grid size increases
5. Neighbor accuracy is consistently higher than position accuracy

### Configuration

- Dataset: COCO val2017
- Image size: 300x300
- Images tested: 50 per grid
- Random seed: 42 (for default solver)
