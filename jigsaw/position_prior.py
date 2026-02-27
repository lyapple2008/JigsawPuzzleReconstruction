"""Lightweight coordinate prior for puzzle piece placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .splitter import Patch, PuzzleSplitter
from .utils import generate_natural_like_image


@dataclass
class LinearPositionPrior:
    """Simple ridge-regression model mapping patch features to (row, col)."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray  # [feature_dim + 1, 2], last row is bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.feature_mean) / self.feature_std
        return x @ self.weights[:-1] + self.weights[-1]


_MODEL_CACHE: Dict[Tuple[int, int, int, int], LinearPositionPrior] = {}


def _extract_patch_feature(patch: Patch) -> np.ndarray:
    img = patch.image.astype(np.float32)
    feats: List[float] = []
    feats.extend(np.mean(img, axis=(0, 1)).tolist())
    feats.extend(np.std(img, axis=(0, 1)).tolist())
    center = img[img.shape[0] // 4 : 3 * img.shape[0] // 4, img.shape[1] // 4 : 3 * img.shape[1] // 4, :]
    feats.extend(np.mean(center, axis=(0, 1)).tolist())
    feats.extend(np.std(center, axis=(0, 1)).tolist())
    # coarse edge statistics
    feats.extend(np.mean(img[0, :, :], axis=0).tolist())
    feats.extend(np.mean(img[-1, :, :], axis=0).tolist())
    feats.extend(np.mean(img[:, 0, :], axis=0).tolist())
    feats.extend(np.mean(img[:, -1, :], axis=0).tolist())
    return np.asarray(feats, dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    xtx = x.T @ x
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def train_position_prior(rows: int, cols: int, samples: int = 24, seed: int = 42) -> LinearPositionPrior:
    """Train a tiny coordinate prior on synthetic images for a specific grid size."""
    key = (rows, cols, samples, seed)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    rng = np.random.default_rng(seed)
    splitter = PuzzleSplitter()
    image_size = max(rows, cols) * 60

    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for _ in range(samples):
        img_seed = int(rng.integers(0, 1_000_000))
        image = generate_natural_like_image(size=image_size, seed=img_seed)
        patches = splitter.split(image, rows, cols)
        for idx, patch in enumerate(patches):
            r, c = divmod(idx, cols)
            features.append(_extract_patch_feature(patch))
            targets.append(np.array([r / max(rows - 1, 1), c / max(cols - 1, 1)], dtype=np.float64))

    x = np.stack(features, axis=0)
    y = np.stack(targets, axis=0)
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    x_norm = (x - mu) / sigma
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float64)], axis=1)
    w = _fit_ridge(x_aug, y, alpha=1e-2)

    model = LinearPositionPrior(feature_mean=mu, feature_std=sigma, weights=w)
    _MODEL_CACHE[key] = model
    return model


def build_position_penalty(
    patches: List[Patch], rows: int, cols: int, samples: int = 24, seed: int = 42
) -> np.ndarray:
    """Return position prior penalty tensor [rows, cols, piece_index]."""
    model = train_position_prior(rows=rows, cols=cols, samples=samples, seed=seed)
    feat = np.stack([_extract_patch_feature(p) for p in patches], axis=0)
    pred = model.predict(feat)  # [n, 2], normalized row/col
    pred = np.clip(pred, 0.0, 1.0)

    rr = np.arange(rows, dtype=np.float64) / max(rows - 1, 1)
    cc = np.arange(cols, dtype=np.float64) / max(cols - 1, 1)

    penalty = np.zeros((rows, cols, len(patches)), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            dr = pred[:, 0] - rr[r]
            dc = pred[:, 1] - cc[c]
            penalty[r, c, :] = dr * dr + dc * dc
    return penalty
