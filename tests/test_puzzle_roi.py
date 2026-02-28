"""Tests for puzzle region extraction from screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jigsaw.puzzle_roi import (
    PuzzleROIResult,
    extract_puzzle_region,
    extract_puzzle_region_with_metadata,
)

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = ROOT / "examples" / "IMG_0970.PNG"


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
def test_extract_puzzle_region_plain_image_returns_unchanged() -> None:
    """Without grid lines, extraction returns the full image."""
    image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    out = extract_puzzle_region(image)
    assert out.shape == image.shape
    np.testing.assert_array_equal(out, image)


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
def test_extract_puzzle_region_return_bbox() -> None:
    """With return_bbox=True, returns (cropped, bbox)."""
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    out = extract_puzzle_region(image, return_bbox=True)
    assert isinstance(out, tuple)
    cropped, bbox = out
    assert cropped.shape == image.shape
    x_min, y_min, x_max, y_max = bbox
    assert 0 <= x_min <= x_max <= image.shape[1]
    assert 0 <= y_min <= y_max <= image.shape[0]


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
def test_extract_puzzle_region_with_metadata_result_shape() -> None:
    """extract_puzzle_region_with_metadata returns valid PuzzleROIResult."""
    image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    result = extract_puzzle_region_with_metadata(image)
    assert isinstance(result, PuzzleROIResult)
    assert result.image.shape == image.shape
    x_min, y_min, x_max, y_max = result.bbox
    assert 0 <= x_min <= x_max <= image.shape[1]
    assert 0 <= y_min <= y_max <= image.shape[0]


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
def test_extract_puzzle_region_invalid_input_raises() -> None:
    """Non-RGB image raises ValueError."""
    with pytest.raises(ValueError, match="image must be HxWx3"):
        extract_puzzle_region(np.random.randint(0, 256, (50, 50), dtype=np.uint8))


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
@pytest.mark.skipif(not EXAMPLE_IMAGE.exists(), reason="examples/IMG_0970.PNG not present")
def test_extract_roi_on_screenshot_bbox_reasonable() -> None:
    """On IMG_0970.PNG, extracted ROI has reasonable bbox and area."""
    image = cv2.imread(str(EXAMPLE_IMAGE), cv2.IMREAD_COLOR)
    assert image is not None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    result = extract_puzzle_region_with_metadata(image)
    x_min, y_min, x_max, y_max = result.bbox
    assert 0 <= x_min < x_max <= w
    assert 0 <= y_min < y_max <= h
    crop_area = (x_max - x_min) * (y_max - y_min)
    full_area = w * h
    assert 0.2 <= crop_area / full_area <= 1.0, "crop should be 20%-100% of image"


@pytest.mark.skipif(cv2 is None, reason="opencv-python required for puzzle_roi")
@pytest.mark.skipif(not EXAMPLE_IMAGE.exists(), reason="examples/IMG_0970.PNG not present")
def test_extract_roi_on_screenshot_inferred_grid_plausible() -> None:
    """On IMG_0970.PNG, inferred grid (if any) is plausible (e.g. 7x9)."""
    image = cv2.imread(str(EXAMPLE_IMAGE), cv2.IMREAD_COLOR)
    assert image is not None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = extract_puzzle_region_with_metadata(image)
    if result.rows is not None and result.cols is not None:
        assert 2 <= result.rows <= 30
        assert 2 <= result.cols <= 30
        crop_h = result.image.shape[0]
        crop_w = result.image.shape[1]
        assert crop_h > 0 and crop_w > 0
