"""Tests for mobile transfer module."""

from __future__ import annotations

import numpy as np
import pytest

from jigsaw.utils import generate_natural_like_image
from mobile_transfer.state_extractor import ScreenStateExtractor


@pytest.fixture
def test_image():
    """Generate a test image."""
    return generate_natural_like_image(size=300, seed=42)


@pytest.fixture
def extractor_3x3():
    """Create a 3x3 state extractor."""
    return ScreenStateExtractor(grid_size=(3, 3))


@pytest.fixture
def extractor_6x6():
    """Create a 6x6 state extractor."""
    return ScreenStateExtractor(grid_size=(6, 6))


class TestScreenStateExtractor:
    """Test suite for ScreenStateExtractor."""

    def test_init(self, extractor_3x3):
        """Test initialization."""
        assert extractor_3x3.rows == 3
        assert extractor_3x3.cols == 3
        assert extractor_3x3.n == 9

    def test_extract_from_image(self, extractor_3x3, test_image):
        """Test extracting patches and cost matrix from clean image."""
        patches, cost_matrix = extractor_3x3.extract_from_image(test_image)

        assert len(patches) == 9
        assert cost_matrix.shape == (9, 9, 2)
        assert np.all(np.isfinite(cost_matrix))
        assert np.all(cost_matrix >= 0.0)
        assert np.all(cost_matrix <= 1.0)

    def test_build_observation(self, extractor_3x3, test_image):
        """Test observation building."""
        patches, cost_matrix = extractor_3x3.extract_from_image(test_image)
        grid = np.arange(9, dtype=np.int32).reshape(3, 3)
        obs = extractor_3x3.build_observation(grid, cost_matrix)

        assert "grid" in obs
        assert "edge_costs" in obs
        assert obs["grid"].shape == (9,)
        assert obs["edge_costs"].shape == (9, 9, 2)
        assert obs["grid"].dtype == np.int32
        assert obs["edge_costs"].dtype == np.float32

    def test_extract_6x6(self, extractor_6x6, test_image):
        """Test 6x6 extraction."""
        patches, cost_matrix = extractor_6x6.extract_from_image(test_image)

        assert len(patches) == 36
        assert cost_matrix.shape == (36, 36, 2)

    def test_estimate_grid_from_screenshot(self, extractor_3x3, test_image):
        """Test grid estimation by matching patches."""
        from jigsaw.splitter import PuzzleSplitter
        from jigsaw.utils import shuffle_patches

        splitter = PuzzleSplitter()
        patches = splitter.split(test_image, 3, 3)

        # Get reference patches
        ref_patches, _ = extractor_3x3.extract_from_image(test_image)

        # Shuffle and try to recover
        shuffled, order = shuffle_patches(patches, seed=42)
        estimated_grid = extractor_3x3.estimate_grid_from_screenshot(
            shuffled, ref_patches
        )

        assert estimated_grid.shape == (3, 3)
        # Each value should be a valid patch index
        assert np.all(estimated_grid >= 0)
        assert np.all(estimated_grid < 9)


class TestMobileExecutorOffline:
    """Test offline execution pipeline."""

    def test_offline_pipeline(self, test_image):
        """Test the offline test function."""
        from mobile_transfer.executor import test_offline

        # Save test image to temp file
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f.name, bgr)
            result = test_offline(f.name, grid_size="3x3")

        assert result["grid_size"] == "3x3"
        assert result["n_patches"] == 9
        assert result["observation_valid"] is True
