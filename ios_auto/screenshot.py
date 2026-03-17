"""Screenshot capture from iOS device."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .connector import DeviceConnector

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


class Screenshot:
    """Captures screenshots from iOS device via WDA."""

    def __init__(self, connector: DeviceConnector):
        """Initialize screenshot capture.

        Args:
            connector: Device connector instance
        """
        self.connector = connector

    def capture(self, save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Capture screenshot from device.

        Args:
            save_path: Optional path to save screenshot

        Returns:
            RGB image as numpy array (HxWx3)
        """
        if cv2 is None:
            raise RuntimeError("opencv-python (cv2) is required for screenshot capture")

        # Capture screenshot via WDA
        img = self.connector.session.screenshot()

        # If img is already a numpy array
        if isinstance(img, np.ndarray):
            image = img
        else:
            # If it's a PIL-like image
            image = np.array(img)

        # WDA returns PIL Image (PNG) which is RGB format
        # numpy array from PIL is already RGB, no conversion needed

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save as BGR for OpenCV compatibility
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), bgr)

        return image

    def capture_to_temp(self, prefix: str = "screenshot") -> tuple[np.ndarray, Path]:
        """Capture screenshot to a temporary file.

        Args:
            prefix: Prefix for temporary file name

        Returns:
            (image_array, temp_file_path)
        """
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=prefix, delete=False
        ) as f:
            temp_path = Path(f.name)

        image = self.capture(save_path=temp_path)
        return image, temp_path

    def get_puzzle_region(
        self,
        image: np.ndarray,
        method: str = "color",
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Extract puzzle region from screenshot.

        Args:
            image: Full screenshot image
            method: Extraction method ("color" or "grid")

        Returns:
            (cropped_image, bbox) where bbox = (x_min, y_min, x_max, y_max)
        """
        if cv2 is None:
            raise RuntimeError("opencv-python (cv2) is required for ROI extraction")

        from jigsaw.puzzle_roi import extract_puzzle_region

        result = extract_puzzle_region(image, return_bbox=True, method=method)
        if isinstance(result, tuple):
            return result
        else:
            # Shouldn't happen with return_bbox=True
            h, w = image.shape[:2]
            return result, (0, 0, w, h)
