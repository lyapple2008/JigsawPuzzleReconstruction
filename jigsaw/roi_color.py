"""基于背景颜色的拼图 ROI 提取。

利用背景色相对固定的特点，将图像分为背景与非背景，
非背景中最大的连通域视为拼图区域。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from .puzzle_roi import PuzzleROIResult


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("roi_color 需要 opencv-python (cv2)")


class ColorBasedPuzzleExtractor:
    """基于背景颜色的拼图区域提取器。输入、输出均为 RGB。"""

    def __init__(
        self,
        sample_border: int = 30,
        distance_threshold: int = 35,
        min_area_ratio: float = 0.15,
        max_area_ratio: float = 0.85,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        center_offset_tol: float = 0.3,
        refine_margin: int = 5,
    ):
        self.sample_border = sample_border
        self.distance_threshold = distance_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.center_offset_tol = center_offset_tol
        self.refine_margin = refine_margin
        self._bg_color: Optional[Tuple[int, int, int]] = None

    def extract_background_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """从图像四周边缘采样得到背景色 (R, G, B)。"""
        _require_cv2()
        h, w = image.shape[:2]
        border = min(self.sample_border, h // 4, w // 4)
        if border < 2:
            border = 2

        top = image[:border, :]
        bottom = image[h - border :, :]
        left = image[:, :border]
        right = image[:, w - border :]
        samples = [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)]
        all_pixels = np.vstack([s for s in samples if s.size > 0])
        n = min(4000, all_pixels.shape[0])
        indices = np.random.default_rng(42).choice(all_pixels.shape[0], n, replace=False)
        pixels = all_pixels[indices]
        bg_color = np.median(pixels, axis=0).astype(int)
        bg_color = np.clip(bg_color, 0, 255)
        self._bg_color = tuple(bg_color.tolist())
        return self._bg_color

    def create_color_distance_mask(self, image: np.ndarray) -> np.ndarray:
        """二值掩码：非背景=255，背景=0。基于 RGB 欧氏距离。"""
        _require_cv2()
        if self._bg_color is None:
            self.extract_background_color(image)
        bg = np.array(self._bg_color, dtype=np.float64)
        diff = image.astype(np.float64) - bg
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = (distance > self.distance_threshold).astype(np.uint8) * 255
        return mask

    def morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """形态学清理：开运算去噪、闭运算填洞。"""
        _require_cv2()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_largest_region(
        self,
        mask: np.ndarray,
        height: int,
        width: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """返回最大合法连通域的 (x, y, w, h)，不满足条件则 None。"""
        _require_cv2()
        total = height * width
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cx, cy = width / 2.0, height / 2.0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area <= 0:
                continue
            if area / total < self.min_area_ratio or area / total > self.max_area_ratio:
                continue
            aspect = cw / float(ch) if ch > 0 else 0
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue
            mid_x = x + cw / 2.0
            mid_y = y + ch / 2.0
            if abs(mid_x - cx) / width > self.center_offset_tol or abs(mid_y - cy) / height > self.center_offset_tol:
                continue
            return (x, y, cw, ch)
        if contours:
            x, y, cw, ch = cv2.boundingRect(contours[0])
            return (x, y, cw, ch)
        return None

    def refine_bbox(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """基于掩码微调边界框，并加边距。"""
        _require_cv2()
        margin = self.refine_margin if margin is None else margin
        x, y, w, h = bbox
        h_m, w_m = mask.shape
        for i in range(x - 1, max(0, x - 20), -1):
            if np.any(mask[y : y + h, i] > 0):
                x = i
                break
        for i in range(x + w, min(w_m, x + w + 20)):
            if np.any(mask[y : y + h, i] > 0):
                w = i - x
                break
        for i in range(y - 1, max(0, y - 20), -1):
            if np.any(mask[i, x : x + w] > 0):
                y = i
                break
        for i in range(y + h, min(h_m, y + h + 20)):
            if np.any(mask[i, x : x + w] > 0):
                h = i - y
                break
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w_m - x, w + 2 * margin)
        h = min(h_m - y, h + 2 * margin)
        return (x, y, w, h)

    def extract(self, image: np.ndarray) -> PuzzleROIResult:
        """
        从 RGB 图像中提取拼图 ROI。

        Args:
            image: HxWx3 RGB 图像。

        Returns:
            PuzzleROIResult，若未检测到有效区域则 bbox 为整图、rows/cols 为 None。
        """
        _require_cv2()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 须为 HxWx3 RGB")
        h, w = image.shape[:2]
        self.extract_background_color(image)
        mask = self.create_color_distance_mask(image)
        mask = self.morphological_cleanup(mask)
        bbox = self.find_largest_region(mask, h, w)
        if bbox is None:
            return PuzzleROIResult(
                image=image.copy(),
                bbox=(0, 0, w, h),
                rows=None,
                cols=None,
            )
        bbox = self.refine_bbox(mask, bbox)
        x, y, cw, ch = bbox
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        if cw <= 0 or ch <= 0:
            return PuzzleROIResult(
                image=image.copy(),
                bbox=(0, 0, w, h),
                rows=None,
                cols=None,
            )
        cropped = image[y : y + ch, x : x + cw, :].copy()
        return PuzzleROIResult(
            image=cropped,
            bbox=(x, y, x + cw, y + ch),
            rows=None,
            cols=None,
        )


def extract_puzzle_region_by_color(
    image: np.ndarray,
    sample_border: int = 30,
    distance_threshold: int = 35,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 0.85,
    **kwargs: object,
) -> PuzzleROIResult:
    """
    基于背景颜色提取拼图 ROI（便捷函数）。

    Args:
        image: HxWx3 RGB 图像。
        sample_border: 边缘采样宽度。
        distance_threshold: 颜色距离阈值。
        min_area_ratio / max_area_ratio: 面积占比范围。
        **kwargs: 传给 ColorBasedPuzzleExtractor 的其它参数。

    Returns:
        PuzzleROIResult。
    """
    extractor = ColorBasedPuzzleExtractor(
        sample_border=sample_border,
        distance_threshold=distance_threshold,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        **kwargs,
    )
    return extractor.extract(image)
