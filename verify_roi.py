#!/usr/bin/env python3
"""单独验证拼图 ROI 提取：加载截图、提取拼图区域并可视化对比。

在 conda 环境 jigsaw 中执行：
    conda activate jigsaw
    python verify_roi.py --image examples/IMG_0970.PNG
    python verify_roi.py --image path/to/screenshot.png --output crop.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None

from jigsaw.puzzle_roi import extract_puzzle_region_with_metadata


def load_image(path: Path) -> np.ndarray:
    """加载图像为 RGB (HxWx3) uint8."""
    if cv2 is None:
        raise RuntimeError("verify_roi 需要 opencv-python (cv2)")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载图像: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证拼图区域提取：在原图上显示检测到的 ROI，并展示裁剪结果。"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/IMG_0970.PNG",
        help="输入截图路径（默认 examples/IMG_0970.PNG）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将裁剪出的拼图区域保存到此路径",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出可视化窗口（仅打印信息，可与 --output 一起用）",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"文件不存在: {image_path}")

    image = load_image(image_path)
    result = extract_puzzle_region_with_metadata(image)

    x_min, y_min, x_max, y_max = result.bbox
    h, w = image.shape[:2]
    crop_area = (x_max - x_min) * (y_max - y_min)
    total_area = w * h
    ratio_pct = 100.0 * crop_area / total_area if total_area else 0

    print("=== 拼图 ROI 提取结果 ===")
    print(f"输入: {image_path}")
    print(f"原图尺寸: {w} x {h}")
    print(f"检测 bbox: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")
    print(f"裁剪区域尺寸: {x_max - x_min} x {y_max - y_min}")
    print(f"裁剪面积占比: {ratio_pct:.1f}%")
    if result.rows is not None and result.cols is not None:
        print(f"推断网格: {result.rows} 行 x {result.cols} 列")
    else:
        print("推断网格: 未检测到规则网格，使用整图")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            out_bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), out_bgr)
        print(f"已保存裁剪图: {out_path.resolve()}")

    if not args.no_show and plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # 左：原图 + bbox
        axes[0].imshow(image)
        axes[0].plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            "r-",
            linewidth=2,
        )
        axes[0].set_title("原图与检测到的拼图区域 (红框)")
        axes[0].axis("off")
        # 右：裁剪出的拼图区域
        axes[1].imshow(result.image)
        axes[1].set_title("提取的拼图区域")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
