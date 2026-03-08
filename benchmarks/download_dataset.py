"""COCO dataset downloader for real image benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# COCO val2017 URL and info
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_MD5 = "442b8e91e65b32d1cd1e5ecab1cf3c2b"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _get_size_format(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _check_md5(file_path: Path, expected_md5: str) -> bool:
    """Check if file matches expected MD5 hash."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def _download_file(url: Path, dest: Path, show_progress: bool = True) -> None:
    """Download a file using curl (more reliable for large files)."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    # Use curl with progress
    cmd = ["curl", "-L", "-o", str(dest), str(url)]
    if not show_progress:
        cmd.insert(1, "-s")

    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed with return code {result.returncode}")


def download_coco_val2017(
    output_dir: Path,
    skip_annotations: bool = True,
    force: bool = False,
) -> Path:
    """Download COCO val2017 dataset.

    Args:
        output_dir: Directory to save the dataset
        skip_annotations: If True, skip downloading annotations
        force: If True, re-download even if files exist

    Returns:
        Path to the val2017 directory
    """
    output_dir = Path(output_dir)
    val2017_dir = output_dir / "val2017"
    zip_path = output_dir / "val2017.zip"
    ann_zip_path = output_dir / "annotations_trainval2017.zip"

    # Check if already exists
    if val2017_dir.exists() and not force:
        num_images = len(list(val2017_dir.glob("*.jpg")))
        print(f"COCO val2017 already exists at {val2017_dir} ({num_images} images)")
        return val2017_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download validation images
    if not zip_path.exists() or force:
        _download_file(COCO_VAL2017_URL, zip_path)

        # Verify MD5 (optional, can be slow for large files)
        print("Verifying download...")
        if _check_md5(zip_path, COCO_VAL2017_MD5):
            print("  MD5 verified!")
        else:
            print("  Warning: MD5 mismatch, but continuing...")

        # Extract
        print(f"Extracting to {output_dir}...")
        shutil.unpack_archive(zip_path, output_dir)

        # Cleanup zip to save space
        if not force:
            print(f"Removing zip file to save space...")
            zip_path.unlink()
    else:
        print(f"Using existing zip file: {zip_path}")
        if not val2017_dir.exists():
            print(f"Extracting to {output_dir}...")
            shutil.unpack_archive(zip_path, output_dir)

    # Download annotations (optional)
    if not skip_annotations:
        ann_dir = output_dir / "annotations"
        if not ann_dir.exists():
            if not ann_zip_path.exists():
                _download_file(COCO_ANNOTATIONS_URL, ann_zip_path)
            print(f"Extracting annotations...")
            shutil.unpack_archive(ann_zip_path, output_dir)
            ann_zip_path.unlink()

    # Count images
    num_images = len(list(val2017_dir.glob("*.jpg")))
    print(f"\nCOCO val2017 ready at: {val2017_dir}")
    print(f"  Total images: {num_images}")

    return val2017_dir


def verify_dataset(directory: Path, min_images: int = 4000) -> bool:
    """Verify that the dataset is complete.

    Args:
        directory: Path to the val2017 directory
        min_images: Minimum expected number of images

    Returns:
        True if dataset is valid
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return False

    image_files = list(directory.glob("*.jpg"))
    num_images = len(image_files)

    print(f"Found {num_images} images in {directory}")

    if num_images < min_images:
        print(f"Warning: Expected at least {min_images} images, found {num_images}")
        return False

    print("Dataset verification passed!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO val2017 dataset for benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/coco",
        help="Output directory (default: datasets/coco)",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        default=True,
        help="Skip downloading annotations (default: True)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset after download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    val2017_dir = download_coco_val2017(
        output_dir,
        skip_annotations=args.skip_annotations,
        force=args.force,
    )

    if args.verify:
        verify_dataset(val2017_dir)


if __name__ == "__main__":
    main()
