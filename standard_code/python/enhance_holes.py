"""
Parallel CPU hole scoring for greyscale images.

Computes a per-pixel hole score by filling dark holes and measuring the
intensity gain required to fill each pixel.

Author: BIPHUB , University of Oslo
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from skimage.morphology import reconstruction
from skimage.transform import resize
from tqdm import tqdm

# Local imports
import bioimage_pipeline_utils as rp


def _parse_channels(channel_str: str) -> list[int] | None:
    """
    Parse channels from string format like '0 2', '0,2', or None.
    Handles both space-separated and comma-separated formats.
    """
    if channel_str is None:
        return None
    parts = str(channel_str).replace(",", " ").split()
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse channels: {e}")


def _parse_downsample_factor(value: str) -> int:
    """
    Parse downsample factor from CLI.

    Accepts positive integers or the string "none" (treated as 1).
    """
    if value is None:
        return 1

    value_str = str(value).strip().lower()
    if value_str in {"none", "no", "false"}:
        return 1

    try:
        factor = int(value_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid --downsample-factor '{value}': {e}")

    if factor < 1:
        raise argparse.ArgumentTypeError("--downsample-factor must be >= 1 or 'none'")

    return factor


def fill_holes_2d(image: np.ndarray) -> np.ndarray:
    """Fill holes in one 2D greyscale plane using morphological reconstruction."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = reconstruction(seed, inverted, method="dilation")
    return image_max - reconstructed


def _cast_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast values back to a target dtype without min-max normalization."""
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        rounded = np.rint(values)
        clipped = np.clip(rounded, info.min, info.max)
        return clipped.astype(target_dtype)
    return values.astype(target_dtype, copy=False)


def hole_score_2d(plane: np.ndarray, downsample_factor: int = 1) -> np.ndarray:
    """Compute a hole score map for one 2D plane (YX)."""
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")

    h, w = plane.shape

    if downsample_factor > 1:
        ds_h = max(1, h // downsample_factor)
        ds_w = max(1, w // downsample_factor)
        plane_work = resize(
            plane,
            (ds_h, ds_w),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
    else:
        plane_work = plane

    filled = fill_holes_2d(plane_work)
    # Positive difference corresponds to dark-hole depth that was filled.
    score = filled.astype(np.float64) - plane_work.astype(np.float64)
    score[score < 0] = 0.0

    if downsample_factor > 1:
        score = resize(
            score,
            (h, w),
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        score[score < 0] = 0.0

    return score


def run_parallel_cpu(
    img_data: np.ndarray,
    timeout_seconds: float | None = None,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Parallel CPU: distributes XY planes across all CPU cores."""
    if img_data.ndim != 5:
        raise ValueError(f"Expected TCZYX (5D), got shape {img_data.shape}")

    t, c, z, h, w = img_data.shape
    flat = [img_data[ti, ci, zi] for ti in range(t) for ci in range(c) for zi in range(z)]

    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(hole_score_2d, plane, downsample_factor) for plane in flat]
        results: list[np.ndarray] = []
        for future in futures:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            if remaining == 0.0:
                for f in futures:
                    f.cancel()
                raise TimeoutError("run_parallel_cpu exceeded the runtime limit")
            results.append(future.result(timeout=remaining))

    return np.stack(results).reshape(t, c, z, h, w)


def run_sequential_cpu(
    img_data: np.ndarray,
    timeout_seconds: float | None = None,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Sequential CPU fallback for debugging and reproducibility checks."""
    if img_data.ndim != 5:
        raise ValueError(f"Expected TCZYX (5D), got shape {img_data.shape}")

    t, c, z, h, w = img_data.shape
    flat = [img_data[ti, ci, zi] for ti in range(t) for ci in range(c) for zi in range(z)]

    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds
    results: list[np.ndarray] = []

    for plane in flat:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("run_sequential_cpu exceeded the runtime limit")
        results.append(hole_score_2d(plane, downsample_factor=downsample_factor))

    return np.stack(results).reshape(t, c, z, h, w)


def process_single_image(
    input_path: str,
    output_path: str,
    channels: Optional[list[int]] = None,
    mode_3d: bool = False,
    timeout_seconds: float | None = None,
    use_parallel: bool = True,
    output_format: str = "ome.tif",
    downsample_factor: int = 1,
) -> bool:
    """Load one image, compute hole score maps, and save output."""
    try:
        logging.info(f"Loading image: {Path(input_path).name}")
        img = rp.load_tczyx_image(input_path)
    except Exception as e:
        logging.error(f"Failed to load image {Path(input_path).name}: {e}")
        logging.debug(f"File path: {input_path}", exc_info=True)
        return False

    t, c, z, y, x = img.shape
    logging.info(f"  Shape: T={t}, C={c}, Z={z}, Y={y}, X={x}")

    if channels is None:
        channels_to_process = list(range(c))
    else:
        channels_to_process = [ch for ch in channels if 0 <= ch < c]
        if not channels_to_process:
            raise ValueError(f"No valid channels found. Image has {c} channels, requested {channels}")

    logging.info(f"  Processing channels: {channels_to_process}")
    if mode_3d:
        logging.warning("  --mode-3d is accepted for CLI compatibility but this tool scores 2D XY planes")
    logging.info(f"  Processing mode: {'parallel CPU' if use_parallel else 'sequential CPU'}")
    logging.info(f"  Downsample factor: {downsample_factor}")
    logging.info(f"  Input dtype: {img.data.dtype}")

    input_dtype = img.data.dtype
    output_data = img.data.copy()
    target = img.data[:, channels_to_process, :, :, :]

    if use_parallel:
        scored = run_parallel_cpu(
            target,
            timeout_seconds=timeout_seconds,
            downsample_factor=downsample_factor,
        )
    else:
        scored = run_sequential_cpu(
            target,
            timeout_seconds=timeout_seconds,
            downsample_factor=downsample_factor,
        )

    scored_cast = _cast_to_dtype(scored, input_dtype)
    output_data[:, channels_to_process, :, :, :] = scored_cast

    logging.info(f"Saving scored image to: {Path(output_path).name}")
    if output_format == "npy":
        np.save(output_path, output_data)
    else:
        rp.save_tczyx_image(output_data, output_path)
    logging.info("  Done!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hole score maps from greyscale images using CPU parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Hole score on nucleus images
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*nucleus*.tif'
  - --output-folder: '%YAML%/output_data'

- name: Hole score for channels 0 and 2
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '0 2'

- name: Hole score with runtime limit and sequential fallback
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --timeout-seconds: 120
  - --no-parallel

- name: Save output as NumPy arrays
    environment: uv@3.11:convert-to-tif
    commands:
    - python
    - '%REPO%/standard_code/python/enhance_holes.py'
    - --input-search-pattern: '%YAML%/input_data/**/*.tif'
    - --output-folder: '%YAML%/output_data'
    - --output-format: npy

- name: Speed up with XY downsampling
    environment: uv@3.11:convert-to-tif
    commands:
    - python
    - '%REPO%/standard_code/python/enhance_holes.py'
    - --input-search-pattern: '%YAML%/input_data/**/*.tif'
    - --output-folder: '%YAML%/output_data'
    - --downsample-factor: 2

Description:
  Computes per-pixel hole score maps:
  score = fill_holes(image) - image

  Dark enclosed regions get positive values, while unchanged pixels remain 0.

Notes:
  - Input should be greyscale intensity images.
  - --mode-3d is accepted for CLI compatibility with fill_greyscale_holes.py.
    This tool still processes XY planes in parallel across TCZ slices.
    - --downsample-factor > 1 downsamples XY before scoring and upscales back.
        Use this for faster runtime at the cost of smoothing/detail loss.
  - Multi-channel images: specify --channels or process all channels by default.
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help='Glob pattern for input images (e.g., "data/**/*.tif")',
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for scored images",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_hole_score",
        help='Suffix to add to output filenames (default: "_hole_score")',
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["ome.tif", "npy"],
        default="ome.tif",
        help='Output format. Choices: "ome.tif" (default) or "npy".',
    )
    parser.add_argument(
        "--downsample-factor",
        type=_parse_downsample_factor,
        default=1,
        help='Optional XY downsample factor (integer >= 1) before scoring. Use "none" or 1 to disable (default: 1).',
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=None,
        help='Channel indices to process (0-based). Space or comma separated. Examples: --channels "0 2" or --channels "0,2"',
    )
    parser.add_argument(
        "--mode-3d",
        action="store_true",
        help="Accepted for CLI compatibility. Current implementation scores 2D XY planes.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional global runtime limit in seconds for per-file processing.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Do not use parallel processing.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Searching for files: {args.input_search_pattern}")
    input_files = rp.get_files_to_process2(args.input_search_pattern, True)
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")

    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Found {len(input_files)} files to process")
    logging.info(f"Output folder: {args.output_folder}")

    run_plane_parallel = (not args.no_parallel) and (len(input_files) > 1)
    logging.info(f"Internal plane parallelism enabled: {run_plane_parallel}")

    successful = 0
    failed = 0

    for i, input_path in enumerate(tqdm(input_files, desc="Processing files", unit="file"), 1):
        logging.info(f"\n{'=' * 70}")
        logging.info(f"Processing file {i}/{len(input_files)}")

        input_filename = Path(input_path).name
        input_filename_lower = input_filename.lower()
        if input_filename_lower.endswith(".ome.tif"):
            input_name = input_filename[:-8]
        elif input_filename_lower.endswith(".ome.tiff"):
            input_name = input_filename[:-9]
        else:
            input_name = Path(input_path).stem
        output_extension = ".ome.tif" if args.output_format == "ome.tif" else ".npy"
        output_name = f"{input_name}{args.output_suffix}{output_extension}"
        output_path = os.path.join(args.output_folder, output_name)

        try:
            ok = process_single_image(
                input_path=input_path,
                output_path=output_path,
                channels=args.channels,
                mode_3d=args.mode_3d,
                timeout_seconds=args.timeout_seconds,
                use_parallel=run_plane_parallel,
                output_format=args.output_format,
                downsample_factor=args.downsample_factor,
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            logging.debug("Stack trace:", exc_info=True)
            ok = False

        if ok:
            successful += 1
        else:
            failed += 1

    logging.info(f"\n{'=' * 70}")
    logging.info("Processing complete!")
    logging.info(f"Processed {len(input_files)} files: {successful} succeeded, {failed} failed")
    logging.info(f"Output saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
