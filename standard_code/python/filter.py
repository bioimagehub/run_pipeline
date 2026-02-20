"""
Image filter utility for batch processing (TCZYX-aware).

Applies various filters (mean, min, max, median, gaussian, etc.) either per-slice
2D (on YX for each Z) or full 3D (on ZYX), preserving T and C dimensions. Designed
to integrate with the BIPHUB Pipeline Manager via YAML configs for run_pipeline.exe.

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage

import bioimage_pipeline_utils as rp


# Module-level logger
logger = logging.getLogger(__name__)


def _dtype_limits(dtype: np.dtype) -> tuple[float, float] | None:
    """Return min/max limits for integer dtypes; None for float types."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        return None
    # Default: treat like float (no clipping)
    return None


def apply_filter(
    data: np.ndarray,
    method: Literal["mean", "min", "max", "median", "gaussian"],
    mode: Literal["2d", "3d"],
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    size_y: int,
    size_x: int,
    size_z: int,
) -> np.ndarray:
    """
    Apply a filter to a TCZYX array.

    - mode "2d": apply filter only on Y and X (per Z slice)
    - mode "3d": apply filter on Z, Y, X together

    Parameters:
        data: 5D TCZYX array
        method: Filter method (mean, min, max, median, gaussian)
        mode: "2d" or "3d"
        sigma_xy: Sigma for Y,X (for gaussian)
        sigma_z: Sigma for Z (for gaussian, 3d mode)
        truncate: Truncate gaussian at this many sigmas
        size_y: Kernel size for Y dimension (for non-gaussian filters)
        size_x: Kernel size for X dimension (for non-gaussian filters)
        size_z: Kernel size for Z dimension (for non-gaussian filters)
    """
    if data.ndim != 5:
        raise ValueError("Expected 5D TCZYX array")

    logger.info(f"Applying {method} filter (mode={mode}, size=(Y:{size_y}, X:{size_x}, Z:{size_z}))")

    if method == "gaussian":
        if mode == "2d":
            sigma = (0.0, 0.0, 0.0, float(sigma_xy), float(sigma_xy))
        else:  # "3d"
            sigma = (0.0, 0.0, float(sigma_z), float(sigma_xy), float(sigma_xy))

        return ndimage.gaussian_filter(
            data,
            sigma=sigma,
            truncate=truncate,
            mode="reflect",
        )

    elif method == "mean":
        if mode == "2d":
            footprint = np.ones((1, 1, 1, size_y, size_x), dtype=bool)
        else:  # "3d"
            footprint = np.ones((1, 1, size_z, size_y, size_x), dtype=bool)
        return ndimage.uniform_filter(data, footprint=footprint, mode="reflect")

    elif method == "median":
        if mode == "2d":
            # Apply median per Z slice
            output = np.zeros_like(data, dtype=float)
            T, C, Z, Y, X = data.shape
            for t in range(T):
                for c in range(C):
                    for z in range(Z):
                        output[t, c, z, :, :] = ndimage.median_filter(
                            data[t, c, z, :, :],
                            size=(size_y, size_x),
                            mode="reflect",
                        )
            return output
        else:  # "3d"
            return ndimage.median_filter(
                data,
                size=(1, 1, size_z, size_y, size_x),
                mode="reflect",
            )

    elif method == "min":
        if mode == "2d":
            footprint = np.ones((1, 1, 1, size_y, size_x), dtype=bool)
        else:  # "3d"
            footprint = np.ones((1, 1, size_z, size_y, size_x), dtype=bool)
        return ndimage.minimum_filter(data, footprint=footprint, mode="reflect")

    elif method == "max":
        if mode == "2d":
            footprint = np.ones((1, 1, 1, size_y, size_x), dtype=bool)
        else:  # "3d"
            footprint = np.ones((1, 1, size_z, size_y, size_x), dtype=bool)
        return ndimage.maximum_filter(data, footprint=footprint, mode="reflect")

    else:
        raise ValueError(f"Unknown filter method: {method}")


def process_single_file(
    input_path: str,
    output_path: str,
    method: Literal["mean", "min", "max", "median", "gaussian"],
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    size_y: int,
    size_x: int,
    size_z: int,
    force: bool,
) -> bool:
    """Load one image, filter, and save."""
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")

        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True

        img = rp.load_tczyx_image(input_path)
        original_dtype = img.data.dtype
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}, dtype={original_dtype}")

        # Determine channels to process
        C_total = img.shape[1]
        if channels is None:
            channels_to_process = list(range(C_total))
        else:
            channels_to_process = [c for c in channels if 0 <= c < C_total]
            if not channels_to_process:
                raise ValueError(
                    f"No valid channels found. Image has {C_total} channels, requested {channels}"
                )

        logger.info(f"Channels to process: {channels_to_process}")

        if len(channels_to_process) == C_total:
            # Process entire stack
            filtered = apply_filter(
                img.data,
                method=method,
                mode=mode,
                sigma_xy=sigma_xy,
                sigma_z=sigma_z,
                truncate=truncate,
                size_y=size_y,
                size_x=size_x,
                size_z=size_z,
            )
        else:
            # Preserve non-selected channels, filter only selected ones
            output = img.data.astype(np.float32, copy=True)
            for c in channels_to_process:
                slice5d = img.data[:, c:c+1, :, :, :]
                filtered_c = apply_filter(
                    slice5d,
                    method=method,
                    mode=mode,
                    sigma_xy=sigma_xy,
                    sigma_z=sigma_z,
                    truncate=truncate,
                    size_y=size_y,
                    size_x=size_x,
                    size_z=size_z,
                )
                output[:, c:c+1, :, :, :] = filtered_c
            filtered = output

        # Safe cast back to original dtype
        limits = _dtype_limits(original_dtype)
        if limits is not None:
            lo, hi = limits
            filtered = np.clip(filtered, lo, hi)
            filtered = filtered.astype(original_dtype, copy=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(filtered, output_path)
        logger.info(f"Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    input_pattern: str,
    output_folder: str | None,
    method: Literal["mean", "min", "max", "median", "gaussian"],
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    size_y: int,
    size_x: int,
    size_z: int,
    no_parallel: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Process many files matching the search pattern."""
    search_subfolders = "**" in input_pattern
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not input_files:
        logger.error(f"No files matched pattern: {input_pattern}")
        return

    logger.info(f"Found {len(input_files)} files")

    # Default output folder next to the input root
    if output_folder is None:
        base_dir = os.path.dirname(input_pattern.replace("**/", "").replace("*", ""))
        output_folder = (base_dir or ".") + f"_{method}_filtered"

    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    for input_path in input_files:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{basename}_{method}.tif"
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path))

    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Method: {method}, mode={mode}, size=(Y:{size_y}, X:{size_x}, Z:{size_z})")
        if method == "gaussian":
            print(f"[DRY RUN]   sigma_xy={sigma_xy}, sigma_z={sigma_z}, truncate={truncate}")
        if channels is None:
            print("[DRY RUN] Channels: all")
        else:
            print(f"[DRY RUN] Channels: {channels}")
        for inp, out in tasks[:5]:
            print(f"[DRY RUN]   {os.path.basename(inp)} -> {os.path.basename(out)}")
        if len(tasks) > 5:
            print(f"[DRY RUN]   ... and {len(tasks) - 5} more files")
        return

    if no_parallel or len(tasks) == 1:
        logger.info("Processing files sequentially")
        ok = 0
        for inp, out in tasks:
            if process_single_file(
                inp, out, method, mode, channels, sigma_xy, sigma_z, truncate, size_y, size_x, size_z, force
            ):
                ok += 1
        logger.info(f"Done: {ok} succeeded, {len(tasks)-ok} failed")
        return

    # Optional: simple parallelism using processes
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cpu_count = os.cpu_count() or 1
    max_workers = max(cpu_count - 1, 1)
    logger.info(f"Processing files in parallel (workers={max_workers})")

    ok = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_file,
                inp, out, method, mode, channels, sigma_xy, sigma_z, truncate, size_y, size_x, size_z, force,
            )
            for inp, out in tasks
        ]
        for f in as_completed(futures):
            try:
                if f.result():
                    ok += 1
            except Exception as e:  # pragma: no cover
                logger.error(f"Task failed: {e}")

    logger.info(f"Done: {ok} succeeded, {len(tasks)-ok} failed")


def _parse_size(size_str: str) -> tuple[int, int, int]:
    """
    Parse kernel size(s) from string format.

    - Single value: applies to Y, X, Z equally
    - Two values: first for Y/X, second for Z
    - Three values: Y, X, Z individually

    Examples:
      '3' → (3, 3, 3)
      '3 5' → (3, 3, 5)
      '3 4 5' → (3, 4, 5)
    """
    if size_str is None:
        return (3, 3, 3)

    parts = str(size_str).replace(',', ' ').split()
    if not parts:
        return (3, 3, 3)

    try:
        values = [int(p) for p in parts]
        if len(values) == 1:
            val = values[0]
            return (val, val, val)
        elif len(values) == 2:
            return (values[0], values[0], values[1])
        elif len(values) == 3:
            return (values[0], values[1], values[2])
        else:
            raise argparse.ArgumentTypeError(
                f"Expected 1-3 size values, got {len(values)}"
            )
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(f"Could not parse size: {e}")


def _parse_channels(channel_str: str) -> list[int] | None:
    """
    Parse channels from string format like '0 2', '0,2', or None.
    Handles both space-separated and comma-separated formats.
    """
    if channel_str is None:
        return None
    # Replace commas with spaces and split
    parts = str(channel_str).replace(',', ' ').split()
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse channels: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply various filters to images (TCZYX-aware).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Mean filter (2D per Z, uniform 3x3x3)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_mean2d'
  - --method: mean
  - --size: 3

- name: Median filter (3D, XY=3 Z=5)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_median3d'
  - --method: median
  - --mode: 3d
  - --size: '3 5'

- name: Max filter (2D, Y=3 X=5)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_max'
  - --method: max
  - --size: '3 5'

- name: Min filter (3D, Y=3 X=4 Z=5)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_min'
  - --method: min
  - --mode: 3d
  - --size: '3 4 5'

- name: Gaussian blur (3D)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian3d'
  - --method: gaussian
  - --mode: 3d
  - --sigma: 1.0
  - --sigma-z: 0.5

- name: Mean filter, only channels 0 and 2
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/filter.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_mean_ch02'
  - --method: mean
  - --size: 3
  - --channels: '0 2'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards; use '**' to recurse)",
    )
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: <input_root>_<method>_filtered)",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "min", "max", "median", "gaussian"],
        default="gaussian",
        help="Filter method to apply",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="2d = filter YX per Z slice; 3d = filter ZYX",
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=None,
        help=(
            "Channel indices to process (0-based). Space or comma separated. "
            "Examples: --channels '0 2' or --channels '0,2'"
        ),
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=(3, 3, 3),
        help=(
            "Kernel size(s) for non-gaussian filters. "
            "Single value applies to all dims; "
            "two values: Y/X then Z; "
            "three values: Y, X, Z individually. "
            "Examples: --size 3 or --size '3 5' or --size '3 4 5'"
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma for Y and X (pixels, for gaussian)",
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=0.0,
        help="Sigma for Z (pixels) when method=gaussian and mode=3d",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="Truncate Gaussian at this many sigmas (for gaussian method)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if outputs exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned actions without executing",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args()

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"filter.py version: {version}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    size_y, size_x, size_z = args.size

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        method=args.method,
        mode=args.mode,
        channels=args.channels,
        sigma_xy=args.sigma,
        sigma_z=args.sigma_z,
        truncate=args.truncate,
        size_y=size_y,
        size_x=size_x,
        size_z=size_z,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
