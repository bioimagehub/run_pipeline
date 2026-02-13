"""
Gaussian blur utility for batch image processing (TCZYX-aware).

Applies either per-slice 2D Gaussian blur (on YX for each Z) or full 3D
Gaussian blur (on ZYX), preserving T and C dimensions. Designed to integrate
with the BIPHUB Pipeline Manager via YAML configs for run_pipeline.exe.

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


def gaussian_blur_stack(
    data: np.ndarray,
    mode: Literal["2d", "3d"],
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
) -> np.ndarray:
    """
    Apply Gaussian blur to a TCZYX array.

    - mode "2d": blur only Y and X (per Z slice), sigma = (0,0,0,sigma_xy,sigma_xy)
    - mode "3d": blur Z, Y, X together, sigma = (0,0,sigma_z,sigma_xy,sigma_xy)
    """
    if data.ndim != 5:
        raise ValueError("Expected 5D TCZYX array")

    if mode == "2d":
        sigma = (0.0, 0.0, 0.0, float(sigma_xy), float(sigma_xy))
    else:  # "3d"
        sigma = (0.0, 0.0, float(sigma_z), float(sigma_xy), float(sigma_xy))

    logger.info(f"Applying Gaussian (mode={mode}, sigma={sigma}, truncate={truncate})")
    # Use reflect mode for edges (common in microscopy)
    return ndimage.gaussian_filter(
        data,
        sigma=sigma,
        truncate=truncate,
        mode="reflect",
    )


def process_single_file(
    input_path: str,
    output_path: str,
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    force: bool,
) -> bool:
    """Load one image, blur, and save."""
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
            blurred = gaussian_blur_stack(
                img.data,
                mode=mode,
                sigma_xy=sigma_xy,
                sigma_z=sigma_z,
                truncate=truncate,
            )
        else:
            # Preserve non-selected channels, blur only selected ones
            output = img.data.astype(np.float32, copy=True)
            for c in channels_to_process:
                slice5d = img.data[:, c:c+1, :, :, :]
                blurred_c = gaussian_blur_stack(
                    slice5d,
                    mode=mode,
                    sigma_xy=sigma_xy,
                    sigma_z=sigma_z,
                    truncate=truncate,
                )
                output[:, c:c+1, :, :, :] = blurred_c
            blurred = output

        # Safe cast back to original dtype
        limits = _dtype_limits(original_dtype)
        if limits is not None:
            lo, hi = limits
            blurred = np.clip(blurred, lo, hi)
            blurred = blurred.astype(original_dtype, copy=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(blurred, output_path)
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
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
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
        output_folder = (base_dir or ".") + "_gaussian"

    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    for input_path in input_files:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{basename}_gauss.tif"
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path))

    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Mode: {mode}, sigma_xy={sigma_xy}, sigma_z={sigma_z}, truncate={truncate}")
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
            if process_single_file(inp, out, mode, channels, sigma_xy, sigma_z, truncate, force):
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
                inp, out, mode, channels, sigma_xy, sigma_z, truncate, force,
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
        description="Apply Gaussian blur to images (TCZYX-aware).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Gaussian blur (2D per Z)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/gaussian_blur.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian2d'
  - --mode: 2d
  - --sigma: 1.5

- name: Gaussian blur (3D)
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/gaussian_blur.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian3d'
  - --mode: 3d
  - --sigma: 1.0
  - --sigma-z: 0.5

- name: Gaussian blur, only channels 0 and 2
  environment: uv@3.11:image-filters
  commands:
  - python
  - '%REPO%/standard_code/python/gaussian_blur.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian_ch02'
  - --mode: 2d
  - --sigma: 1.2
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
        help="Output folder (default: <input_root>_gaussian)",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="2d = blur YX per Z slice; 3d = blur ZYX",
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
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma for Y and X (pixels)",
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=0.0,
        help="Sigma for Z (pixels) when mode=3d",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="Truncate Gaussian at this many sigmas",
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
        print(f"gaussian_blur.py version: {version}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        mode=args.mode,
        channels=args.channels,
        sigma_xy=args.sigma,
        sigma_z=args.sigma_z,
        truncate=args.truncate,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
