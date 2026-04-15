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
import tempfile
from pathlib import Path
from typing import Any, Literal

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


def _get_image_dtype(img: Any) -> np.dtype:
    """Return the image dtype without forcing eager full-array loading when possible."""
    if hasattr(img, "dask_data") and img.dask_data is not None and hasattr(img.dask_data, "dtype"):
        return np.dtype(img.dask_data.dtype)
    return np.dtype(img.data.dtype)


def _load_tczyx_block(img: Any, t_index: int, c_index: int) -> np.ndarray:
    """Load a single TCZYX block for one timepoint/channel, using lazy loading when available."""
    if hasattr(img, "dask_data") and img.dask_data is not None:
        return np.asarray(img.dask_data[t_index:t_index + 1, c_index:c_index + 1, :, :, :].compute())
    return np.asarray(img.data[t_index:t_index + 1, c_index:c_index + 1, :, :, :])


def _should_force_sequential(input_paths: list[str]) -> tuple[bool, str | None]:
    """Return whether per-file sequential processing should be forced for memory safety."""
    if any(path.lower().endswith((".h5", ".hdf5")) for path in input_paths):
        return True, "H5 input detected; forcing sequential processing to reduce peak memory usage"

    large_input_threshold = 512 * 1024**2
    total_size = 0
    has_large_input = False

    for path in input_paths:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        total_size += size
        if size >= large_input_threshold:
            has_large_input = True

    if has_large_input:
        return True, "Large input file detected; forcing sequential processing to reduce peak memory usage"
    if len(input_paths) > 1 and total_size >= 2 * 1024**3:
        return True, "Large total batch size detected; forcing sequential processing to reduce peak memory usage"
    return False, None


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
    output_format: str,
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    force: bool,
) -> bool:
    """Load one image, blur, and save."""
    temp_memmap_path: str | None = None
    blurred: np.ndarray | np.memmap | None = None

    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")

        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True

        img = rp.load_tczyx_image(input_path)
        original_dtype = _get_image_dtype(img)
        T, C, Z, Y, X = (int(v) for v in img.shape)
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}, dtype={original_dtype}")
        logger.info("Using block-wise Gaussian blur to reduce peak memory usage")

        # Determine channels to process
        C_total = C
        if channels is None:
            channels_to_process = list(range(C_total))
        else:
            channels_to_process = [c for c in channels if 0 <= c < C_total]
            if not channels_to_process:
                raise ValueError(
                    f"No valid channels found. Image has {C_total} channels, requested {channels}"
                )

        selected_channels = set(channels_to_process)
        logger.info(f"Channels to process: {channels_to_process}")

        fd, temp_memmap_path = tempfile.mkstemp(prefix="gaussian_blur_", suffix=".dat")
        os.close(fd)
        blurred = np.memmap(
            temp_memmap_path,
            dtype=original_dtype,
            mode="w+",
            shape=(T, C, Z, Y, X),
        )

        limits = _dtype_limits(original_dtype)

        for t in range(T):
            for c in range(C_total):
                block = _load_tczyx_block(img, t, c)
                if c in selected_channels:
                    block = gaussian_blur_stack(
                        block,
                        mode=mode,
                        sigma_xy=sigma_xy,
                        sigma_z=sigma_z,
                        truncate=truncate,
                    )

                if limits is not None:
                    lo, hi = limits
                    block = np.clip(block, lo, hi)

                blurred[t:t+1, c:c+1, :, :, :] = np.asarray(block, dtype=original_dtype)

        blurred.flush()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_with_output_format(blurred, output_path, output_format)
        logger.info(f"Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if blurred is not None:
            del blurred
        if temp_memmap_path is not None and os.path.exists(temp_memmap_path):
            try:
                os.remove(temp_memmap_path)
            except OSError:
                logger.debug("Could not remove temporary memmap file: %s", temp_memmap_path)


def process_files(
    input_pattern: str,
    output_folder: str | None,
    output_format: str,
    mode: Literal["2d", "3d"],
    channels: list[int] | None,
    sigma_xy: float,
    sigma_z: float,
    truncate: float,
    maxcores: int | None,
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
        output_extension = rp.output_extension_for_format(output_format, tiff_extension=".tif")
        output_filename = os.path.basename(
            rp.resolve_output_path(input_path, extension=output_extension, suffix="_gauss")
        )
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

    if not no_parallel:
        should_force_sequential, reason = _should_force_sequential([inp for inp, _ in tasks])
        if should_force_sequential:
            logger.warning("%s", reason)
            no_parallel = True

    if no_parallel or len(tasks) == 1:
        logger.info("Processing files sequentially")
        ok = 0
        for inp, out in tasks:
            if process_single_file(inp, out, output_format, mode, channels, sigma_xy, sigma_z, truncate, force):
                ok += 1
        logger.info(f"Done: {ok} succeeded, {len(tasks)-ok} failed")
        return

    # Optional: simple parallelism using processes
    from concurrent.futures import ProcessPoolExecutor, as_completed

    max_workers = rp.resolve_maxcores(maxcores, len(tasks))
    logger.info(f"Processing files in parallel (workers={max_workers})")

    ok = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_file,
                inp, out, output_format, mode, channels, sigma_xy, sigma_z, truncate, force,
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
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/gaussian_blur.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian2d'
  - --mode: 2d
  - --sigma: 1.5

- name: Gaussian blur (3D)
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/gaussian_blur.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data_gaussian3d'
  - --mode: 3d
  - --sigma: 1.0
  - --sigma-z: 0.5

- name: Gaussian blur, only channels 0 and 2
    environment: uv@3.11:default
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
        "--output-format",
        type=str,
        choices=["tif", "npy", "ilastik-h5"],
        default="tif",
        help="Output format (default: tif)",
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
        "--maxcores",
        type=int,
        default=None,
        help="Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). Ignored if --no-parallel is set.",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
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
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        output_format=args.output_format,
        mode=args.mode,
        channels=args.channels,
        sigma_xy=args.sigma,
        sigma_z=args.sigma_z,
        truncate=args.truncate,
        maxcores=args.maxcores,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
