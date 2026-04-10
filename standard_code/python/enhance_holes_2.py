"""
Hole enhancement for greyscale images using variance-based ROI masking.

Identifies dark holes within bright objects using:
  1. Variance-based ROI detection (Otsu threshold on local variance)
  2. Per-region hole scoring (morphological reconstruction within each ROI)
  3. Optional integer max-pool downsampling for faster ROI detection
  4. Median filter smoothing of output score map

Processes TCZYX images. Each (channel, Z-slice) T-stack is handled independently.
File-level CPU parallelism is used when more than one input file is found.

Author: BIPHUB, University of Oslo
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction
from tqdm import tqdm

import bioimage_pipeline_utils as rp


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_channels(channel_str: str) -> list[int] | None:
    """Parse channels from '0 2' or '0,2' format."""
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
    """Parse downsample factor; accepts positive integers or 'none' (= 1)."""
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


def _cast_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast values back to target dtype without any min-max normalization."""
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        rounded = np.rint(values)
        clipped = np.clip(rounded, info.min, info.max)
        return clipped.astype(target_dtype)
    return values.astype(target_dtype, copy=False)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def integer_downsample_max(img_3d: np.ndarray, factor: int) -> np.ndarray:
    """Downsample (T, H, W) by an integer factor using max-pooling."""
    if img_3d.ndim != 3:
        raise ValueError(f"Expected 3D image (T, H, W), got shape {img_3d.shape}")
    if factor == 1:
        return img_3d
    t, h, w = img_3d.shape
    h_ds = h // factor
    w_ds = w // factor
    if h_ds < 1 or w_ds < 1:
        raise ValueError(
            f"factor={factor} too large for spatial shape {(h, w)}; "
            f"downsampled shape would be {(h_ds, w_ds)}"
        )
    trimmed = img_3d[:, : h_ds * factor, : w_ds * factor]
    return trimmed.reshape(t, h_ds, factor, w_ds, factor).max(axis=(2, 4))


def fill_holes_2d(image: np.ndarray) -> np.ndarray:
    """Fill bright-object holes via erosion-based morphological reconstruction."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0:
        return image.copy()
    if image.min() == image.max():
        return image.copy()
    seed = image.copy()
    seed[1:-1, 1:-1] = image.max()
    return reconstruction(seed, image, method="erosion")


def vectorized_variance(img_3d: np.ndarray, variance_sigma: float = 4.0) -> np.ndarray:
    """Compute local variance map across a (T, H, W) stack."""
    if img_3d.ndim != 3:
        raise ValueError(f"Expected 3D image (T, H, W), got shape {img_3d.shape}")
    if variance_sigma <= 0:
        raise ValueError(f"variance_sigma must be > 0, got {variance_sigma}")
    img_f32 = img_3d.astype(np.float32, copy=False)
    mean = gaussian_filter(img_f32, sigma=(0.0, variance_sigma, variance_sigma))
    sq_mean = gaussian_filter(img_f32 * img_f32, sigma=(0.0, variance_sigma, variance_sigma))
    variance = sq_mean - mean * mean
    np.maximum(variance, 0, out=variance)
    return variance


def vectorized_otsu(variance_3d: np.ndarray) -> tuple[np.ndarray, float]:
    """Return binary mask and Otsu threshold from a variance map."""
    if variance_3d.ndim != 3:
        raise ValueError(f"Expected 3D variance map (T, H, W), got shape {variance_3d.shape}")
    threshold = float(threshold_otsu(variance_3d))
    return variance_3d > threshold, threshold


def hole_score_2d(
    plane: np.ndarray,
    roi_mask: np.ndarray,
    variance_sigma: float = 4.0,
    roi_padding: int | None = None,
    min_region_area: int = 20,
) -> np.ndarray:
    """
    Score dark holes inside each labeled ROI region using morphological fill.

    Returns a score map (same shape as plane) where positive values indicate
    the depth of hole that was filled. Regions smaller than min_region_area
    are skipped. A padded bounding box gives fill_holes_2d sufficient context.
    """
    if plane.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {plane.shape}")
    if roi_mask.shape != plane.shape:
        raise ValueError(
            f"roi_mask shape {roi_mask.shape} must match plane shape {plane.shape}"
        )
    h, w = plane.shape
    pad = roi_padding if roi_padding is not None else max(8, int(np.ceil(4 * variance_sigma)))
    labels = label(roi_mask)
    score = np.zeros_like(plane)

    for region in regionprops(labels):
        if region.area < min_region_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        row0 = max(0, min_row - pad)
        col0 = max(0, min_col - pad)
        row1 = min(h, max_row + pad)
        col1 = min(w, max_col + pad)

        plane_window = plane[row0:row1, col0:col1]
        filled_window = fill_holes_2d(plane_window)
        if filled_window.dtype != plane_window.dtype:
            filled_window = filled_window.astype(plane_window.dtype, copy=False)

        local_score = np.zeros_like(plane_window)
        np.subtract(
            filled_window,
            plane_window,
            out=local_score,
            where=filled_window > plane_window,
        )
        np.maximum(
            score[row0:row1, col0:col1],
            local_score,
            out=score[row0:row1, col0:col1],
        )

    return score


def _score_tyx_stack(
    img_tyx: np.ndarray,
    downsample_factor: int,
    variance_sigma: float,
    median_sigma: int,
    roi_padding: int | None,
    min_region_area: int,
) -> np.ndarray:
    """
    Full scoring pipeline for one (T, H, W) greyscale stack.

    ROI detection runs at downsampled resolution; hole scoring and output
    are at full resolution. Returns array with same shape and dtype as input.
    """
    t, h, w = img_tyx.shape
    input_dtype = img_tyx.dtype

    # ROI detection at (optionally) downsampled resolution
    img_ds = integer_downsample_max(img_tyx, factor=downsample_factor)
    variance_ds = vectorized_variance(img_ds, variance_sigma=variance_sigma)
    roi_mask_ds, _ = vectorized_otsu(variance_ds)
    # Max-project across time: any frame with a high-variance region contributes
    roi_mask_2d_ds = roi_mask_ds.max(axis=0)

    # Upsample 2D ROI mask back to full resolution using nearest-neighbour repeat
    ds_h, ds_w = roi_mask_2d_ds.shape
    up_h = int(np.ceil(h / ds_h))
    up_w = int(np.ceil(w / ds_w))
    roi_mask_full = np.repeat(
        np.repeat(roi_mask_2d_ds, up_h, axis=0), up_w, axis=1
    )[:h, :w]

    # Per-plane hole scoring at full resolution.
    # Keep temporary arrays in float32 and write each frame directly to the
    # output dtype to avoid allocating a full float64 (T, H, W) volume.
    img_out = np.empty((t, h, w), dtype=input_dtype)
    for ti in range(t):
        score_plane = hole_score_2d(
            plane=img_tyx[ti].astype(np.float32, copy=False),
            roi_mask=roi_mask_full,
            variance_sigma=variance_sigma,
            roi_padding=roi_padding,
            min_region_area=min_region_area,
        )

        # Optional median smoothing per frame (equivalent to 3D filter with
        # temporal kernel size 1).
        if median_sigma > 1:
            score_plane = median_filter(
                score_plane, size=(int(median_sigma), int(median_sigma))
            )

        np.maximum(score_plane, 0, out=score_plane)
        img_out[ti] = _cast_to_dtype(score_plane, input_dtype)

    return img_out


# ---------------------------------------------------------------------------
# Per-file processing (called by parallel workers)
# ---------------------------------------------------------------------------

def process_single_image(
    input_path: str,
    output_path: str,
    channels: Optional[list[int]] = None,
    output_format: str = "ome.tif",
    downsample_factor: int = 1,
    variance_sigma: float = 4.0,
    median_sigma: int = 2,
    roi_padding: int | None = None,
    min_region_area: int = 20,
) -> bool:
    """Load one TCZYX image, compute hole score maps per channel/Z, and save."""
    try:
        logging.info(f"Loading: {Path(input_path).name}")
        img = rp.load_tczyx_image(input_path)
    except Exception as e:
        logging.error(f"Failed to load {Path(input_path).name}: {e}")
        logging.debug(f"Path: {input_path}", exc_info=True)
        return False

    t, c, z, y, x = img.shape
    input_dtype = img.data.dtype
    logging.info(f"  Shape: T={t}, C={c}, Z={z}, Y={y}, X={x}  dtype={input_dtype}")

    if channels is None:
        channels_to_process = list(range(c))
    else:
        channels_to_process = [ch for ch in channels if 0 <= ch < c]
        if not channels_to_process:
            raise ValueError(
                f"No valid channels. Image has {c} channels, requested {channels}"
            )

    logging.info(
        f"  Channels: {channels_to_process}  downsample={downsample_factor}"
        f"  variance_sigma={variance_sigma}  median_sigma={median_sigma}"
    )

    if channels is None or len(channels_to_process) == c:
        # All channels are recomputed, so no need to duplicate full input volume.
        output_data = np.empty_like(img.data)
    else:
        output_data = img.data.copy()

    for ci in channels_to_process:
        for zi in range(z):
            tyx = img.data[:, ci, zi, :, :]
            scored = _score_tyx_stack(
                tyx,
                downsample_factor=downsample_factor,
                variance_sigma=variance_sigma,
                median_sigma=median_sigma,
                roi_padding=roi_padding,
                min_region_area=min_region_area,
            )
            output_data[:, ci, zi, :, :] = scored

    logging.info(f"  Saving -> {Path(output_path).name}")
    if output_format == "npy":
        np.save(output_path, output_data)
    else:
        rp.save_tczyx_image(output_data, output_path)
    logging.info("  Done!")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hole score maps using variance-based ROI masking "
            "and per-region morphological fill"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Hole score (default settings)
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_2.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'

- name: Hole score channels 0 and 2 with 4x speed-up
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_2.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '0 2'
  - --downsample-factor: 4

- name: Hole score with custom variance and median settings
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_2.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --variance-sigma: 6.0
  - --median-sigma: 3

- name: Save as NumPy arrays, sequential processing
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_2.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --output-format: npy
  - --no-parallel

Description:
  Scores dark holes inside bright objects using:
    1. Local variance map computed over the full T-stack
    2. Otsu threshold on variance to find active ROI regions
    3. Morphological erosion-fill within each padded ROI bounding box
    4. Optional median filter smoothing of the final score map

  Output dtype always matches input dtype (no min-max rescaling).

Notes:
  - Designed for 2D + time data. Z-slices are scored independently.
  - --downsample-factor applies integer max-pool downsampling for ROI
    detection only. Hole scoring and output are always at full resolution.
  - --variance-sigma controls the spatial scale of ROI detection.
    Larger values capture larger structures; try 2-8 for typical nuclei.
  - --median-sigma smooths the output score map. Set to 1 to disable.
  - File-level parallelism is used when >1 file is found unless
    --no-parallel is set. Processing inside each file is always sequential.
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
        help='Suffix appended to output filenames (default: "_hole_score")',
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["ome.tif", "npy"],
        default="ome.tif",
        help='Output format: "ome.tif" (default) or "npy"',
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=None,
        help='Channel indices to process (0-based). Space or comma separated, e.g. "0 2"',
    )
    parser.add_argument(
        "--downsample-factor",
        type=_parse_downsample_factor,
        default=1,
        help=(
            'Integer max-pool downsample factor for ROI detection (default: 1). '
            'Use "none" to disable. Hole scoring always runs at full resolution.'
        ),
    )
    parser.add_argument(
        "--variance-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for local variance used in ROI detection (default: 4.0)",
    )
    parser.add_argument(
        "--median-sigma",
        type=int,
        default=2,
        help="Kernel size for median filter on output score map. Set to 1 to disable (default: 2)",
    )
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=None,
        help="Padding (pixels) around each ROI bounding box. Default: max(8, 4*variance_sigma)",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=20,
        help="Minimum labeled region area in pixels to score (default: 20)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Process files sequentially instead of in parallel",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Searching for files: {args.input_search_pattern}")
    input_files = rp.get_files_to_process2(args.input_search_pattern, True)
    if not input_files:
        raise FileNotFoundError(f"No files found matching: {args.input_search_pattern}")

    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Found {len(input_files)} files")
    logging.info(f"Output folder: {args.output_folder}")

    use_parallel = (not args.no_parallel) and (len(input_files) > 1)
    logging.info(f"File-level parallelism: {use_parallel}")

    def _make_output_path(input_path: str) -> str:
        fname = Path(input_path).name
        fname_lower = fname.lower()
        if fname_lower.endswith(".ome.tif"):
            stem = fname[:-8]
        elif fname_lower.endswith(".ome.tiff"):
            stem = fname[:-9]
        else:
            stem = Path(input_path).stem
        ext = ".ome.tif" if args.output_format == "ome.tif" else ".npy"
        return os.path.join(args.output_folder, f"{stem}{args.output_suffix}{ext}")

    def _process(input_path: str) -> bool:
        try:
            return process_single_image(
                input_path=input_path,
                output_path=_make_output_path(input_path),
                channels=args.channels,
                output_format=args.output_format,
                downsample_factor=args.downsample_factor,
                variance_sigma=args.variance_sigma,
                median_sigma=args.median_sigma,
                roi_padding=args.roi_padding,
                min_region_area=args.min_region_area,
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            logging.debug("Stack trace:", exc_info=True)
            return False

    if use_parallel:
        from contextlib import contextmanager
        from joblib import Parallel, delayed

        @contextmanager
        def _tqdm_joblib(tqdm_object):
            import joblib
            class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)
            old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            try:
                yield tqdm_object
            finally:
                joblib.parallel.BatchCompletionCallBack = old_batch_callback
                tqdm_object.close()

        logging.info("Processing files in parallel...")
        n_jobs = min(2, os.cpu_count() or 1)
        logging.info(f"Parallel workers: {n_jobs}")
        with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
            with _tqdm_joblib(pbar):
                results = list(Parallel(n_jobs=n_jobs)(
                    delayed(_process)(f) for f in input_files
                ))
    else:
        logging.info("Processing files sequentially...")
        results = []
        for i, f in enumerate(tqdm(input_files, desc="Processing files", unit="file"), 1):
            logging.info(f"\n{'=' * 70}")
            logging.info(f"File {i}/{len(input_files)}")
            results.append(_process(f))

    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    logging.info(f"\n{'=' * 70}")
    logging.info(f"Complete: {successful} succeeded, {failed} failed")
    logging.info(f"Output: {args.output_folder}")


if __name__ == "__main__":
    main()




