"""
Hole enhancement for greyscale images based on an ImageJ prototype.

This implementation reproduces the prototype logic per 2D plane:
  1. Fill greyscale holes (morphological reconstruction, erosion mode)
  2. Compute hole score = filled - input (positive values only)
  3. Build foreground mask from Otsu threshold ("dark" mode equivalent)
  4. Apply ImageJ-like selection smoothing: enlarge +5, -10, +5
  5. Zero score values outside the smoothed foreground (background)

Processes TCZYX images. Each (channel, Z-slice) T-stack is handled independently.
File-level CPU parallelism is used when more than one input file is found.

The original ImageJ prototype text is preserved in IMAGEJ_PROTOTYPE for comparison.

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
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import disk, reconstruction
from skimage.morphology.binary import  binary_dilation, binary_erosion
from tqdm import tqdm

import bioimage_pipeline_utils as rp


IMAGEJ_PROTOTYPE = r'''This is a working ImageJ prototype, please try to re-create this in Python. The code is not optimized, but it should work.
    Keep this not for now and do not delete ti so that I can compare  the logic
    make a cli similar to #enhance_holes_2.py, but with the new logic.

run("Close All");

run("Bio-Formats Importer", "open=E:/Oyvind/OF_Training/macropinosome/Training_input/Kia_Wee/250225_RPE-mNG-Phafin2_BSD_10ul_001_1.ome.tif color_mode=Composite rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack");
// makeRectangle(536, 1517, 84, 84);

run("Duplicate...", "title=input duplicate range=1-10");

run("Select None");
run("Duplicate...", "title=filled duplicate range=1-10");

// Fill greyscale holes
selectWindow("filled");
fill_holes()

// find regions that increase
imageCalculator("Subtract create stack", "filled","input");
rename("hole_score");

for (i = 1; i <= nSlices; i++) {

    // Define background pixels
    selectWindow("input");
    setSlice(i);
    run("Select None");
    setAutoThreshold("Otsu dark no-reset");
    run("Create Selection");
    run("Enlarge...", "enlarge=5 pixel");
    run("Enlarge...", "enlarge=-10 pixel");
    run("Enlarge...", "enlarge=5 pixel");
    run("Make Inverse"); // Defines background

    // Delete background
    selectImage("hole_score");
    setSlice(i);
    run("Select None");
    run("Restore Selection");
    getRawStatistics(nPixels, mean, min, max, std, histogram);
    changeValues(min, max, 0);
    run("Select None");
}

selectWindow("input");
run("Select None");
selectImage("hole_score");

function fill_holes() {
    title = getTitle();

    n = nSlices();

    for (i = 1; i <= n; i++) {
        setSlice(i);

        // Duplicate current slice
        run("Duplicate...", "title=temp_slice");

        // Process duplicate
        selectWindow("temp_slice");
        run("Fill Holes (Binary/Gray)");

        // Copy result
        run("Select All");
        run("Copy");

        // Paste back into original stack
        selectWindow(title);
        setSlice(i);
        run("Paste");

        // Cleanup
        selectWindow("temp_slice");
        close();
        selectImage("temp_slice-fillHoles");
        close();
    }
}
'''


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
    """Cast values to target dtype without normalization."""
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        rounded = np.rint(values)
        clipped = np.clip(rounded, info.min, info.max)
        return clipped.astype(target_dtype)
    return values.astype(target_dtype, copy=False)


# ---------------------------------------------------------------------------
# Core algorithm (ImageJ prototype recreation)
# ---------------------------------------------------------------------------

def fill_holes_2d(image: np.ndarray) -> np.ndarray:
    """Fill greyscale holes via erosion-based morphological reconstruction."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0:
        return image.copy()
    if image.min() == image.max():
        return image.copy()
    seed = image.copy()
    seed[1:-1, 1:-1] = image.max()
    return reconstruction(seed, image, method="erosion")


def _integer_downsample_max_2d(img_2d: np.ndarray, factor: int) -> np.ndarray:
    """Downsample 2D image by integer max-pooling."""
    if factor == 1:
        return img_2d
    h, w = img_2d.shape
    h_ds = h // factor
    w_ds = w // factor
    if h_ds < 1 or w_ds < 1:
        raise ValueError(
            f"factor={factor} too large for spatial shape {(h, w)}; "
            f"downsampled shape would be {(h_ds, w_ds)}"
        )
    trimmed = img_2d[: h_ds * factor, : w_ds * factor]
    return trimmed.reshape(h_ds, factor, w_ds, factor).max(axis=(1, 3))


def _upsample_nearest_2d(mask_2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsampling from downsampled mask to full resolution."""
    h, w = target_shape
    ds_h, ds_w = mask_2d.shape
    up_h = int(np.ceil(h / ds_h))
    up_w = int(np.ceil(w / ds_w))
    return np.repeat(np.repeat(mask_2d, up_h, axis=0), up_w, axis=1)[:h, :w]


def _smooth_selection_like_imagej(
    foreground_mask: np.ndarray,
    footprint_enlarge_1: np.ndarray | None,
    footprint_shrink: np.ndarray | None,
    footprint_enlarge_2: np.ndarray | None,
) -> np.ndarray:
    """Apply ImageJ-like selection sequence: enlarge +a, -b, +c."""
    smoothed = foreground_mask.astype(bool, copy=True)
    if footprint_enlarge_1 is not None:
        smoothed = binary_dilation(smoothed, footprint=footprint_enlarge_1)
    if footprint_shrink is not None:
        smoothed = binary_erosion(smoothed, footprint=footprint_shrink)
    if footprint_enlarge_2 is not None:
        smoothed = binary_dilation(smoothed, footprint=footprint_enlarge_2)
    return smoothed


def _build_selection_footprints(
    enlarge_px_1: int,
    shrink_px: int,
    enlarge_px_2: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Build and cache morphology footprints once per stack."""
    fp_enlarge_1 = disk(int(enlarge_px_1)) if enlarge_px_1 > 0 else None
    fp_shrink = disk(int(shrink_px)) if shrink_px > 0 else None
    fp_enlarge_2 = disk(int(enlarge_px_2)) if enlarge_px_2 > 0 else None
    return fp_enlarge_1, fp_shrink, fp_enlarge_2


def _compute_foreground_mask_2d(
    plane: np.ndarray,
    downsample_factor: int,
    footprint_enlarge_1: np.ndarray | None,
    footprint_shrink: np.ndarray | None,
    footprint_enlarge_2: np.ndarray | None,
) -> np.ndarray:
    """Compute foreground mask with Otsu threshold (dark background assumption)."""
    plane_f32 = plane.astype(np.float32, copy=False)
    if downsample_factor > 1:
        plane_ds = _integer_downsample_max_2d(plane_f32, downsample_factor)
    else:
        plane_ds = plane_f32

    if plane_ds.min() == plane_ds.max():
        fg_ds = np.zeros_like(plane_ds, dtype=bool)
    else:
        otsu_thr = float(threshold_otsu(plane_ds))
        fg_ds = plane_ds > otsu_thr

    fg_full = (
        _upsample_nearest_2d(fg_ds, plane.shape)
        if downsample_factor > 1
        else fg_ds
    )
    return _smooth_selection_like_imagej(
        fg_full,
        footprint_enlarge_1=footprint_enlarge_1,
        footprint_shrink=footprint_shrink,
        footprint_enlarge_2=footprint_enlarge_2,
    )


def _score_tyx_stack_imagej_style(
    img_tyx: np.ndarray,
    downsample_factor: int,
    median_sigma: int,
    enlarge_px_1: int,
    shrink_px: int,
    enlarge_px_2: int,
    mask_background: bool,
    prezero_background_before_fill: bool,
    time_start_1based: int,
    time_stop_1based: int,
) -> np.ndarray:
    """
    Recreate the ImageJ prototype scoring for one (T, H, W) stack.

    Only frames within the 1-based inclusive time range [time_start_1based,
    time_stop_1based] are processed by default (matching duplicate range=1-10).
    Frames outside the selected range are set to zero.
    """
    t, h, w = img_tyx.shape
    input_dtype = img_tyx.dtype
    output = np.zeros((t, h, w), dtype=input_dtype)

    start_idx = max(0, int(time_start_1based) - 1)
    stop_idx_exclusive = min(t, int(time_stop_1based))
    if start_idx >= stop_idx_exclusive:
        return output

    fp_enlarge_1, fp_shrink, fp_enlarge_2 = _build_selection_footprints(
        enlarge_px_1=enlarge_px_1,
        shrink_px=shrink_px,
        enlarge_px_2=enlarge_px_2,
    )

    t_sel = stop_idx_exclusive - start_idx
    planes = img_tyx[start_idx:stop_idx_exclusive].astype(np.float32, copy=False)
    masks = None

    if mask_background or prezero_background_before_fill:
        masks = np.empty((t_sel, h, w), dtype=bool)
        for i in range(t_sel):
            masks[i] = _compute_foreground_mask_2d(
                plane=planes[i],
                downsample_factor=downsample_factor,
                footprint_enlarge_1=fp_enlarge_1,
                footprint_shrink=fp_shrink,
                footprint_enlarge_2=fp_enlarge_2,
            )

    filled_stack = np.empty_like(planes, dtype=np.float32)
    for i in range(t_sel):
        plane_for_fill = planes[i]
        if prezero_background_before_fill and masks is not None:
            # Optional speed/behavior tradeoff: fill only within detected foreground.
            plane_for_fill = plane_for_fill.copy()
            plane_for_fill[~masks[i]] = 0
        filled_stack[i] = fill_holes_2d(plane_for_fill)

    score_stack = filled_stack - planes
    np.maximum(score_stack, 0, out=score_stack)

    if mask_background and masks is not None:
        score_stack[~masks] = 0

    if median_sigma > 1:
        # Vectorized temporal batch median without smoothing across time.
        score_stack = median_filter(
            score_stack,
            size=(1, int(median_sigma), int(median_sigma)),
        )

    output[start_idx:stop_idx_exclusive] = _cast_to_dtype(score_stack, input_dtype)

    return output


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
    median_sigma: int = 1,
    roi_padding: int | None = None,
    min_region_area: int = 20,
    enlarge_px_1: int = 5,
    shrink_px: int = 10,
    enlarge_px_2: int = 5,
    mask_background: bool = True,
    prezero_background_before_fill: bool = False,
    time_start_1based: int = 1,
    time_stop_1based: int = 999999,
) -> bool:
    """Load one TCZYX image, compute ImageJ-style hole score maps, and save."""
    _ = variance_sigma
    _ = roi_padding
    _ = min_region_area

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
        f"  median_sigma={median_sigma}  range={time_start_1based}-{time_stop_1based}"
        f"  selection:+{enlarge_px_1}/-{shrink_px}/+{enlarge_px_2}"
        f"  mask_background={mask_background}"
        f"  prezero_before_fill={prezero_background_before_fill}"
    )

    if channels is None or len(channels_to_process) == c:
        output_data = np.zeros_like(img.data)
    else:
        output_data = img.data.copy()

    for ci in channels_to_process:
        for zi in range(z):
            tyx = img.data[:, ci, zi, :, :]
            scored = _score_tyx_stack_imagej_style(
                tyx,
                downsample_factor=downsample_factor,
                median_sigma=median_sigma,
                enlarge_px_1=enlarge_px_1,
                shrink_px=shrink_px,
                enlarge_px_2=enlarge_px_2,
                mask_background=mask_background,
                prezero_background_before_fill=prezero_background_before_fill,
                time_start_1based=time_start_1based,
                time_stop_1based=time_stop_1based,
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
            "Compute ImageJ-style hole score maps (fill holes -> subtract -> "
            "mask background via Otsu + enlarge/shrink sequence)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Hole score (ImageJ-style defaults)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_3.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'

- name: Hole score channels 0 and 2
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_3.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '0 2'

- name: Save as NumPy arrays, sequential processing
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/enhance_holes_3.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --output-format: npy
  - --no-parallel

Description:
  Recreates the ImageJ prototype behavior with defaults matching the macro:
    - Duplicate/process range 1-10 (1-based inclusive)
    - Otsu threshold in dark-background mode (foreground = > threshold)
    - Selection smoothing sequence: +5, -10, +5 pixels
    - Background in hole score map is set to zero

Notes:
  - CLI arguments are kept as similar as possible to enhance_holes_2.py.
  - --variance-sigma, --roi-padding and --min-region-area are accepted for
    CLI compatibility but are not used in this ImageJ-style implementation.
  - --downsample-factor only affects foreground mask computation.
  - --median-sigma defaults to 1 (disabled) to match the prototype.
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
            'Integer max-pool downsample factor for foreground mask detection '
            '(default: 1). Use "none" to disable.'
        ),
    )
    parser.add_argument(
        "--variance-sigma",
        type=float,
        default=4.0,
        help="Compatibility argument (unused in ImageJ-style logic)",
    )
    parser.add_argument(
        "--median-sigma",
        type=int,
        default=1,
        help="Kernel size for median filter on score map. Set to 1 to disable (default: 1)",
    )
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=None,
        help="Compatibility argument (unused in ImageJ-style logic)",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=20,
        help="Compatibility argument (unused in ImageJ-style logic)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Process files sequentially instead of in parallel",
    )

    parser.add_argument(
        "--enlarge-px-1",
        type=int,
        default=5,
        help="First selection enlarge radius in pixels (default: 5)",
    )
    parser.add_argument(
        "--shrink-px",
        type=int,
        default=10,
        help="Selection shrink radius in pixels (default: 10)",
    )
    parser.add_argument(
        "--enlarge-px-2",
        type=int,
        default=5,
        help="Second selection enlarge radius in pixels (default: 5)",
    )
    parser.add_argument(
        "--skip-background-mask",
        action="store_true",
        help=(
            "Skip background masking entirely for speed. "
            "This changes output behavior versus the ImageJ prototype."
        ),
    )
    parser.add_argument(
        "--prezero-background-before-fill",
        action="store_true",
        help=(
            "Set detected background to zero before hole filling. "
            "Can speed up and reduce background artifacts, but may alter scores near edges."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.enlarge_px_1 < 0 or args.shrink_px < 0 or args.enlarge_px_2 < 0:
        raise ValueError("--enlarge-px-1, --shrink-px and --enlarge-px-2 must be >= 0")

    if args.variance_sigma != 4.0 or args.roi_padding is not None or args.min_region_area != 20:
        logging.info(
            "Compatibility args provided (--variance-sigma/--roi-padding/--min-region-area); "
            "they are currently ignored in ImageJ-style logic."
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
                enlarge_px_1=args.enlarge_px_1,
                shrink_px=args.shrink_px,
                enlarge_px_2=args.enlarge_px_2,
                mask_background=not args.skip_background_mask,
                prezero_background_before_fill=args.prezero_background_before_fill,
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
        n_jobs = min(4, os.cpu_count() or 1)
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