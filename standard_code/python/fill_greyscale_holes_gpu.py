"""
Fill Holes in Greyscale Images (GPU-accelerated)

Fills dark regions (holes) in greyscale images while preserving intensity information.
This variant adds optional GPU acceleration for morphological reconstruction.

Author: BIPHUB, University of Oslo
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


def _get_tqdm() -> Callable[..., Any]:
    """Return tqdm if installed, otherwise a minimal no-op fallback."""
    try:
        from tqdm import tqdm as tqdm_impl

        return tqdm_impl
    except ImportError:
        class _TqdmFallback:
            """Minimal tqdm-compatible fallback when tqdm is not installed."""

            def __init__(self, iterable=None, total=None, **kwargs):
                self.iterable = iterable
                self.total = total

            def __iter__(self):
                if self.iterable is None:
                    return iter(())
                return iter(self.iterable)

            def update(self, n=1):
                _ = n

            def close(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                _ = (exc_type, exc_val, exc_tb)
                return False

        def tqdm_fallback(iterable=None, total=None, **kwargs):
            return _TqdmFallback(iterable=iterable, total=total, **kwargs)

        return tqdm_fallback


tqdm = _get_tqdm()

# Local imports
import bioimage_pipeline_utils as rp

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from cucim.skimage.morphology import reconstruction as reconstruction_gpu
except ImportError:
    reconstruction_gpu = None


def _reconstruction_cpu(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lazy import skimage reconstruction for CPU path only."""
    from skimage.morphology import reconstruction as reconstruction_cpu

    return reconstruction_cpu(seed, mask, method="dilation")


def _gpu_runtime_available() -> bool:
    """Return True when CuPy is importable and at least one CUDA device exists."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _reconstruction_gpu_fallback(seed_cp: Any, mask_cp: Any) -> Any:
    """Geodesic reconstruction via iterative dilation on GPU when cuCIM is unavailable."""
    from cupyx.scipy.ndimage import grey_dilation

    marker = seed_cp.copy()
    footprint = cp.ones((3,) * marker.ndim, dtype=cp.uint8)
    max_iters = int(sum(marker.shape)) + 1

    for _ in range(max_iters):
        dilated = grey_dilation(marker, footprint=footprint)
        updated = cp.minimum(dilated, mask_cp)
        if cp.array_equal(updated, marker):
            return updated
        marker = updated

    return marker


def _reconstruction_gpu(seed_cp: Any, mask_cp: Any) -> Any:
    """Use cuCIM reconstruction when available, otherwise CuPy iterative fallback."""
    if reconstruction_gpu is not None:
        return reconstruction_gpu(seed_cp, mask_cp, method="dilation")
    return _reconstruction_gpu_fallback(seed_cp, mask_cp)


def _validate_gpu_requested(require_gpu: bool) -> bool:
    """Resolve whether GPU can be used. Raise when required but unavailable."""
    available = _gpu_runtime_available()
    if require_gpu and not available:
        raise RuntimeError(
            "GPU requested but unavailable. Need: cupy + cucim + CUDA device."
        )
    return available


def _cast_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast values to target dtype without normalization."""
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        rounded = np.rint(values)
        clipped = np.clip(rounded, info.min, info.max)
        return clipped.astype(target_dtype)
    return values.astype(target_dtype, copy=False)


def _positive_delta(filled: np.ndarray, original: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return positive-only delta map (filled - original) cast to original dtype."""
    delta = filled.astype(np.float64, copy=False) - original.astype(np.float64, copy=False)
    np.maximum(delta, 0, out=delta)
    return _cast_to_dtype(delta, dtype)


def _resolve_kernel_overlap(kernel_size: int, kernel_overlap: str) -> int:
    """Resolve kernel overlap from CLI value."""
    if kernel_size < 1:
        raise ValueError(f"--kernel-size must be >= 1, got {kernel_size}")

    if str(kernel_overlap).strip().lower() == "half":
        overlap = kernel_size // 2
    else:
        try:
            overlap = int(str(kernel_overlap).strip())
        except ValueError as e:
            raise ValueError(
                f"Invalid --kernel-overlap '{kernel_overlap}'. Use 'half' or integer >= 0"
            ) from e

    if overlap < 0:
        raise ValueError(f"--kernel-overlap must be >= 0, got {overlap}")
    if overlap >= kernel_size:
        raise ValueError(
            f"--kernel-overlap must be < --kernel-size ({kernel_size}), got {overlap}"
        )
    return overlap


def greyscale_fill_holes_2d_cpu(image: np.ndarray) -> np.ndarray:
    """CPU fill holes in 2D greyscale image using skimage reconstruction."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = _reconstruction_cpu(seed, inverted)
    filled = image_max - reconstructed
    return filled


def greyscale_fill_holes_3d_cpu(image: np.ndarray) -> np.ndarray:
    """CPU fill holes in 3D greyscale image using skimage reconstruction."""
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1, 1:-1] = inverted.min()

    reconstructed = _reconstruction_cpu(seed, inverted)
    filled = image_max - reconstructed
    return filled


def _greyscale_fill_holes_2d_gpu_cp(image_cp: Any) -> Any:
    """GPU fill holes for a 2D CuPy array."""
    if image_cp.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_cp.shape}")

    if image_cp.size == 0:
        return image_cp.copy()

    image_max = image_cp.max()
    image_min = image_cp.min()
    if image_max == image_min:
        return image_cp.copy()

    inverted = image_max - image_cp

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = _reconstruction_gpu(seed, inverted)
    filled = image_max - reconstructed
    return filled


def _greyscale_fill_holes_3d_gpu_cp(image_cp: Any) -> Any:
    """GPU fill holes for a 3D CuPy array."""
    if image_cp.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image_cp.shape}")

    if image_cp.size == 0:
        return image_cp.copy()

    image_max = image_cp.max()
    image_min = image_cp.min()
    if image_max == image_min:
        return image_cp.copy()

    inverted = image_max - image_cp

    seed = inverted.copy()
    seed[1:-1, 1:-1, 1:-1] = inverted.min()

    reconstructed = _reconstruction_gpu(seed, inverted)
    filled = image_max - reconstructed
    return filled


def greyscale_fill_holes_2d_gpu(image: np.ndarray) -> np.ndarray:
    """GPU fill holes in 2D greyscale image."""
    if cp is None:
        raise RuntimeError("CuPy is unavailable")

    image_cp = cp.asarray(image)
    filled_cp = _greyscale_fill_holes_2d_gpu_cp(image_cp)
    return cp.asnumpy(filled_cp)


def greyscale_fill_holes_3d_gpu(image: np.ndarray) -> np.ndarray:
    """GPU fill holes in 3D greyscale image."""
    if cp is None:
        raise RuntimeError("CuPy is unavailable")

    image_cp = cp.asarray(image)
    filled_cp = _greyscale_fill_holes_3d_gpu_cp(image_cp)
    return cp.asnumpy(filled_cp)


def greyscale_fill_holes_kernel_2d_cpu(
    image: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
) -> np.ndarray:
    """
    Fill holes in 2D greyscale image using sliding local windows on CPU.

    Each local window is filled independently, then merged by pixelwise max.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_overlap < 0 or kernel_overlap >= kernel_size:
        raise ValueError(
            f"kernel_overlap must be >= 0 and < kernel_size ({kernel_size}), got {kernel_overlap}"
        )

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    h, w = image.shape
    stride = kernel_size - kernel_overlap
    if stride < 1:
        raise ValueError("Invalid stride: kernel_size - kernel_overlap must be >= 1")

    y_last = max(0, h - kernel_size)
    x_last = max(0, w - kernel_size)

    y_starts = list(range(0, y_last + 1, stride))
    x_starts = list(range(0, x_last + 1, stride))
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    if y_starts[-1] != y_last:
        y_starts.append(y_last)
    if x_starts[-1] != x_last:
        x_starts.append(x_last)

    merged = image.astype(np.float64, copy=True)

    for y0 in y_starts:
        y1 = min(y0 + kernel_size, h)
        for x0 in x_starts:
            x1 = min(x0 + kernel_size, w)
            tile = image[y0:y1, x0:x1]
            filled_tile = greyscale_fill_holes_2d_cpu(tile)
            np.maximum(merged[y0:y1, x0:x1], filled_tile, out=merged[y0:y1, x0:x1])

    return _cast_to_dtype(merged, image.dtype)


def greyscale_fill_holes_kernel_2d_gpu(
    image: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
) -> np.ndarray:
    """
    Fill holes in 2D greyscale image using sliding local windows on GPU.

    Each local window is filled independently, then merged by pixelwise max.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_overlap < 0 or kernel_overlap >= kernel_size:
        raise ValueError(
            f"kernel_overlap must be >= 0 and < kernel_size ({kernel_size}), got {kernel_overlap}"
        )

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    if cp is None:
        raise RuntimeError("CuPy is unavailable")

    h, w = image.shape
    stride = kernel_size - kernel_overlap
    if stride < 1:
        raise ValueError("Invalid stride: kernel_size - kernel_overlap must be >= 1")

    y_last = max(0, h - kernel_size)
    x_last = max(0, w - kernel_size)

    y_starts = list(range(0, y_last + 1, stride))
    x_starts = list(range(0, x_last + 1, stride))
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    if y_starts[-1] != y_last:
        y_starts.append(y_last)
    if x_starts[-1] != x_last:
        x_starts.append(x_last)

    image_cp = cp.asarray(image)
    merged_cp = image_cp.astype(cp.float64, copy=True)

    for y0 in y_starts:
        y1 = min(y0 + kernel_size, h)
        for x0 in x_starts:
            x1 = min(x0 + kernel_size, w)
            tile_cp = image_cp[y0:y1, x0:x1]
            filled_tile_cp = _greyscale_fill_holes_2d_gpu_cp(tile_cp)
            cp.maximum(merged_cp[y0:y1, x0:x1], filled_tile_cp, out=merged_cp[y0:y1, x0:x1])

    merged = cp.asnumpy(merged_cp)
    return _cast_to_dtype(merged, image.dtype)


def process_single_image(
    input_path: str,
    output_path: str,
    channels: Optional[list[int]] = None,
    mode_3d: bool = False,
    show_progress: bool = True,
    kernel_size: Optional[int] = None,
    kernel_overlap: str = "half",
    return_delta: bool = False,
    use_gpu: bool = False,
    gpu_device: int = 0,
) -> bool:
    """Process a single image file: load, fill holes, save."""
    try:
        logging.info(f"Loading image: {Path(input_path).name}")
        img = rp.load_tczyx_image(input_path)
    except Exception as e:
        logging.error(f"Failed to load image {Path(input_path).name}: {e}")
        logging.debug(f"File path: {input_path}", exc_info=True)
        return False

    T, C, Z, Y, X = img.shape
    logging.info(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

    if channels is None:
        channels_to_process = list(range(C))
    else:
        channels_to_process = [c for c in channels if 0 <= c < C]
        if not channels_to_process:
            raise ValueError(f"No valid channels found. Image has {C} channels, requested {channels}")

    logging.info(f"  Processing channels: {channels_to_process}")
    logging.info(f"  Mode: {'3D (Z-stack as volume)' if mode_3d else '2D (slice-by-slice)'}")
    logging.info(f"  Backend: {'GPU' if use_gpu else 'CPU'}")

    if kernel_size is not None:
        overlap_px = _resolve_kernel_overlap(kernel_size, kernel_overlap)
        stride_px = kernel_size - overlap_px
        logging.info(
            f"  Kernel mode: size={kernel_size}, overlap={overlap_px}, stride={stride_px}"
        )
        if mode_3d and Z > 1:
            logging.info("  Kernel mode runs per 2D Z-slice (XY local windows)")
    else:
        overlap_px = 0

    if return_delta:
        logging.info("  Output mode: positive delta map (filled - input)")

    output_data = img.data.copy()
    total_operations = T * len(channels_to_process)

    if use_gpu:
        if cp is None:
            raise RuntimeError("GPU backend selected but CuPy is unavailable")
        cp.cuda.Device(gpu_device).use()

    with tqdm(total=total_operations, desc="Filling holes", disable=not show_progress) as pbar:
        for t in range(T):
            for c in channels_to_process:
                if kernel_size is None and mode_3d and Z > 1:
                    volume = img.data[t, c, :, :, :]
                    if use_gpu:
                        filled_volume = greyscale_fill_holes_3d_gpu(volume)
                    else:
                        filled_volume = greyscale_fill_holes_3d_cpu(volume)

                    if return_delta:
                        output_data[t, c, :, :, :] = _positive_delta(
                            filled_volume,
                            volume,
                            img.data.dtype,
                        )
                    else:
                        output_data[t, c, :, :, :] = filled_volume
                else:
                    for z in range(Z):
                        slice_2d = img.data[t, c, z, :, :]

                        if kernel_size is None:
                            if use_gpu:
                                filled_slice = greyscale_fill_holes_2d_gpu(slice_2d)
                            else:
                                filled_slice = greyscale_fill_holes_2d_cpu(slice_2d)
                        else:
                            if use_gpu:
                                filled_slice = greyscale_fill_holes_kernel_2d_gpu(
                                    slice_2d,
                                    kernel_size=kernel_size,
                                    kernel_overlap=overlap_px,
                                )
                            else:
                                filled_slice = greyscale_fill_holes_kernel_2d_cpu(
                                    slice_2d,
                                    kernel_size=kernel_size,
                                    kernel_overlap=overlap_px,
                                )

                        if return_delta:
                            output_data[t, c, z, :, :] = _positive_delta(
                                filled_slice,
                                slice_2d,
                                img.data.dtype,
                            )
                        else:
                            output_data[t, c, z, :, :] = filled_slice

                pbar.update(1)

    logging.info(f"Saving filled image to: {Path(output_path).name}")
    rp.save_tczyx_image(output_data, output_path)
    logging.info("  Done!")
    return True


def _parse_channels(channel_str: str) -> list[int] | None:
    """Parse channels from string format like '0 2', '0,2', or None."""
    if channel_str is None:
        return None
    parts = str(channel_str).replace(",", " ").split()
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse channels: {e}")


@contextmanager
def _tqdm_joblib(tqdm_object):
    """Patch joblib to report progress to tqdm on task completion."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill holes in greyscale images while preserving intensity information (with optional GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fill greyscale holes on GPU (kernel mode)
    environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes_gpu.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_gpu'
  - --kernel-size: 40
  - --kernel-overlap: half
  - --use-gpu

- name: Fill greyscale holes on CPU (comparison)
    environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes_gpu.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_cpu'
  - --kernel-size: 40
  - --kernel-overlap: half

- name: Require GPU and fail if unavailable
    environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes_gpu.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_gpu'
  - --use-gpu
  - --require-gpu

Notes:
  - Kernel mode behavior matches CPU script: overlapping windows merged by pixelwise max.
  - GPU path needs cupy + cucim and a CUDA-capable GPU.
  - For GPU runs, keep file-level workers low (default 1) to avoid GPU oversubscription.
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Glob pattern for input images (e.g., 'data/**/*.tif')",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for filled images",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_filled",
        help="Suffix to add to output filenames (default: '_filled')",
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=None,
        help="Channel indices to process (0-based). Space or comma separated.",
    )
    parser.add_argument(
        "--mode-3d",
        action="store_true",
        help="Process Z-stacks as 3D volumes instead of individual 2D slices.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Optional local XY kernel size (pixels). If set, holes are filled per kernel window.",
    )
    parser.add_argument(
        "--kernel-overlap",
        type=str,
        default="half",
        help=(
            "Kernel overlap in pixels between adjacent windows (not stride). "
            "Stride = kernel_size - overlap. Use 'half' (default) or integer."
        ),
    )
    parser.add_argument(
        "--return-delta",
        action="store_true",
        help="Return positive delta max(filled - input, 0) instead of filled intensity image.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration (CuPy + cuCIM) when available.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if GPU backend is not available.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device index to use (default: 0).",
    )
    parser.add_argument(
        "--gpu-max-workers",
        type=int,
        default=1,
        help="Max file-level workers when GPU is enabled (default: 1).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Do not use parallel file processing.",
    )

    args = parser.parse_args()

    if args.kernel_size is not None and args.kernel_size < 1:
        raise ValueError(f"--kernel-size must be >= 1, got {args.kernel_size}")
    if args.kernel_size is None and str(args.kernel_overlap).strip().lower() != "half":
        logging.warning("--kernel-overlap is ignored unless --kernel-size is set")
    if args.kernel_size is not None:
        _ = _resolve_kernel_overlap(args.kernel_size, args.kernel_overlap)

    if args.gpu_max_workers < 1:
        raise ValueError(f"--gpu-max-workers must be >= 1, got {args.gpu_max_workers}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    resolved_use_gpu = False
    if args.use_gpu or args.require_gpu:
        resolved_use_gpu = _validate_gpu_requested(require_gpu=args.require_gpu)
        if resolved_use_gpu:
            logging.info("GPU backend is available and enabled")
            logging.info(f"Using CUDA device index: {args.gpu_device}")
        else:
            logging.warning("GPU backend unavailable; falling back to CPU")

    logging.info(f"Searching for files: {args.input_search_pattern}")
    input_files = rp.get_files_to_process2(args.input_search_pattern, True)

    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")

    logging.info(f"Found {len(input_files)} files to process")

    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output folder: {args.output_folder}")

    def process_file_wrapper(input_path: str) -> bool:
        input_name = Path(input_path).stem
        output_name = f"{input_name}{args.suffix}.tif"
        output_path = os.path.join(args.output_folder, output_name)

        try:
            return process_single_image(
                input_path=input_path,
                output_path=output_path,
                channels=args.channels,
                mode_3d=args.mode_3d,
                show_progress=args.no_parallel,
                kernel_size=args.kernel_size,
                kernel_overlap=args.kernel_overlap,
                return_delta=args.return_delta,
                use_gpu=resolved_use_gpu,
                gpu_device=args.gpu_device,
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            import traceback

            logging.error(traceback.format_exc())
            return False

    if not args.no_parallel:
        from joblib import Parallel, delayed

        if resolved_use_gpu:
            n_jobs = min(len(input_files), args.gpu_max_workers)
            logging.info(
                f"Processing {len(input_files)} files with GPU-aware parallelism (n_jobs={n_jobs}, prefer='threads')..."
            )
            with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
                with _tqdm_joblib(pbar):
                    results = list(
                        Parallel(n_jobs=n_jobs, prefer="threads")(
                            delayed(process_file_wrapper)(file) for file in input_files
                        )
                    )
        else:
            logging.info(f"Processing {len(input_files)} files in CPU parallel mode...")
            with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
                with _tqdm_joblib(pbar):
                    results = list(
                        Parallel(n_jobs=-1)(
                            delayed(process_file_wrapper)(file) for file in input_files
                        )
                    )

        successful = sum(1 for r in results if r)
        failed = len(results) - successful
    else:
        logging.info(f"Processing {len(input_files)} files sequentially...")
        successful = 0
        failed = 0
        for i, input_path in enumerate(input_files, 1):
            logging.info(f"\n{'=' * 70}")
            logging.info(f"Processing file {i}/{len(input_files)}")
            if process_file_wrapper(input_path):
                successful += 1
            else:
                failed += 1

    logging.info(f"\n{'=' * 70}")
    logging.info("Processing complete!")
    logging.info(f"Processed {len(input_files)} files: {successful} succeeded, {failed} failed")
    logging.info(f"Output saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
