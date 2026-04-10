"""
Fill Holes in Greyscale Images

Fills dark regions (holes) in greyscale images while preserving intensity information.
Perfect for filling nucleoli inside nuclei or other dark internal structures.

Uses morphological reconstruction (flood-fill from borders) - equivalent to MATLAB's imfill.

Author: BIPHUB , University of Oslo
Written by: Øyvind Ødegård Fougner
License: MIT
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import sysconfig
from pathlib import Path
from typing import Optional
import logging
import h5py
import json
from contextlib import ExitStack, contextmanager
import tifffile
from skimage.morphology import reconstruction
from tqdm import tqdm

# Local imports
import bioimage_pipeline_utils as rp


def _prepend_env_path(name: str, values: list[str]) -> None:
    existing = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    merged: list[str] = []
    for value in values + existing:
        if value and value not in merged:
            merged.append(value)
    if merged:
        os.environ[name] = os.pathsep.join(merged)


def _configure_gpu_runtime_environment() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = purelib / "nvidia"
    if not nvidia_root.exists():
        return

    lib_dirs: list[str] = []
    bin_dirs: list[str] = []
    cuda_path: Optional[str] = None

    for child in sorted(nvidia_root.iterdir()):
        if not child.is_dir():
            continue
        lib_dir = child / "lib"
        bin_dir = child / "bin"
        if lib_dir.is_dir():
            lib_dirs.append(str(lib_dir))
        if bin_dir.is_dir():
            bin_dirs.append(str(bin_dir))
        if child.name == "cuda_runtime":
            cuda_path = str(child)

    if cuda_path is not None:
        os.environ.setdefault("CUDA_PATH", cuda_path)
        os.environ.setdefault("CUDA_HOME", cuda_path)

    _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)
    _prepend_env_path("PATH", bin_dirs + lib_dirs)


_configure_gpu_runtime_environment()

try:
    import cupy as cp
    from cucim.skimage.morphology import reconstruction as gpu_reconstruction
    _GPU_IMPORT_ERROR: Optional[Exception] = None
except Exception as gpu_import_error:
    cp = None
    gpu_reconstruction = None
    _GPU_IMPORT_ERROR = gpu_import_error

_GPU_DISABLED_REASON: Optional[str] = (
    None if _GPU_IMPORT_ERROR is None else f"GPU imports unavailable: {_GPU_IMPORT_ERROR}"
)


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logging.warning("Invalid float for %s=%r, using default %s", name, raw_value, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logging.warning("Invalid integer for %s=%r, using default %s", name, raw_value, default)
        return default


_GPU_KERNEL_TARGET_FREE_FRACTION = min(
    max(_get_env_float("RP_GPU_KERNEL_TARGET_FREE_FRACTION", 0.6), 0.05),
    0.9,
)
_GPU_KERNEL_MAX_BATCH_WINDOWS = max(_get_env_int("RP_GPU_KERNEL_MAX_BATCH_WINDOWS", 8192), 1)
_GPU_KERNEL_WINDOW_BYTES_FACTOR = max(_get_env_int("RP_GPU_KERNEL_WINDOW_BYTES_FACTOR", 8), 1)
_GPU_KERNEL_FRAME_GROUP_SIZE = max(_get_env_int("RP_GPU_KERNEL_FRAME_GROUP_SIZE", 1), 1)


def _estimate_nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def _kernel_window_starts(length: int, kernel_size: int, stride: int) -> list[int]:
    last_start = max(0, length - kernel_size)
    starts = list(range(0, last_start + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _estimate_gpu_kernel_batch_size(
    tile_shape: tuple[int, int],
    dtype: np.dtype,
    window_count: int,
) -> tuple[int, int, int]:
    if cp is None:
        return 1, 0, 0

    tile_bytes = _estimate_nbytes(tile_shape, dtype)
    bytes_per_window = max(tile_bytes * _GPU_KERNEL_WINDOW_BYTES_FACTOR, 1)
    try:
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        target_bytes = max(int(free_bytes * _GPU_KERNEL_TARGET_FREE_FRACTION), bytes_per_window)
        batch_size = target_bytes // bytes_per_window
    except Exception:
        free_bytes = 0
        batch_size = 256
        target_bytes = bytes_per_window * batch_size

    resolved_batch_size = max(1, min(window_count, int(batch_size), _GPU_KERNEL_MAX_BATCH_WINDOWS))
    return resolved_batch_size, free_bytes, bytes_per_window


def _greyscale_fill_holes_gpu_batch(
    image_gpu,
    windows: list[tuple[int, int, int, int]],
    tile_shape: tuple[int, int],
):
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if not windows:
        return cp.empty((0,) + tile_shape, dtype=image_gpu.dtype)

    tile_height, tile_width = tile_shape
    tile_count = len(windows)
    gap = 1
    tiles_per_row = max(1, int(np.ceil(np.sqrt(tile_count))))
    tile_rows = int(np.ceil(tile_count / tiles_per_row))
    packed_height = tile_rows * tile_height + (tile_rows + 1) * gap
    packed_width = tiles_per_row * tile_width + (tiles_per_row + 1) * gap

    filled_tiles = cp.empty((tile_count, tile_height, tile_width), dtype=image_gpu.dtype)
    tile_maxima = cp.empty((tile_count,), dtype=image_gpu.dtype)

    packed_mask = cp.empty((packed_height, packed_width), dtype=image_gpu.dtype)
    for tile_index, (y0, x0, *_rest) in enumerate(windows):
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        tile_gpu = image_gpu[y0:y1, x0:x1]
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_mask[packed_y0:packed_y0 + tile_height, packed_x0:packed_x0 + tile_width] = tile_gpu
        tile_maxima[tile_index] = tile_gpu.max()

    separator_value = image_gpu.min()
    packed_mask.fill(separator_value)
    for tile_index, (y0, x0, *_rest) in enumerate(windows):
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        tile_gpu = image_gpu[y0:y1, x0:x1]
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_mask[packed_y0:packed_y0 + tile_height, packed_x0:packed_x0 + tile_width] = tile_gpu

    image_max = packed_mask.max()
    packed_inverted = image_max - packed_mask
    packed_seed = packed_inverted.copy()

    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_seed[
            packed_y0 + 1:packed_y0 + tile_height - 1,
            packed_x0 + 1:packed_x0 + tile_width - 1,
        ] = image_max - tile_maxima[tile_index]

    packed_filled = image_max - gpu_reconstruction(packed_seed, packed_inverted, method='dilation')
    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        filled_tiles[tile_index] = packed_filled[
            packed_y0:packed_y0 + tile_height,
            packed_x0:packed_x0 + tile_width,
        ]

    return filled_tiles


def _greyscale_fill_holes_gpu_batch_across_frames(
    stack_gpu,
    windows: list[tuple[int, int, int]],
    tile_shape: tuple[int, int],
):
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if not windows:
        return cp.empty((0,) + tile_shape, dtype=stack_gpu.dtype)

    tile_height, tile_width = tile_shape
    tile_count = len(windows)
    gap = 1
    tiles_per_row = max(1, int(np.ceil(np.sqrt(tile_count))))
    tile_rows = int(np.ceil(tile_count / tiles_per_row))
    packed_height = tile_rows * tile_height + (tile_rows + 1) * gap
    packed_width = tiles_per_row * tile_width + (tiles_per_row + 1) * gap

    filled_tiles = cp.empty((tile_count, tile_height, tile_width), dtype=stack_gpu.dtype)
    tile_maxima = cp.empty((tile_count,), dtype=stack_gpu.dtype)

    packed_mask = cp.empty((packed_height, packed_width), dtype=stack_gpu.dtype)
    separator_value = stack_gpu.min()
    packed_mask.fill(separator_value)

    for tile_index, (frame_index, y0, x0) in enumerate(windows):
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        tile_gpu = stack_gpu[frame_index, y0:y1, x0:x1]
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_mask[packed_y0:packed_y0 + tile_height, packed_x0:packed_x0 + tile_width] = tile_gpu
        tile_maxima[tile_index] = tile_gpu.max()

    image_max = packed_mask.max()
    packed_inverted = image_max - packed_mask
    packed_seed = packed_inverted.copy()

    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_seed[
            packed_y0 + 1:packed_y0 + tile_height - 1,
            packed_x0 + 1:packed_x0 + tile_width - 1,
        ] = image_max - tile_maxima[tile_index]

    packed_filled = image_max - gpu_reconstruction(packed_seed, packed_inverted, method='dilation')
    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        filled_tiles[tile_index] = packed_filled[
            packed_y0:packed_y0 + tile_height,
            packed_x0:packed_x0 + tile_width,
        ]

    return filled_tiles


def _merge_gpu_tiles(
    merged_gpu,
    filled_tiles_gpu,
    batch_windows: list[tuple[int, int, int, int]],
    tile_shape: tuple[int, int],
) -> None:
    if cp is None:
        raise ImportError("cupy GPU support is not available")

    tile_height, tile_width = tile_shape
    filled_tiles_gpu = filled_tiles_gpu.astype(merged_gpu.dtype, copy=False)
    for tile_index, (y0, x0, _y_index, _x_index) in enumerate(batch_windows):
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        cp.maximum(
            merged_gpu[y0:y1, x0:x1],
            filled_tiles_gpu[tile_index],
            out=merged_gpu[y0:y1, x0:x1],
        )


def _build_ome_metadata(physical_pixel_sizes) -> dict[str, object]:
    metadata: dict[str, object] = {"axes": "TCZYX"}
    if physical_pixel_sizes is None:
        return metadata

    x_size = getattr(physical_pixel_sizes, "X", None)
    y_size = getattr(physical_pixel_sizes, "Y", None)
    z_size = getattr(physical_pixel_sizes, "Z", None)

    if x_size is not None:
        metadata["PhysicalSizeX"] = float(x_size)
    if y_size is not None:
        metadata["PhysicalSizeY"] = float(y_size)
    if z_size is not None:
        metadata["PhysicalSizeZ"] = float(z_size)

    return metadata


def _load_czyx_timepoint(img, t: int) -> np.ndarray:
    if hasattr(img, "dask_data") and img.dask_data is not None:
        return np.asarray(img.dask_data[t].compute())
    return np.asarray(img.get_image_data("CZYX", T=t))


def _disable_gpu_hole_fill(reason: str) -> None:
    global _GPU_DISABLED_REASON
    if _GPU_DISABLED_REASON is None:
        logging.warning("GPU hole filling unavailable, falling back to CPU: %s", reason)
    _GPU_DISABLED_REASON = reason


def _describe_hole_fill_backend(use_gpu: bool) -> str:
    if not use_gpu:
        return "CPU (forced by --no-gpu)"
    if _GPU_DISABLED_REASON is None and cp is not None and gpu_reconstruction is not None:
        return "GPU (cuCIM reconstruction)"
    if _GPU_DISABLED_REASON is not None:
        return f"CPU fallback ({_GPU_DISABLED_REASON})"
    return "CPU fallback (GPU libraries unavailable)"


def _greyscale_fill_holes_2d_cpu(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = reconstruction(seed, inverted, method='dilation')
    filled = image_max - reconstructed
    return filled


def _greyscale_fill_holes_2d_gpu_array(image_gpu):
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if image_gpu.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_gpu.shape}")

    if image_gpu.size == 0:
        return image_gpu.copy()
    if bool(cp.all(image_gpu == image_gpu.ravel()[0]).item()):
        return image_gpu.copy()

    image_max = image_gpu.max()
    inverted = image_max - image_gpu

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = gpu_reconstruction(seed, inverted, method='dilation')
    return image_max - reconstructed


def _greyscale_fill_holes_2d_gpu(image: np.ndarray) -> np.ndarray:
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    image_gpu = cp.asarray(image)
    filled = _greyscale_fill_holes_2d_gpu_array(image_gpu)
    return cp.asnumpy(filled)


def _greyscale_fill_holes_kernel_2d_gpu(
    image: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
) -> np.ndarray:
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    h, w = image.shape
    stride = kernel_size - kernel_overlap
    if stride < 1:
        raise ValueError("Invalid stride: kernel_size - kernel_overlap must be >= 1")

    y_starts = _kernel_window_starts(h, kernel_size, stride)
    x_starts = _kernel_window_starts(w, kernel_size, stride)
    tile_shape = (min(kernel_size, h), min(kernel_size, w))
    windows = [
        (y0, x0, y_index, x_index)
        for y_index, y0 in enumerate(y_starts)
        for x_index, x0 in enumerate(x_starts)
    ]
    image_gpu = cp.asarray(image)
    merged_gpu = image_gpu.astype(cp.float64, copy=True)

    batch_size, free_bytes, bytes_per_window = _estimate_gpu_kernel_batch_size(
        tile_shape,
        image_gpu.dtype,
        len(windows),
    )
    logging.info(
        "GPU kernel batching: %s windows, batch size %s, tile shape %sx%s, free VRAM %.2f GiB, est %.2f MiB/window group",
        len(windows),
        batch_size,
        tile_shape[0],
        tile_shape[1],
        free_bytes / (1024 ** 3),
        (batch_size * bytes_per_window) / (1024 ** 2),
    )

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]
        filled_batch_gpu = _greyscale_fill_holes_gpu_batch(image_gpu, batch_windows, tile_shape)
        _merge_gpu_tiles(
            merged_gpu,
            filled_batch_gpu,
            batch_windows,
            tile_shape,
        )

    merged = cp.asnumpy(merged_gpu)
    return _cast_to_dtype(merged, image.dtype)


def _greyscale_fill_holes_kernel_multiframe_2d_gpu(
    stack: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
    frame_group_size: int,
) -> np.ndarray:
    if cp is None or gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape (T, Y, X), got {stack.shape}")
    if stack.size == 0:
        return stack.copy()

    frame_count, height, width = stack.shape
    stride = kernel_size - kernel_overlap
    if stride < 1:
        raise ValueError("Invalid stride: kernel_size - kernel_overlap must be >= 1")

    effective_frame_group_size = max(1, min(frame_group_size, frame_count))
    y_starts = _kernel_window_starts(height, kernel_size, stride)
    x_starts = _kernel_window_starts(width, kernel_size, stride)
    tile_shape = (min(kernel_size, height), min(kernel_size, width))

    stack_gpu = cp.asarray(stack)
    merged_gpu = stack_gpu.astype(cp.float64, copy=True)

    windows_per_frame = len(y_starts) * len(x_starts)
    window_batch_size, free_bytes, bytes_per_window = _estimate_gpu_kernel_batch_size(
        tile_shape,
        stack_gpu.dtype,
        windows_per_frame * effective_frame_group_size,
    )
    logging.info(
        "GPU kernel multiframe batching: %s frames, frame group size %s, windows/frame %s, tile shape %sx%s, free VRAM %.2f GiB, est %.2f MiB/window group",
        frame_count,
        effective_frame_group_size,
        windows_per_frame,
        tile_shape[0],
        tile_shape[1],
        free_bytes / (1024 ** 3),
        (window_batch_size * bytes_per_window) / (1024 ** 2),
    )

    positions_per_batch = max(1, window_batch_size // effective_frame_group_size)
    window_positions = [(y0, x0) for y0 in y_starts for x0 in x_starts]

    for frame_start in range(0, frame_count, effective_frame_group_size):
        frame_stop = min(frame_count, frame_start + effective_frame_group_size)
        group_frame_count = frame_stop - frame_start
        group_frame_slice = slice(frame_start, frame_stop)

        for position_start in range(0, len(window_positions), positions_per_batch):
            batch_positions = window_positions[position_start:position_start + positions_per_batch]
            batch_windows = [
                (frame_index, y0, x0)
                for y0, x0 in batch_positions
                for frame_index in range(frame_start, frame_stop)
            ]
            filled_tiles_gpu = _greyscale_fill_holes_gpu_batch_across_frames(
                stack_gpu,
                batch_windows,
                tile_shape,
            )
            filled_tiles_gpu = filled_tiles_gpu.astype(merged_gpu.dtype, copy=False)
            filled_tiles_gpu = filled_tiles_gpu.reshape(
                len(batch_positions),
                group_frame_count,
                tile_shape[0],
                tile_shape[1],
            )

            for position_index, (y0, x0) in enumerate(batch_positions):
                y1 = y0 + tile_shape[0]
                x1 = x0 + tile_shape[1]
                cp.maximum(
                    merged_gpu[group_frame_slice, y0:y1, x0:x1],
                    filled_tiles_gpu[position_index],
                    out=merged_gpu[group_frame_slice, y0:y1, x0:x1],
                )

    merged = cp.asnumpy(merged_gpu)
    return _cast_to_dtype(merged, stack.dtype)


def greyscale_fill_holes_2d(image: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Fill holes in 2D greyscale image using morphological reconstruction.
    
    This is the Python equivalent of MATLAB's imfill(image, 'holes').
    Works by flood-filling from image borders - holes (regions not connected
    to borders) remain unfilled and are then inverted back.
    
    Perfect for filling dark nucleoli inside bright nuclei.
    
    Parameters
    ----------
    image : np.ndarray
        2D greyscale image (YX) with holes to fill.
        Dark regions (holes) will be filled with surrounding intensities.
        
    Returns
    -------
    filled : np.ndarray
        Image with holes filled, preserving greyscale intensities.
        
    Examples
    --------
    >>> # Fill nucleoli in nucleus image
    >>> nucleus = load_image("nucleus.tif")
    >>> filled = greyscale_fill_holes_2d(nucleus)
    
    Notes
    -----
    The algorithm:
    1. Inverts the image (dark holes become bright peaks)
    2. Seeds from image borders
    3. Flood-fills inward (cannot reach holes)
    4. Inverts back (filled peaks become filled valleys)
    
    References
    ----------
    MATLAB's imfill: http://www.ece.northwestern.edu/CSEL/local-apps/matlabhelp/toolbox/images/morph13.html
    """
    if use_gpu and _GPU_DISABLED_REASON is None:
        try:
            return _greyscale_fill_holes_2d_gpu(image)
        except Exception as exc:
            _disable_gpu_hole_fill(str(exc))

    return _greyscale_fill_holes_2d_cpu(image)


# def greyscale_fill_holes_3d(image: np.ndarray) -> np.ndarray:
#     """
#     Fill holes in 3D greyscale image (Z-stack) using morphological reconstruction.
    
#     Extends the 2D hole filling to 3D volumes. Useful for filling nucleoli
#     in confocal Z-stacks of nuclei.
    
#     Parameters
#     ----------
#     image : np.ndarray
#         3D greyscale image (ZYX) with holes to fill.
#         Dark regions (holes) will be filled with surrounding intensities.
        
#     Returns
#     -------
#     filled : np.ndarray
#         Volume with holes filled, preserving greyscale intensities.
        
#     Examples
#     --------
#     >>> # Fill nucleoli in 3D nucleus Z-stack
#     >>> nucleus_stack = load_image("nucleus_zstack.tif")  # ZYX
#     >>> filled = greyscale_fill_holes_3d(nucleus_stack)
    
#     Notes
#     -----
#     Uses 3D morphological reconstruction to fill holes that are isolated
#     in all three dimensions.
#     """
#     if image.ndim != 3:
#         raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
#     # Handle edge case: empty or constant image
#     if image.size == 0 or np.all(image == image.flat[0]):
#         return image.copy()
    
#     # Invert the image (holes become peaks)
#     image_max = image.max()
#     inverted = image_max - image
    
#     # Create seed: keep border, clear interior volume
#     seed = inverted.copy()
#     seed[1:-1, 1:-1, 1:-1] = inverted.min()
    
#     # Morphological reconstruction in 3D
#     reconstructed = reconstruction(seed, inverted, method='dilation')
    
#     # Invert back to get filled result
#     filled = image_max - reconstructed
    
#     return filled


# def fill_holes_auto(image: np.ndarray) -> np.ndarray:
#     """
#     Automatically detect dimensionality and fill holes appropriately.
    
#     Parameters
#     ----------
#     image : np.ndarray
#         2D (YX) or 3D (ZYX) greyscale image with holes to fill.
        
#     Returns
#     -------
#     filled : np.ndarray
#         Image with holes filled.
        
#     Raises
#     ------
#     ValueError
#         If image is not 2D or 3D.
#     """
#     if image.ndim == 2:
#         return greyscale_fill_holes_2d(image)
#     elif image.ndim == 3:
#         return greyscale_fill_holes_3d(image)
#     else:
#         raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D with shape {image.shape}")


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


def greyscale_fill_holes_kernel_2d(
    image: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Fill holes in 2D greyscale image using sliding local windows.

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

    y_starts = _kernel_window_starts(h, kernel_size, stride)
    x_starts = _kernel_window_starts(w, kernel_size, stride)

    if use_gpu and _GPU_DISABLED_REASON is None:
        try:
            return _greyscale_fill_holes_kernel_2d_gpu(
                image,
                kernel_size=kernel_size,
                kernel_overlap=kernel_overlap,
            )
        except Exception as exc:
            _disable_gpu_hole_fill(str(exc))

    merged = image.astype(np.float64, copy=True)

    for y0 in y_starts:
        y1 = min(y0 + kernel_size, h)
        for x0 in x_starts:
            x1 = min(x0 + kernel_size, w)
            tile = image[y0:y1, x0:x1]
            filled_tile = greyscale_fill_holes_2d(tile, use_gpu=False)
            np.maximum(merged[y0:y1, x0:x1], filled_tile, out=merged[y0:y1, x0:x1])

    return _cast_to_dtype(merged, image.dtype)


def process_single_image(
    input_path: str,
    output_stem: str,
    output_formats: list[str],
    channels: Optional[list[int]] = None,
    mode_3d: bool = False,
    show_progress: bool = True,
    kernel_size: Optional[int] = None,
    kernel_overlap: str = "half",
    return_delta: bool = False,
    include_input: bool = False,
    use_gpu: bool = True,
) -> bool:
    """
    Process a single image file: load, fill holes, save.
    
    Parameters
    ----------
    input_path : str
        Path to input image file.
    output_stem : str
        Output path without file extension (e.g. ``output_folder/name_filled``).
        The correct extension is appended for each format in output_formats.
    output_formats : list[str]
        List of formats to save: ``"tif"``, ``"npy"``, and/or ``"ilastik-h5"``.
    channels : list[int], optional
        List of channel indices to process. If None, processes all channels.
    mode_3d : bool
        If True, treat each Z-stack as a 3D volume for hole filling.
        If False, process each Z-slice independently as 2D.
    show_progress : bool
        If True, show tqdm progress bar. If False, suppress progress bar.
    include_input : bool
        If True, output is paired per processed channel: channel 0 = original,
        channel 1 = filled/delta. Output shape becomes
        (T, 2 * len(channels_to_process), Z, Y, X).
        
    Returns
    -------
    bool
        True if processing succeeded, False otherwise.
        
    Notes
    -----
    For multi-channel, multi-timepoint, or Z-stack images:
    - Processes each timepoint independently
    - Can process each channel independently or all channels
    - Can process Z-slices as 2D or entire Z-stack as 3D
    """
    try:
        logging.info(f"Loading image: {Path(input_path).name}")
        img = rp.load_tczyx_image(input_path)
    except Exception as e:
        logging.error(f"Failed to load image {Path(input_path).name}: {e}")
        logging.debug(f"File path: {input_path}", exc_info=True)
        return False
    
    T, C, Z, Y, X = img.shape
    img_dtype = img.dask_data.dtype
    logging.info(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    # Determine which channels to process
    if channels is None:
        channels_to_process = list(range(C))
    else:
        channels_to_process = [c for c in channels if 0 <= c < C]
        if not channels_to_process:
            raise ValueError(f"No valid channels found. Image has {C} channels, requested {channels}")
    
    logging.info(f"  Processing channels: {channels_to_process}")
    logging.info(f"  Mode: {'3D (Z-stack as volume)' if mode_3d else '2D (slice-by-slice)'}")
    logging.info(f"  Hole-fill backend: {_describe_hole_fill_backend(use_gpu)}")
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

    gpu_kernel_frame_group_size = 1
    if kernel_size is not None and use_gpu and not mode_3d:
        gpu_kernel_frame_group_size = min(_GPU_KERNEL_FRAME_GROUP_SIZE, T)
        if gpu_kernel_frame_group_size > 1:
            logging.info("  GPU kernel frame grouping: %s timepoints per batch", gpu_kernel_frame_group_size)

    if return_delta:
        logging.info("  Output mode: positive delta map (filled - input)")

    # Determine output channel count
    n_out = len(channels_to_process) * 2 if include_input else C
    if include_input:
        logging.info(
            f"  Include input: {n_out} output channels "
            f"(original+filled pairs for each processed channel)"
        )

    # --- Memory-efficient streaming approach ---
    # ilastik-h5: pre-create the dataset and write one T-slice at a time (no full array needed).
    # tif/npy: accumulate in a pre-allocated array (one allocation instead of load + copy).
    needs_full_array = 'npy' in output_formats
    full_output: Optional[np.ndarray] = (
        np.empty((T, n_out, Z, Y, X), dtype=img_dtype) if needs_full_array else None
    )

    total_operations = T * len(channels_to_process)
    write_tif = 'tif' in output_formats
    tif_output_path = f"{output_stem}.tif"

    saved_h5_paths: list[str] = []
    with ExitStack() as exit_stack:
        open_h5_files: dict[str, tuple] = {}

        for fmt in output_formats:
            if fmt == 'ilastik-h5':
                output_path = f"{output_stem}.h5"
                logging.info(f"Preparing Ilastik HDF5 (streaming): {Path(output_path).name}")
                axis_configs = [
                    {'key': 't', 'typeFlags': 8, 'resolution': 0, 'description': ''},
                    {'key': 'z', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'y', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'x', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'c', 'typeFlags': 1, 'resolution': 0, 'description': ''},
                ]
                f = exit_stack.enter_context(h5py.File(output_path, 'w'))
                # Ilastik layout: TZYXC
                dset = f.create_dataset(
                    'data',
                    shape=(T, Z, Y, X, n_out),
                    dtype=img_dtype,
                    compression='gzip',
                    compression_opts=4,
                )
                dset.attrs['axistags'] = json.dumps({'axes': axis_configs})
                open_h5_files[output_path] = (f, dset)
                saved_h5_paths.append(output_path)

        def _processed_planes():
            with tqdm(total=total_operations, desc="Filling holes", disable=not show_progress) as pbar:
                timepoint_step = gpu_kernel_frame_group_size if gpu_kernel_frame_group_size > 1 else 1
                for t_start in range(0, T, timepoint_step):
                    t_stop = min(T, t_start + timepoint_step)
                    czyx_group = [_load_czyx_timepoint(img, t) for t in range(t_start, t_stop)]

                    if include_input:
                        out_group = [np.empty((n_out, Z, Y, X), dtype=img_dtype) for _ in czyx_group]
                    else:
                        out_group = [czyx.copy() for czyx in czyx_group]

                    for c_idx, c in enumerate(channels_to_process):
                        if kernel_size is None and mode_3d and Z > 1:
                            raise NotImplementedError("3D kernel mode is not implemented. Use 2D kernel or disable kernel mode.")

                        for z in range(Z):
                            used_grouped_gpu = False
                            if (
                                kernel_size is not None
                                and use_gpu
                                and gpu_kernel_frame_group_size > 1
                                and _GPU_DISABLED_REASON is None
                                and len(czyx_group) > 1
                            ):
                                try:
                                    slice_stack = np.stack([czyx[c, z] for czyx in czyx_group], axis=0)
                                    filled_stack = _greyscale_fill_holes_kernel_multiframe_2d_gpu(
                                        slice_stack,
                                        kernel_size=kernel_size,
                                        kernel_overlap=overlap_px,
                                        frame_group_size=gpu_kernel_frame_group_size,
                                    )
                                    used_grouped_gpu = True
                                except Exception as exc:
                                    logging.warning(
                                        "GPU kernel frame grouping failed, falling back to per-frame processing: %s",
                                        exc,
                                    )

                            for local_index, czyx in enumerate(czyx_group):
                                slice_2d = czyx[c, z]
                                if used_grouped_gpu:
                                    filled_slice = filled_stack[local_index]
                                else:
                                    filled_slice = (
                                        greyscale_fill_holes_kernel_2d(
                                            slice_2d,
                                            kernel_size=kernel_size,
                                            kernel_overlap=overlap_px,
                                            use_gpu=use_gpu,
                                        )
                                        if kernel_size is not None
                                        else greyscale_fill_holes_2d(slice_2d, use_gpu=use_gpu)
                                    )
                                result_2d = (
                                    _positive_delta(filled_slice, slice_2d, img_dtype)
                                    if return_delta
                                    else _cast_to_dtype(filled_slice, img_dtype)
                                )
                                if include_input:
                                    out_group[local_index][c_idx * 2, z] = slice_2d
                                    out_group[local_index][c_idx * 2 + 1, z] = result_2d
                                else:
                                    out_group[local_index][c, z] = result_2d

                        pbar.update(len(czyx_group))

                    for local_index, out_czyx in enumerate(out_group):
                        t = t_start + local_index
                        for _path, (_f, _dset) in open_h5_files.items():
                            _dset[t] = np.transpose(out_czyx, (1, 2, 3, 0))

                        if full_output is not None:
                            full_output[t] = out_czyx

                        if write_tif:
                            for c in range(n_out):
                                for z in range(Z):
                                    yield out_czyx[c, z]

        if write_tif:
            logging.info(f"Preparing OME-TIFF (streaming): {Path(tif_output_path).name}")
            tifffile.imwrite(
                tif_output_path,
                _processed_planes(),
                shape=(T, n_out, Z, Y, X),
                dtype=np.dtype(img_dtype),
                photometric='minisblack',
                compression='deflate',
                ome=True,
                metadata=_build_ome_metadata(img.physical_pixel_sizes),
                bigtiff=_estimate_nbytes((T, n_out, Z, Y, X), img_dtype) >= (4 * 1024**3 - 32 * 1024**2),
            )
        else:
            for _ in _processed_planes():
                pass

    for output_path in saved_h5_paths:
        logging.info(f"Saved Ilastik HDF5: {Path(output_path).name}")
    if write_tif:
        logging.info(f"Saved OME-TIFF: {Path(tif_output_path).name}")

    # Write formats that require the complete array
    if full_output is not None:
        for fmt in output_formats:
            if fmt == 'npy':
                output_path = f"{output_stem}.npy"
                logging.info(f"Saving NumPy array: {Path(output_path).name}")
                np.save(output_path, full_output)

    logging.info("  Done!")
    return True


def parse_output_formats(fmt_str) -> list[str]:
    """Parse output format(s) from string, handling both space-separated and single values."""
    if isinstance(fmt_str, list):
        return fmt_str
    formats = fmt_str.split()
    valid_formats = {"tif", "npy", "ilastik-h5"}
    for fmt in formats:
        if fmt not in valid_formats:
            raise argparse.ArgumentTypeError(f"Invalid format '{fmt}'. Choose from: tif, npy, ilastik-h5")
    return formats


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


def main():
    parser = argparse.ArgumentParser(
        description='Fill holes in greyscale images while preserving intensity information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fill nucleoli in nucleus images
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*nucleus*.tif'
  - --output-folder: '%YAML%/output_data'

# 3D mode is written but not tested yet - commented out to avoid confusion until verified
# - name: Fill holes in 3D Z-stacks (volumetric)
#   environment: uv@3.11:default
#   commands:
#   - python
#   - '%REPO%/standard_code/python/fill_greyscale_holes.py'
#   - --input-search-pattern: '%YAML%/input_data/**/*.tif'
#   - --output-folder: '%YAML%/output_data'
#   - --mode-3d

- name: Fill holes in channels 0 and 2
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '0 2'

- name: Fill holes with local sliding kernel
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/fill_greyscale_holes.py'
    - --input-search-pattern: '%YAML%/input_data/**/*.tif'
    - --output-folder: '%YAML%/output_data'
    - --kernel-size: 40
    - --kernel-overlap: half

- name: Save positive fill delta map instead of filled intensity
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/fill_greyscale_holes.py'
    - --input-search-pattern: '%YAML%/input_data/**/*.tif'
    - --output-folder: '%YAML%/output_data'
    - --kernel-size: 40
    - --return-delta

- name: Save as NumPy array
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --output-format: npy

- name: Save as Ilastik HDF5
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --output-format: ilastik-h5

- name: Save as both OME-TIFF and Ilastik HDF5
    environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --output-format: 'tif ilastik-h5'

Description:
  Fills dark regions (holes) in greyscale images while preserving intensity.
  Perfect for filling nucleoli inside nuclei or similar dark internal structures.
  
  Uses morphological reconstruction (flood-fill from borders) - equivalent to:
    MATLAB: imcomplement(imfill(imcomplement(image), 'holes'))
  
Algorithm:
  1. Inverts image (dark holes become bright peaks)
  2. Seeds from image borders
  3. Flood-fills inward (cannot reach isolated holes)
  4. Inverts back (filled peaks become filled valleys)
  
Use Cases:
  - Fill nucleoli in nucleus images
  - Fill vacuoles in cell bodies
  - Fill any dark internal structures in bright objects
  
2D vs 3D Mode:
  - 2D mode (default): Processes each Z-slice independently
  - 3D mode (--mode-3d): Treats each Z-stack as a single 3D volume
    Use 3D mode when holes extend through multiple Z-slices

Notes:
  - Input should be greyscale (not binary masks)
  - For binary masks, use scipy.ndimage.binary_fill_holes instead
  - Preserves original intensities (unlike binary fill which returns 0/1)
  - Multi-channel images: specify --channels or processes all by default
    - --kernel-size enables local XY window processing
    - --kernel-overlap is overlap in pixels (not movement). stride = kernel_size - overlap
    - Example: kernel_size=40, overlap=1 -> stride=39 (large jumps)
    - Example: kernel_size=40, overlap=39 -> stride=1 (maximal overlap, slowest)
    - Use --kernel-overlap half (default) for balanced speed/quality
    - --return-delta outputs max(filled - input, 0) instead of filled image
        """
    )
    
    parser.add_argument(
        '--input-search-pattern',
        type=str,
        required=True,
        help='Glob pattern for input images (e.g., "data/**/*.tif")'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Output folder for filled images'
    )
    
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_filled',
        help='Suffix to add to output filenames (default: "_filled")'
    )
    
    parser.add_argument(
        '--channels',
        type=_parse_channels,
        default=None,
        help='Channel indices to process (0-based). Space or comma separated. Examples: --channels "0 2" or --channels "0,2"'
    )
    
    parser.add_argument(
        '--mode-3d',
        action='store_true',
        help='Process Z-stacks as 3D volumes instead of individual 2D slices. Use when holes span multiple Z-slices.'
    )

    parser.add_argument(
        '--kernel-size',
        type=int,
        default=None,
        help='Optional local XY kernel size (pixels). If set, holes are filled per kernel window.'
    )

    parser.add_argument(
        '--kernel-overlap',
        type=str,
        default='half',
        help=(
            "Kernel overlap in pixels between adjacent windows (not stride). "
            "Stride = kernel_size - overlap. Use 'half' (default) or integer. "
            "Examples with kernel_size=40: overlap=1 -> stride=39, overlap=39 -> stride=1."
        )
    )

    parser.add_argument(
        '--return-delta',
        action='store_true',
        help='Return positive delta max(filled - input, 0) instead of the filled intensity image.'
    )

    parser.add_argument(
        '--include-input',
        action='store_true',
        help=(
            'Include the original input as channel 0 alongside the filled/delta result as channel 1. '
            'Produces interleaved pairs per processed channel: [orig_c0, filled_c0, orig_c1, filled_c1, ...].'
        )
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU hole filling and force the CPU reconstruction implementation.'
    )
    
    parser.add_argument(
        '--output-format',
        type=parse_output_formats,
        default='tif',
        help=(
            "Output format(s): 'tif' (OME-TIFF), 'npy' (NumPy array), or 'ilastik-h5' (HDF5 for Ilastik). "
            "Specify multiple as a space-separated string, e.g. 'tif npy' or 'tif ilastik-h5'."
        )
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Do not use parallel processing.'
    )

    parser.add_argument(
        '--maxcores',
        type=int,
        default=None,
        help=(
            'Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). '
            'Reduce when processing large files to avoid out-of-memory errors.'
        )
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: WARNING)'
    )
    
    args = parser.parse_args()

    if args.kernel_size is not None and args.kernel_size < 1:
        raise ValueError(f"--kernel-size must be >= 1, got {args.kernel_size}")
    if args.kernel_size is None and str(args.kernel_overlap).strip().lower() != 'half':
        logging.warning("--kernel-overlap is ignored unless --kernel-size is set")
    if args.kernel_size is not None:
        _ = _resolve_kernel_overlap(args.kernel_size, args.kernel_overlap)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Find input files
    logging.info(f"Searching for files: {args.input_search_pattern}")
    input_files = rp.get_files_to_process2(args.input_search_pattern, True)
    
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")
    
    logging.info(f"Found {len(input_files)} files to process")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output folder: {args.output_folder}")
    
    # Define processing function for a single file
    def process_file_wrapper(input_path):
        """Wrapper function for processing a single file (used for parallel processing)."""
        # Generate output stem (without extension)
        input_name = Path(input_path).stem
        output_stem = os.path.join(args.output_folder, f"{input_name}{args.output_suffix}")
        
        try:
            return process_single_image(
                input_path=input_path,
                output_stem=output_stem,
                output_formats=args.output_format,
                channels=args.channels,
                mode_3d=args.mode_3d,
                show_progress=args.no_parallel,  # Only show per-file progress in sequential mode
                kernel_size=args.kernel_size,
                kernel_overlap=args.kernel_overlap,
                return_delta=args.return_delta,
                include_input=args.include_input,
                use_gpu=not args.no_gpu,
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    # Process files (with or without parallel processing)
    if not args.no_parallel:
        from joblib import Parallel, delayed
        n_jobs = rp.resolve_maxcores(args.maxcores, len(input_files))
        logging.info(f"Processing {len(input_files)} files in parallel (n_jobs={n_jobs})...")
        with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
            with _tqdm_joblib(pbar):
                results = list(
                    Parallel(n_jobs=n_jobs)(
                        delayed(process_file_wrapper)(file)
                        for file in input_files
                    )
                )
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
    else:
        logging.info(f"Processing {len(input_files)} files sequentially...")
        successful = 0
        failed = 0
        for i, input_path in enumerate(input_files, 1):
            logging.info(f"\n{'='*70}")
            logging.info(f"Processing file {i}/{len(input_files)}")
            if process_file_wrapper(input_path):
                successful += 1
            else:
                failed += 1
    
    logging.info(f"\n{'='*70}")
    logging.info("Processing complete!")
    logging.info(f"Processed {len(input_files)} files: {successful} succeeded, {failed} failed")
    logging.info(f"Output saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
