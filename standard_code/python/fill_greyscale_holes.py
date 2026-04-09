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


def _estimate_nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


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


def greyscale_fill_holes_2d(image: np.ndarray) -> np.ndarray:
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
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    # Handle edge case: empty or constant image
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    
    # Invert the image (holes become peaks)
    image_max = image.max()
    inverted = image_max - image
    
    # Create seed: keep border, clear interior
    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()
    
    # Morphological reconstruction: flood fill from borders
    # Border regions are connected, holes are isolated
    reconstructed = reconstruction(seed, inverted, method='dilation')
    
    # Invert back to get filled result
    filled = image_max - reconstructed
    
    return filled


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
            filled_tile = greyscale_fill_holes_2d(tile)
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
                for t in range(T):
                    # Load just one timepoint: shape (C, Z, Y, X) — far smaller than the full array
                    czyx = _load_czyx_timepoint(img, t)

                    if include_input:
                        out_czyx = np.empty((n_out, Z, Y, X), dtype=img_dtype)
                    else:
                        out_czyx = czyx.copy()  # preserve non-processed channels

                    for c_idx, c in enumerate(channels_to_process):
                        if kernel_size is None and mode_3d and Z > 1:
                            raise NotImplementedError("3D kernel mode is not implemented. Use 2D kernel or disable kernel mode.")
                        else:
                            for z in range(Z):
                                slice_2d = czyx[c, z]
                                filled_slice = (
                                    greyscale_fill_holes_kernel_2d(
                                        slice_2d,
                                        kernel_size=kernel_size,
                                        kernel_overlap=overlap_px,
                                    )
                                    if kernel_size is not None
                                    else greyscale_fill_holes_2d(slice_2d)
                                )
                                result_2d = (
                                    _positive_delta(filled_slice, slice_2d, img_dtype)
                                    if return_delta
                                    else _cast_to_dtype(filled_slice, img_dtype)
                                )
                                if include_input:
                                    out_czyx[c_idx * 2, z] = slice_2d
                                    out_czyx[c_idx * 2 + 1, z] = result_2d
                                else:
                                    out_czyx[c, z] = result_2d

                        pbar.update(1)

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
        '--max-workers',
        type=int,
        default=None,
        help=(
            'Maximum number of parallel worker processes (default: use all CPU cores). '
            'Reduce when processing large files to avoid out-of-memory errors.'
        )
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
        level=logging.INFO,
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
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    # Process files (with or without parallel processing)
    if not args.no_parallel:
        from joblib import Parallel, delayed
        n_jobs = -1 if args.max_workers is None else args.max_workers
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
