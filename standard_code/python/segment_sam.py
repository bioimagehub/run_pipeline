"""
Segment Anything Model (SAM) Segmentation for Microscopy Images

Interactive and automated segmentation using Micro-SAM (Segment Anything for microscopy).
Supports manual point prompts (positive and negative clicks), box prompts, and batch processing.

This module implements the Micro-SAM framework for bioimage segmentation, providing
point-based and box-based prompting for precise object segmentation in microscopy data.

Author: BIPHUB, University of Oslo
Written by: Øyvind Ødegård Fougner
License: MIT
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def parse_point_string(point_str: str) -> list[tuple[float, float, float]]:
    """
    Parse space-separated point coordinates from string.
    
    Format: "x1,y1,z1 x2,y2,z2 x3,y3,z3"
    Z coordinate is optional, defaults to 0 if not provided.
    
    Parameters
    ----------
    point_str : str
        Space-separated point coordinates. Each point is "x,y,z" or "x,y".
    
    Returns
    -------
    points : list[tuple[float, float, float]]
        List of (x, y, z) coordinates.
    
    Examples
    --------
    >>> parse_point_string("512,384,10 256,128,5")
    [(512.0, 384.0, 10.0), (256.0, 128.0, 5.0)]
    
    >>> parse_point_string("512,384 256,128")  # Z defaults to 0
    [(512.0, 384.0, 0.0), (256.0, 128.0, 0.0)]
    """
    if not point_str or not point_str.strip():
        return []
    
    points = []
    for point in point_str.strip().split():
        coords = point.split(',')
        if len(coords) < 2:
            logger.warning(f"Invalid point format: {point} (need at least x,y)")
            continue
        
        x = float(coords[0])
        y = float(coords[1])
        z = float(coords[2]) if len(coords) >= 3 else 0.0
        
        points.append((x, y, z))
    
    return points


def interactive_point_selection(
    image: np.ndarray,
    z_slice: int = 0
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Interactive point selection using matplotlib with mouse clicks.
    
    Left click = positive (foreground) point
    Right click = negative (background) point
    Middle click or 'Enter' = finish selection
    'Escape' = cancel and return empty lists
    
    Parameters
    ----------
    image : np.ndarray
        Image array of shape (Z, Y, X) for 3D or (Y, X) for 2D.
    z_slice : int
        Z-slice to display for 3D images. Default is 0 (first slice).
    
    Returns
    -------
    positive_points : list[tuple[float, float, float]]
        List of positive (foreground) points as (x, y, z).
    negative_points : list[tuple[float, float, float]]
        List of negative (background) points as (x, y, z).
    
    Notes
    -----
    - For 3D images, displays a single Z-slice for point selection
    - All points get the same Z coordinate (the displayed slice)
    - Use arrow keys or scrollwheel to change Z-slice before clicking (if implemented)
    """
    positive_points = []
    negative_points = []
    
    # Prepare display image
    is_3d = image.ndim == 3
    if is_3d:
        if z_slice < 0 or z_slice >= image.shape[0]:
            logger.warning(f"Z-slice {z_slice} out of range (0-{image.shape[0]-1}), using 0")
            z_slice = 0
        display_img = image[z_slice]
        logger.info(f"Displaying Z-slice {z_slice} of {image.shape[0]} for point selection")
    else:
        display_img = image
        z_slice = 0
    
    # Normalize for display
    display_img = display_img.astype(float)
    p1, p99 = np.percentile(display_img, [1, 99])
    display_img = np.clip((display_img - p1) / (p99 - p1), 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(display_img, cmap='gray', interpolation='nearest')
    ax.set_title(
        "Interactive Point Selection\n"
        "Left Click: Positive (foreground) | Right Click: Negative (background)\n"
        "Press ENTER or Middle Click when done | ESC to cancel",
        fontsize=11
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Storage for plot objects
    pos_scatter = ax.scatter([], [], c='lime', s=100, marker='+', linewidths=2, label='Positive')
    neg_scatter = ax.scatter([], [], c='red', s=100, marker='x', linewidths=2, label='Negative')
    ax.legend(loc='upper right')
    
    finished = [False]  # Use list to allow modification in nested function
    
    def on_click(event):
        """Handle mouse click events."""
        if event.inaxes != ax:
            return
        
        x, y = event.xdata, event.ydata
        
        # Left click = positive point
        if event.button == MouseButton.LEFT:
            positive_points.append((x, y, float(z_slice)))
            logger.info(f"Added positive point at ({x:.1f}, {y:.1f}, {z_slice})")
            
            # Update plot
            pos_x = [p[0] for p in positive_points]
            pos_y = [p[1] for p in positive_points]
            pos_scatter.set_offsets(np.c_[pos_x, pos_y])
            
        # Right click = negative point
        elif event.button == MouseButton.RIGHT:
            negative_points.append((x, y, float(z_slice)))
            logger.info(f"Added negative point at ({x:.1f}, {y:.1f}, {z_slice})")
            
            # Update plot
            neg_x = [p[0] for p in negative_points]
            neg_y = [p[1] for p in negative_points]
            neg_scatter.set_offsets(np.c_[neg_x, neg_y])
        
        # Middle click = finish
        elif event.button == MouseButton.MIDDLE:
            logger.info("Middle click detected, finishing selection")
            finished[0] = True
            plt.close(fig)
        
        fig.canvas.draw_idle()
    
    def on_key(event):
        """Handle keyboard events."""
        if event.key == 'enter':
            logger.info("Enter pressed, finishing selection")
            finished[0] = True
            plt.close(fig)
        elif event.key == 'escape':
            logger.info("Escape pressed, canceling selection")
            positive_points.clear()
            negative_points.clear()
            finished[0] = True
            plt.close(fig)
        elif event.key == 'backspace' and (positive_points or negative_points):
            # Remove last point (either positive or negative)
            if negative_points:
                removed = negative_points.pop()
                logger.info(f"Removed last negative point: ({removed[0]:.1f}, {removed[1]:.1f})")
                neg_x = [p[0] for p in negative_points]
                neg_y = [p[1] for p in negative_points]
                neg_scatter.set_offsets(np.c_[neg_x, neg_y] if neg_x else np.empty((0, 2)))
            elif positive_points:
                removed = positive_points.pop()
                logger.info(f"Removed last positive point: ({removed[0]:.1f}, {removed[1]:.1f})")
                pos_x = [p[0] for p in positive_points]
                pos_y = [p[1] for p in positive_points]
                pos_scatter.set_offsets(np.c_[pos_x, pos_y] if pos_x else np.empty((0, 2)))
            
            fig.canvas.draw_idle()
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show plot and wait for user
    plt.tight_layout()
    plt.show(block=True)
    
    if not finished[0]:
        logger.warning("Plot closed unexpectedly")
    
    logger.info(f"Interactive selection complete: {len(positive_points)} positive, {len(negative_points)} negative points")
    
    return positive_points, negative_points


def segment_with_sam(
    image: np.ndarray,
    positive_points: list[tuple[float, float, float]],
    negative_points: list[tuple[float, float, float]] | None = None,
    model_type: str = "vit_h",
    checkpoint_path: str | None = None,
    device: str | None = None
) -> np.ndarray:
    """
    Segment objects using Micro-SAM with point prompts.
    
    Parameters
    ----------
    image : np.ndarray
        Input image of shape (Z, Y, X) or (Y, X) for 2D.
        Should be grayscale or RGB (will use first channel if multi-channel).
    positive_points : list[tuple[float, float, float]]
        Foreground point prompts as (x, y, z) coordinates.
    negative_points : list[tuple[float, float, float]] | None
        Background point prompts as (x, y, z) coordinates. Optional.
    model_type : str
        SAM model variant: "vit_h" (huge, most accurate), "vit_l" (large), "vit_b" (base, fastest).
        Default is "vit_h".
    checkpoint_path : str | None
        Path to custom model checkpoint. If None, downloads default Micro-SAM weights.
    device : str | None
        Device to run on: "cuda", "cpu", or None (auto-detect). Default is None (auto).
    
    Returns
    -------
    mask : np.ndarray
        Binary segmentation mask of same shape as input (0 or 255).
    
    Notes
    -----
    - First run will download ~2.4GB model weights for vit_h
    - Uses Micro-SAM's `segment_from_points` function
    - Processes 2D slices independently for 3D volumes
    - Point coordinates are in pixel space (x, y, z)
    """
    try:
        from micro_sam.util import get_sam_model
        from micro_sam.prompt_based_segmentation import segment_from_points
        import torch
    except ImportError as e:
        logger.error(
            "Micro-SAM not installed. Install with: "
            "uv pip install --group micro-sam micro-sam"
        )
        raise e
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading Micro-SAM model: {model_type}")
    if checkpoint_path is not None:
        logger.info(f"Using checkpoint: {checkpoint_path}")
    
    predictor = get_sam_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Prepare image
    is_3d = image.ndim == 3
    if not is_3d and image.ndim != 2:
        raise ValueError(f"Expected 2D (Y,X) or 3D (Z,Y,X) image, got shape {image.shape}")
    
    # Initialize output mask
    output_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Combine all points
    negative_points = negative_points or []
    all_points = positive_points + negative_points
    
    if not all_points:
        logger.warning("No points provided, returning empty mask")
        return output_mask
    
    # Process 2D image
    if not is_3d:
        logger.info(f"Processing 2D image ({image.shape[0]}x{image.shape[1]})")
        
        # Extract 2D point coordinates (ignore Z)
        point_coords = np.array([[p[0], p[1]] for p in all_points], dtype=np.float32)
        point_labels = np.array(
            [1] * len(positive_points) + [0] * len(negative_points),
            dtype=np.int32
        )
        
        logger.info(f"Running SAM with {len(positive_points)} positive and {len(negative_points)} negative points")
        
        # Run segmentation
        mask = segment_from_points(
            predictor=predictor,
            image=image,
            points=point_coords,
            labels=point_labels
        )
        
        output_mask = (mask > 0).astype(np.uint8) * 255
        logger.info(f"Segmented {np.sum(output_mask > 0)} pixels")
    
    # Process 3D volume (slice by slice)
    else:
        logger.info(f"Processing 3D volume ({image.shape[0]} slices of {image.shape[1]}x{image.shape[2]})")
        
        # Group points by Z slice
        z_slices = {}
        for i, p in enumerate(positive_points):
            z = int(round(p[2]))
            if z not in z_slices:
                z_slices[z] = {'positive': [], 'negative': []}
            z_slices[z]['positive'].append((p[0], p[1]))
        
        for i, p in enumerate(negative_points):
            z = int(round(p[2]))
            if z not in z_slices:
                z_slices[z] = {'positive': [], 'negative': []}
            z_slices[z]['negative'].append((p[0], p[1]))
        
        logger.info(f"Points distributed across {len(z_slices)} Z-slices")
        
        # Process each Z slice that has points
        for z, points_dict in tqdm(z_slices.items(), desc="Segmenting slices", unit="slice"):
            if z < 0 or z >= image.shape[0]:
                logger.warning(f"Z={z} out of bounds (0-{image.shape[0]-1}), skipping")
                continue
            
            pos_pts = points_dict['positive']
            neg_pts = points_dict['negative']
            
            if not pos_pts:
                logger.warning(f"Z={z}: no positive points, skipping")
                continue
            
            # Extract 2D coordinates
            slice_2d = image[z]
            point_coords = np.array(pos_pts + neg_pts, dtype=np.float32)
            point_labels = np.array(
                [1] * len(pos_pts) + [0] * len(neg_pts),
                dtype=np.int32
            )
            
            logger.debug(f"Z={z}: {len(pos_pts)} positive, {len(neg_pts)} negative points")
            
            # Run segmentation on this slice
            mask = segment_from_points(
                predictor=predictor,
                image=slice_2d,
                points=point_coords,
                labels=point_labels
            )
            
            output_mask[z] = (mask > 0).astype(np.uint8) * 255
            logger.debug(f"Z={z}: segmented {np.sum(mask > 0)} pixels")
        
        total_pixels = np.sum(output_mask > 0)
        logger.info(f"Total segmented pixels across all slices: {total_pixels}")
    
    return output_mask


def process_single_file(
    input_path: str,
    output_path: str,
    positive_points: list[tuple[float, float, float]],
    negative_points: list[tuple[float, float, float]] | None,
    channels: list[int] | None,
    timepoints: list[int] | None,
    model_type: str,
    checkpoint_path: str | None,
    device: str | None,
    force: bool
) -> bool:
    """
    Process a single image file with SAM segmentation.
    
    Parameters
    ----------
    input_path : str
        Path to input image file.
    output_path : str
        Path where segmented mask will be saved.
    positive_points : list[tuple[float, float, float]]
        Foreground point prompts.
    negative_points : list[tuple[float, float, float]] | None
        Background point prompts (optional).
    channels : list[int] | None
        Channel indices to process. If None, processes all channels.
    timepoints : list[int] | None
        Timepoint indices to process. If None, processes all timepoints.
    model_type : str
        SAM model variant ("vit_h", "vit_l", "vit_b").
    checkpoint_path : str | None
        Path to custom model checkpoint.
    device : str | None
        Device to run on ("cuda", "cpu", or None for auto).
    force : bool
        If True, overwrite existing output files.
    
    Returns
    -------
    success : bool
        True if processing succeeded, False otherwise.
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load image
        img = rp.load_tczyx_image(input_path)
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Determine which channels to process
        if channels is None:
            channels_to_process = list(range(C))
        else:
            channels_to_process = [c for c in channels if 0 <= c < C]
            if not channels_to_process:
                raise ValueError(f"No valid channels. Image has {C} channels, requested {channels}")
        
        # Determine which timepoints to process
        if timepoints is None:
            timepoints_to_process = list(range(T))
        else:
            timepoints_to_process = [t for t in timepoints if 0 <= t < T]
            if not timepoints_to_process:
                raise ValueError(f"No valid timepoints. Image has {T} timepoints, requested {timepoints}")
        
        logger.info(f"Processing channels: {channels_to_process}")
        logger.info(f"Processing timepoints: {timepoints_to_process}")
        logger.info(f"Positive points: {len(positive_points)}, Negative points: {len(negative_points or [])}")
        
        # Create output array
        output_data = np.zeros_like(img.data, dtype=np.uint8)
        
        # Process each timepoint and channel
        total_tasks = len(timepoints_to_process) * len(channels_to_process)
        with tqdm(total=total_tasks, desc="Segmenting", unit="volume") as pbar:
            for t in timepoints_to_process:
                for c in channels_to_process:
                    # Extract ZYX volume for this timepoint and channel
                    volume = img.data[t, c, :, :, :]  # Shape: (Z, Y, X)
                    
                    # Skip empty volumes
                    if not volume.any():
                        logger.debug(f"T={t}, C={c}: empty volume, skipping")
                        pbar.update(1)
                        continue
                    
                    # Run SAM segmentation
                    mask = segment_with_sam(
                        image=volume,
                        positive_points=positive_points,
                        negative_points=negative_points,
                        model_type=model_type,
                        checkpoint_path=checkpoint_path,
                        device=device
                    )
                    
                    output_data[t, c, :, :, :] = mask
                    pbar.update(1)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(output_data, output_path)
        logger.info(f"Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Segment Anything Model (Micro-SAM) for microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: SAM segmentation with manual clicks
  environment: uv@3.11:micro-sam
  commands:
  - python
  - '%REPO%/standard_code/python/segment_sam.py'
  - --input-search-pattern: '%YAML%/raw_images/**/*.tif'
  - --output-folder: '%YAML%/sam_masks'
  - --positive-points: "512,384,10 600,400,10"
  - --negative-points: "100,100,10"
  - --model-type: vit_h
  - --channels
  - '0'

- name: SAM interactive mode
  environment: uv@3.11:micro-sam
  commands:
  - python
  - '%REPO%/standard_code/python/segment_sam.py'
  - --input-search-pattern: '%YAML%/raw_images/**/*.tif'
  - --output-folder: '%YAML%/sam_masks'
  - --interactive
  - --model-type: vit_h

- name: SAM interactive with specific Z-slice
  environment: uv@3.11:micro-sam
  commands:
  - python
  - '%REPO%/standard_code/python/segment_sam.py'
  - --input-search-pattern: '%YAML%/raw_images/**/*.tif'
  - --output-folder: '%YAML%/sam_masks'
  - --interactive
  - --interactive-z-slice: 5
  - --channels
  - '0'

- name: SAM segmentation 2D image
  environment: uv@3.11:micro-sam
  commands:
  - python
  - '%REPO%/standard_code/python/segment_sam.py'
  - --input-search-pattern: '%YAML%/raw_images/*.tif'
  - --output-folder: '%YAML%/sam_masks'
  - --positive-points: "512,384"
  - --model-type: vit_b

- name: SAM with multiple timepoints
  environment: uv@3.11:micro-sam
  commands:
  - python
  - '%REPO%/standard_code/python/segment_sam.py'
  - --input-search-pattern: '%YAML%/timeseries/**/*.tif'
  - --output-folder: '%YAML%/sam_masks'
  - --positive-points: "512,384,5"
  - --timepoints
  - '0'
  - '10'
  - '20'

Description:
  Segments objects in microscopy images using Micro-SAM (Segment Anything for microscopy).
  Supports manual point prompts (positive/negative clicks) for interactive segmentation.
  
Interactive Mode:
  Use --interactive to display the first image and click to select points:
  - Left Click: Add positive (foreground) point
  - Right Click: Add negative (background) point  
  - Press ENTER or Middle Click: Finish selection and start processing
  - Press ESC: Cancel and exit
  - Press BACKSPACE: Remove last point
  
  The selected points will be applied to all images in the batch.

Point Format:
  - 2D: "x,y" (e.g., "512,384")
  - 3D: "x,y,z" (e.g., "512,384,10")
  - Multiple points: space-separated (e.g., "512,384,10 600,400,10")

Model Types:
  - vit_h (huge): Most accurate, ~2.4GB, slowest (recommended)
  - vit_l (large): Balanced, ~1.2GB, medium speed
  - vit_b (base): Fastest, ~350MB, lower accuracy

Usage Notes:
  - First run downloads model weights automatically
  - GPU strongly recommended (50-100x faster than CPU)
  - Processes each channel and timepoint independently
  - For 3D volumes, segments slice-by-slice at Z coordinates of points
  - Positive points = foreground (objects to segment)
  - Negative points = background (areas to exclude)

Typical Workflow:
  1. View image in ImageJ/napari to identify points
  2. Note x,y,z coordinates of object centers (positive)
  3. Optionally note background points (negative)
  4. Run segmentation with manual click coordinates
  5. Iterate: add negative points if over-segmented
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input image file pattern (supports wildcards, use '**' for recursive)"
    )
    
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: input_folder + '_sam_masks')"
    )
    
    parser.add_argument(
        "--positive-points",
        required=False,
        help=(
            "Positive (foreground) point coordinates. "
            "Format: 'x1,y1,z1 x2,y2,z2' (space-separated). "
            "Z is optional for 2D images. "
            "Example: '512,384,10 600,400,10'. "
            "Not required if --interactive is set."
        )
    )
    
    parser.add_argument(
        "--negative-points",
        default=None,
        help=(
            "Negative (background) point coordinates. "
            "Format: 'x1,y1,z1 x2,y2,z2' (space-separated). "
            "Example: '100,100,10 900,900,10'"
        )
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Enable interactive mode: display first image and let user click to select points. "
            "Left click = positive (foreground), Right click = negative (background). "
            "Press ENTER or middle-click when done, ESC to cancel, BACKSPACE to undo last point."
        )
    )
    
    parser.add_argument(
        "--interactive-z-slice",
        type=int,
        default=None,
        help="Z-slice to display in interactive mode (default: middle slice)"
    )
    
    parser.add_argument(
        "--channels",
        type=int,
        nargs='+',
        default=None,
        help="Channel indices to process (0-based). If not specified, processes all channels."
    )
    
    parser.add_argument(
        "--timepoints",
        type=int,
        nargs='+',
        default=None,
        help="Timepoint indices to process (0-based). If not specified, processes all timepoints."
    )
    
    parser.add_argument(
        "--model-type",
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_h",
        help=(
            "SAM model variant: "
            "vit_h (huge, most accurate, ~2.4GB), "
            "vit_l (large, balanced, ~1.2GB), "
            "vit_b (base, fastest, ~350MB). "
            "Default: vit_h"
        )
    )
    
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to custom model checkpoint. If not provided, uses default Micro-SAM weights."
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run on: 'cuda' or 'cpu'. If not specified, auto-detects."
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output files exist"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.positive_points:
        parser.error("--positive-points is required unless --interactive is set")
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"segment_sam.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handle interactive mode
    if args.interactive:
        logger.info("=== Interactive Mode ===")
        logger.info("You will be shown the first image for point selection")
        
        # Get first file
        files_to_process = rp.get_files_to_process2(args.input_search_pattern, search_subfolders="**" in args.input_search_pattern)
        if not files_to_process:
            logger.error(f"No files found matching pattern: {args.input_search_pattern}")
            return
        
        first_file = files_to_process[0]
        logger.info(f"Loading first image: {os.path.basename(first_file)}")
        
        # Load image for interactive selection
        img = rp.load_tczyx_image(first_file)
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Determine which channel/timepoint to show
        display_channel = args.channels[0] if args.channels else 0
        display_timepoint = args.timepoints[0] if args.timepoints else 0
        display_z = args.interactive_z_slice if hasattr(args, 'interactive_z_slice') and args.interactive_z_slice is not None else Z // 2
        
        if display_channel >= C:
            logger.warning(f"Requested channel {display_channel} >= {C}, using channel 0")
            display_channel = 0
        if display_timepoint >= T:
            logger.warning(f"Requested timepoint {display_timepoint} >= {T}, using timepoint 0")
            display_timepoint = 0
        
        logger.info(f"Displaying T={display_timepoint}, C={display_channel}, Z={display_z}")
        
        # Extract volume for display
        volume = img.data[display_timepoint, display_channel, :, :, :]  # Shape: (Z, Y, X)
        
        # Run interactive selection
        positive_points, negative_points = interactive_point_selection(volume, z_slice=display_z)
        
        if not positive_points:
            logger.error("No positive points selected, exiting")
            return
        
        logger.info(f"Selected {len(positive_points)} positive and {len(negative_points)} negative points")
        logger.info("These points will be used for all images in the batch")
        
        # Continue with normal processing using these points
        args.positive_points = " ".join([f"{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}" for p in positive_points])
        if negative_points:
            args.negative_points = " ".join([f"{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}" for p in negative_points])
    
    # Parse point coordinates (either from CLI or from interactive mode)
    positive_points = parse_point_string(args.positive_points)
    negative_points = parse_point_string(args.negative_points) if args.negative_points else None
    
    if not positive_points:
        logger.error("No valid positive points provided")
        return
    
    logger.info(f"Loaded {len(positive_points)} positive point(s)")
    if negative_points:
        logger.info(f"Loaded {len(negative_points)} negative point(s)")
    
    # Get files to process
    files_to_process = rp.get_files_to_process2(args.input_search_pattern, search_subfolders="**" in args.input_search_pattern)
    
    if not files_to_process:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        return
    
    logger.info(f"Found {len(files_to_process)} file(s) to process")
    
    # Default output folder
    if args.output_folder is None:
        base_dir = os.path.dirname(args.input_search_pattern.replace("**/", "").replace("*", ""))
        args.output_folder = (base_dir or ".") + "_sam_masks"
    
    logger.info(f"Output folder: {args.output_folder}")
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Dry run
    if args.dry_run:
        print(f"[DRY RUN] Would process {len(files_to_process)} file(s)")
        print(f"[DRY RUN] Output folder: {args.output_folder}")
        print(f"[DRY RUN] Model: {args.model_type}")
        print(f"[DRY RUN] Device: {args.device or 'auto-detect'}")
        print(f"[DRY RUN] Positive points: {positive_points}")
        print(f"[DRY RUN] Negative points: {negative_points}")
        if args.channels:
            print(f"[DRY RUN] Channels: {args.channels}")
        if args.timepoints:
            print(f"[DRY RUN] Timepoints: {args.timepoints}")
        for input_path in files_to_process[:5]:
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + "_sam_mask.tif"
            print(f"[DRY RUN]   {os.path.basename(input_path)} -> {output_filename}")
        if len(files_to_process) > 5:
            print(f"[DRY RUN]   ... and {len(files_to_process) - 5} more files")
        return
    
    # Process files
    ok = 0
    for input_path in files_to_process:
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + "_sam_mask.tif"
        output_path = os.path.join(args.output_folder, output_filename)
        
        if process_single_file(
            input_path=input_path,
            output_path=output_path,
            positive_points=positive_points,
            negative_points=negative_points,
            channels=args.channels,
            timepoints=args.timepoints,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            force=args.force
        ):
            ok += 1
    
    logger.info(f"Done: {ok} succeeded, {len(files_to_process)-ok} failed")


if __name__ == "__main__":
    main()
