#!/usr/bin/env python3
"""
Synthetic data generators for drift correction testing.

This module provides standardized synthetic data generators for testing drift correction algorithms.
All functions follow TCZYX dimension ordering and return (video, ground_truth, true_shifts) tuples.

Available generators:
1. Simple squares (2D/3D) - Clean patterns for algorithm validation (100% accuracy expected)
2. Cell-like patterns (2D/3D) - Realistic Gaussian blobs with noise for robustness testing
3. Template-based drift - Use real images as templates with applied known drift

All generated data uses TCZYX dimension ordering consistently.
"""

import numpy as np
import sys
import os
import logging
import json
from typing import Tuple, List, Dict, Any, Union
# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    # Go up to standard_code/python directory to find bioimage_pipeline_utils
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def create_simple_squares(T: int = 5, C: int = 1, Z: int = 1, Y: int = 100, X: int = 100, 
                         square_size: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, ...]]]:
    """
    Create synthetic video with simple square patterns for drift testing.
    
    This generator creates clean, sharp squares that should yield 100% perfect
    drift correction results. Ideal for validating algorithm accuracy.
    
    Args:
        T: Number of time frames
        C: Number of channels
        Z: Number of Z slices (1 for 2D, >1 for 3D)
        Y: Image height  
        X: Image width
        square_size: Size of squares in pixels
        
    Returns:
        Tuple of (drifted_video, ground_truth_video, applied_shifts)
        - drifted_video: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_video: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of shifts in format (dy, dx) for 2D or (dz, dy, dx) for 3D
    """
    img_data = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
    ground_truth_img_data = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
    
    # Determine if this is 2D or 3D based on Z dimension
    is_3d = Z > 1
    
    if is_3d:
        # Define small, controlled 3D shifts to avoid boundary issues
        # The order is (dz, dy, dx)
        base_shifts = [
            (0, 0, 0),     # Frame 0: no shift (reference)
            (0, 2, 1),     # Frame 1: no Z, down 2, right 1
            (1, -1, 2),    # Frame 2: Z forward 1, up 1, right 2
            (-1, 1, -1),   # Frame 3: Z back 1, down 1, left 1
            (0, -1, 1)     # Frame 4: no Z, up 1, right 1
        ]
    else:
        # Define small, controlled 2D shifts
        # The order is (dy, dx)
        base_shifts = [
            (0, 0),    # Frame 0: no shift (reference)
            (2, 1),    # Frame 1: down 2, right 1
            (-1, 2),   # Frame 2: up 1, right 2
            (1, -1),   # Frame 3: down 1, left 1
            (-1, 1)    # Frame 4: up 1, right 1
        ]
    
    # Extend or truncate shifts to match T
    true_shifts = []
    for t in range(T):
        if t < len(base_shifts):
            true_shifts.append(base_shifts[t])
        else:
            # Generate random shifts for additional frames
            if is_3d:
                shift = (np.random.randint(-1, 2), np.random.randint(-2, 3), np.random.randint(-2, 3))
            else:
                shift = (np.random.randint(-2, 3), np.random.randint(-2, 3))
            true_shifts.append(shift)
    
    logger.info(f"Creating simple squares video with T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    for t, shift in enumerate(true_shifts):
        if is_3d:
            logger.info(f"  Frame {t}: dz={shift[0]}, dy={shift[1]}, dx={shift[2]}")
        else:
            logger.info(f"  Frame {t}: dy={shift[0]}, dx={shift[1]}")
    
    # Base pattern positions (centered squares with different intensities)
    base_y, base_x = Y//2, X//2
    base_z = Z//2 if is_3d else 0
    
    # Reference shift (frame 0)
    ref_shift = true_shifts[0]
    
    for t in range(T):
        current_shift = true_shifts[t]
        
        for c in range(C):
            if is_3d:
                dz, dy, dx = current_shift
                ref_dz, ref_dy, ref_dx = ref_shift
                
                # Create 3D squares (cubes) spanning multiple Z-slices
                for z_offset in [-1, 0, 1]:
                    # Cube 1: top-left quadrant
                    z1, y1, x1 = base_z + z_offset + dz, base_y//2 + dy, base_x//2 + dx
                    if (0 <= z1 < Z and 0 <= y1 <= Y-square_size and 0 <= x1 <= X-square_size):
                        img_data[t, c, z1, y1:y1+square_size, x1:x1+square_size] = 1000
                    
                    # Ground truth at reference position
                    gt_z1, gt_y1, gt_x1 = base_z + z_offset + ref_dz, base_y//2 + ref_dy, base_x//2 + ref_dx
                    if (0 <= gt_z1 < Z and 0 <= gt_y1 <= Y-square_size and 0 <= gt_x1 <= X-square_size):
                        ground_truth_img_data[t, c, gt_z1, gt_y1:gt_y1+square_size, gt_x1:gt_x1+square_size] = 1000
                    
                    # Cube 2: bottom-right quadrant
                    z2, y2, x2 = base_z + z_offset + dz, base_y + base_y//4 + dy, base_x + base_x//4 + dx
                    if (0 <= z2 < Z and 0 <= y2 <= Y-square_size and 0 <= x2 <= X-square_size):
                        img_data[t, c, z2, y2:y2+square_size, x2:x2+square_size] = 2000
                    
                    # Ground truth cube 2
                    gt_z2, gt_y2, gt_x2 = base_z + z_offset + ref_dz, base_y + base_y//4 + ref_dy, base_x + base_x//4 + ref_dx
                    if (0 <= gt_z2 < Z and 0 <= gt_y2 <= Y-square_size and 0 <= gt_x2 <= X-square_size):
                        ground_truth_img_data[t, c, gt_z2, gt_y2:gt_y2+square_size, gt_x2:gt_x2+square_size] = 2000
                    
                    # Cube 3: top-right quadrant
                    z3, y3, x3 = base_z + z_offset + dz, base_y//4 + dy, base_x + base_x//3 + dx
                    if (0 <= z3 < Z and 0 <= y3 <= Y-square_size and 0 <= x3 <= X-square_size):
                        img_data[t, c, z3, y3:y3+square_size, x3:x3+square_size] = 1500
                    
                    # Ground truth cube 3
                    gt_z3, gt_y3, gt_x3 = base_z + z_offset + ref_dz, base_y//4 + ref_dy, base_x + base_x//3 + ref_dx
                    if (0 <= gt_z3 < Z and 0 <= gt_z3 <= Y-square_size and 0 <= gt_x3 <= X-square_size):
                        ground_truth_img_data[t, c, gt_z3, gt_y3:gt_y3+square_size, gt_x3:gt_x3+square_size] = 1500
            else:
                # 2D case
                dy, dx = current_shift
                ref_dy, ref_dx = ref_shift
                
                # Square 1: top-left quadrant
                y1, x1 = base_y//2 + dy, base_x//2 + dx
                if 0 <= y1 <= Y-square_size and 0 <= x1 <= X-square_size:
                    img_data[t, c, 0, y1:y1+square_size, x1:x1+square_size] = 1000
                
                # Ground truth square 1
                gt_y1, gt_x1 = base_y//2 + ref_dy, base_x//2 + ref_dx
                if 0 <= gt_y1 <= Y-square_size and 0 <= gt_x1 <= X-square_size:
                    ground_truth_img_data[t, c, 0, gt_y1:gt_y1+square_size, gt_x1:gt_x1+square_size] = 1000
                
                # Square 2: bottom-right quadrant
                y2, x2 = base_y + base_y//4 + dy, base_x + base_x//4 + dx
                if 0 <= y2 <= Y-square_size and 0 <= x2 <= X-square_size:
                    img_data[t, c, 0, y2:y2+square_size, x2:x2+square_size] = 2000
                
                # Ground truth square 2
                gt_y2, gt_x2 = base_y + base_y//4 + ref_dy, base_x + base_x//4 + ref_dx
                if 0 <= gt_y2 <= Y-square_size and 0 <= gt_x2 <= X-square_size:
                    ground_truth_img_data[t, c, 0, gt_y2:gt_y2+square_size, gt_x2:gt_x2+square_size] = 2000
                
                # Square 3: top-right quadrant
                y3, x3 = base_y//4 + dy, base_x + base_x//3 + dx
                if 0 <= y3 <= Y-square_size and 0 <= x3 <= X-square_size:
                    img_data[t, c, 0, y3:y3+square_size, x3:x3+square_size] = 1500
                
                # Ground truth square 3
                gt_y3, gt_x3 = base_y//4 + ref_dy, base_x + base_x//3 + ref_dx
                if 0 <= gt_y3 <= Y-square_size and 0 <= gt_x3 <= X-square_size:
                    ground_truth_img_data[t, c, 0, gt_y3:gt_y3+square_size, gt_x3:gt_x3+square_size] = 1500
    
    return img_data, ground_truth_img_data, true_shifts


def create_cell_like_pattern(Y: int = 256, X: int = 256, Z: int = 1,
                           num_cells: int = 8, noise_level: float = 5.0) -> np.ndarray:
    """
    Create a synthetic pattern resembling cells using Gaussian blobs.
    
    Args:
        Y: Pattern height in pixels (Y dimension)
        X: Pattern width in pixels (X dimension)
        Z: Pattern depth in pixels (Z dimension, 1 for 2D)
        num_cells: Number of cell-like blobs to create
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        3D numpy array with cell-like pattern (Z, Y, X)
    """
    pattern = np.zeros((Z, Y, X), dtype=np.float32)
    
    # Generate random cell positions and sizes
    np.random.seed(42)  # For reproducible results
    
    is_3d = Z > 1
    
    for i in range(num_cells):
        # Random position (avoid edges)
        margin_xy = max(20, min(Y, X) // 8)
        cx = np.random.randint(margin_xy, X - margin_xy)
        cy = np.random.randint(margin_xy, Y - margin_xy)
        
        if is_3d:
            margin_z = max(2, Z // 8)
            cz = np.random.randint(margin_z, Z - margin_z)
            size_z = np.random.uniform(2, 6)    # Z size (typically smaller)
        else:
            cz = 0
            size_z = 1
        
        # Random size and intensity
        size_xy = np.random.uniform(8, 20) if not is_3d else np.random.uniform(6, 15)
        intensity = np.random.uniform(80, 200)
        
        # Create coordinate grids
        if is_3d:
            z_coords, y_coords, x_coords = np.ogrid[:Z, :Y, :X]
            # Create 3D Gaussian blob
            blob = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * size_xy**2) - 
                          (z_coords - cz)**2 / (2 * size_z**2))
        else:
            y_coords, x_coords = np.ogrid[:Y, :X]
            # Create 2D Gaussian blob (broadcast to 3D with Z=1)
            blob_2d = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * size_xy**2))
            blob = blob_2d[np.newaxis, :, :]  # Add Z dimension
        
        pattern += blob * intensity
    
    # Add background structures
    if is_3d:
        # 3D gradient
        gradient_x = np.linspace(0, 15, X)
        gradient_y = np.linspace(0, 10, Y)
        gradient_z = np.linspace(0, 8, Z)
        
        for z_idx in range(Z):
            pattern[z_idx] += np.outer(gradient_y, gradient_x) + gradient_z[z_idx]
    else:
        # 2D gradient
        gradient_x = np.linspace(0, 20, X)
        gradient_y = np.linspace(0, 15, Y)
        gradient = np.outer(gradient_y, gradient_x)
        pattern[0] += gradient
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, pattern.shape)
        pattern += noise
    
    # Ensure positive values
    pattern = np.maximum(pattern, 0)
    
    return pattern.astype(np.uint16)


def apply_shift_to_volume(volume: np.ndarray, shift_x: float, shift_y: float, shift_z: float = 0.0) -> np.ndarray:
    """
    Apply known shifts to a volume to create synthetic drift.
    
    Args:
        volume: 3D numpy array to shift (Z, Y, X)
        shift_x: Horizontal shift in pixels (positive = right)
        shift_y: Vertical shift in pixels (positive = down)  
        shift_z: Depth shift in pixels (positive = forward)
        
    Returns:
        Shifted volume with same dtype as input
    """
    try:
        from scipy.ndimage import shift as scipy_shift
        # Apply shifts using scipy (creates the "drifted" version)
        if volume.ndim == 2:
            # 2D case: add Z dimension temporarily
            volume_3d = volume[np.newaxis, :, :]
            shifted = scipy_shift(volume_3d.astype(np.float32), [shift_z, shift_y, shift_x], 
                                order=1, prefilter=False)
            return shifted[0].astype(volume.dtype)  # Remove Z dimension
        else:
            # 3D case
            shifted = scipy_shift(volume.astype(np.float32), [shift_z, shift_y, shift_x], 
                                order=1, prefilter=False)
            return shifted.astype(volume.dtype)
    except ImportError:
        # Fallback to numpy roll for integer shifts
        logger.warning("scipy not available, using simple integer shift")
        shift_z_int = int(round(shift_z))
        shift_y_int = int(round(shift_y))
        shift_x_int = int(round(shift_x))
        
        shifted = volume.copy()
        
        # Apply shifts using numpy roll (periodic boundary conditions)
        if volume.ndim == 3 and shift_z_int != 0:
            shifted = np.roll(shifted, shift_z_int, axis=0)
        if shift_y_int != 0:
            axis_y = 1 if volume.ndim == 3 else 0
            shifted = np.roll(shifted, shift_y_int, axis=axis_y)
        if shift_x_int != 0:
            axis_x = 2 if volume.ndim == 3 else 1
            shifted = np.roll(shifted, shift_x_int, axis=axis_x)
        
        return shifted


def create_cell_like_video(T: int = 8, C: int = 1, Z: int = 1, Y: int = 256, X: int = 256, 
                          num_cells: int = 12, noise_level: float = 8.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, ...]]]:
    """
    Create synthetic video with cell-like patterns for realistic drift testing.
    
    This generator creates more realistic cell-like patterns with noise, suitable
    for testing algorithm robustness under more challenging conditions.
    
    Args:
        T: Number of time frames
        C: Number of channels
        Z: Number of Z slices (1 for 2D, >1 for 3D)
        Y: Image height  
        X: Image width
        num_cells: Number of cell-like structures per frame
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Tuple of (img_data, ground_truth_img_data, applied_shifts)
        - img_data: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_img_data: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of shifts in format (dy, dx) for 2D or (dz, dy, dx) for 3D
    """
    # Create base cell pattern
    base_pattern = create_cell_like_pattern(Y, X, Z, num_cells, noise_level)
    
    is_3d = Z > 1
    
    # Define realistic drift shifts (adaptive to image size)
    max_shift_xy = min(Y, X) // 20
    max_shift_z = Z // 10 if is_3d else 0
    
    # Generate cumulative drift
    applied_shifts = []
    if is_3d:
        applied_shifts.append((0.0, 0.0, 0.0))  # Frame 0: no shift (reference) - (dz, dy, dx)
        np.random.seed(456)  # For reproducible results
    else:
        applied_shifts.append((0.0, 0.0))  # Frame 0: no shift (reference) - (dy, dx)
        np.random.seed(123)  # Different seed for 2D
    
    for t in range(1, T):
        if is_3d:
            # 3D random walk drift
            dz = np.random.uniform(-1.0, 1.0)  # Smaller Z drift
            dy = np.random.uniform(-2.0, 2.0)
            dx = np.random.uniform(-2.0, 2.0)
            
            # Accumulate drift (keep within bounds)
            prev_z, prev_y, prev_x = applied_shifts[t-1]
            new_z = np.clip(prev_z + dz, -max_shift_z, max_shift_z)
            new_y = np.clip(prev_y + dy, -max_shift_xy, max_shift_xy)
            new_x = np.clip(prev_x + dx, -max_shift_xy, max_shift_xy)
            
            applied_shifts.append((new_z, new_y, new_x))
        else:
            # 2D random walk drift
            dy = np.random.uniform(-2.5, 2.5)
            dx = np.random.uniform(-2.5, 2.5)
            
            # Accumulate drift (keep within bounds)
            prev_y, prev_x = applied_shifts[t-1]
            new_y = np.clip(prev_y + dy, -max_shift_xy, max_shift_xy)
            new_x = np.clip(prev_x + dx, -max_shift_xy, max_shift_xy)
            
            applied_shifts.append((new_y, new_x))
    
    logger.info(f"Creating cell-like video with T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    for t, shift in enumerate(applied_shifts):
        if is_3d:
            logger.info(f"  Frame {t}: dz={shift[0]:.2f}, dy={shift[1]:.2f}, dx={shift[2]:.2f}")
        else:
            logger.info(f"  Frame {t}: dy={shift[0]:.2f}, dx={shift[1]:.2f}")
    
    # Initialize video arrays
    img_data = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
    ground_truth_img_data = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
    
    for t in range(T):
        shift = applied_shifts[t]
        
        for c in range(C):
            # Ground truth: no shift applied (same base pattern for all frames)
            ground_truth_img_data[t, c] = base_pattern
            
            # Drifted version: apply the shift  
            if is_3d:
                shift_z, shift_y, shift_x = shift
            else:
                shift_y, shift_x = shift
                shift_z = 0.0
            
            img_data[t, c] = apply_shift_to_volume(base_pattern, shift_x, shift_y, shift_z)
    
    return img_data, ground_truth_img_data, applied_shifts


def create_drift_image_from_template(input_file: str, output_file: str, T: int = 10, 
                                   max_shift: float = 20.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, ...]]]:
    """
    Create a synthetic drift video by taking the first timepoint from an input file
    and applying known shifts to simulate drift over time.
    
    Args:
        input_file: Path to input image file (any format supported by bioimage_pipeline_utils)
        output_file: Path to save synthetic drift image as TIFF
        T: Number of timepoints to generate
        max_shift: Maximum shift in pixels
        
    Returns:
        Tuple of (img_data, ground_truth_img_data, applied_shifts)
        - img_data: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_img_data: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of shifts in format (dy, dx) for 2D or (dz, dy, dx) for 3D
    """
    logger.info(f"Creating synthetic drift image from {input_file}")
    
    # Load the input image and ensure TCZYX format
    img = rp.load_tczyx_image(input_file)
    input_img_data = img.data.astype(np.float32)
    
    # Ensure proper TCZYX format with minimum dimension of 1
    orig_T, orig_C, orig_Z, orig_Y, orig_X = input_img_data.shape
    T_actual = max(T, orig_T)  # Use requested T or original T if larger
    C = max(1, orig_C)
    Z = max(1, orig_Z)
    Y = orig_Y
    X = orig_X
    
    logger.info(f"Input image shape: T={T_actual}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    # Use the first timepoint as the base image for all synthetic timepoints
    base_timepoint = input_img_data[0:1]  # Keep as (1, C, Z, Y, X)
    
    # Update T to match requested timepoints
    T = T_actual
    
    is_3d = Z > 1
    
    # Generate known shifts for each timepoint
    applied_shifts = []
    if is_3d:
        applied_shifts.append((0.0, 0.0, 0.0))  # Frame 0: no shift (reference) - (dz, dy, dx)
    else:
        applied_shifts.append((0.0, 0.0))  # Frame 0: no shift (reference) - (dy, dx)
    
    # Generate cumulative drift pattern
    np.random.seed(42)  # For reproducible results
    
    for t in range(1, T):
        if is_3d:
            # Generate 3D drift increments
            dz = np.random.normal(0, max_shift / 20)  # Smaller Z drift
            dy = np.random.normal(0, max_shift / 10)  # Y drift
            dx = np.random.normal(0, max_shift / 10)  # X drift
            
            # Cumulative drift (add to previous position)
            prev_z, prev_y, prev_x = applied_shifts[t-1]
            new_shifts = (prev_z + dz, prev_y + dy, prev_x + dx)
            
            # Limit maximum cumulative shift
            new_shifts = (
                np.clip(new_shifts[0], -max_shift, max_shift),
                np.clip(new_shifts[1], -max_shift, max_shift),
                np.clip(new_shifts[2], -max_shift, max_shift)
            )
        else:
            # Generate 2D drift increments
            dy = np.random.normal(0, max_shift / 10)  # Y drift
            dx = np.random.normal(0, max_shift / 10)  # X drift
            
            # Cumulative drift (add to previous position)
            prev_y, prev_x = applied_shifts[t-1]
            new_shifts = (prev_y + dy, prev_x + dx)
            
            # Limit maximum cumulative shift
            new_shifts = (
                np.clip(new_shifts[0], -max_shift, max_shift),
                np.clip(new_shifts[1], -max_shift, max_shift)
            )
        
        applied_shifts.append(new_shifts)
    
    logger.info(f"Generated shifts ranging from min to max pixels")
    
    # Create synthetic drift stack and ground truth
    img_data = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    ground_truth_img_data = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    
    for t in range(T):
        shift = applied_shifts[t]
        
        for c in range(C):
            # Ground truth: no shift applied (same base pattern for all frames)
            ground_truth_img_data[t, c] = base_timepoint[0, c]
            
            # Get the base volume for this channel
            base_volume = base_timepoint[0, c]  # Shape: (Z, Y, X)
            
            # Apply shifts
            if is_3d:
                shift_z, shift_y, shift_x = shift
            else:
                shift_y, shift_x = shift
                shift_z = 0.0
            
            img_data[t, c] = apply_shift_to_volume(base_volume, shift_x, shift_y, shift_z)
    
    # Convert back to uint16
    img_data = img_data.astype(np.uint16)
    ground_truth_img_data = ground_truth_img_data.astype(np.uint16)
    
    # Save the synthetic drift image
    logger.info(f"Saving synthetic drift image to {output_file}")
    rp.save_tczyx_image(img_data, output_file)
    
    # Save ground truth
    ground_truth_file = os.path.splitext(output_file)[0] + "_ground_truth.tif"
    rp.save_tczyx_image(ground_truth_img_data, ground_truth_file)
    
    # Save the known shifts as JSON file
    shifts_file = os.path.splitext(output_file)[0] + "_known_shifts.json"
    shifts_data = {
        "description": "Known shifts applied to create synthetic drift image",
        "input_file": input_file,
        "output_file": output_file,
        "ground_truth_file": ground_truth_file,
        "T": T,
        "max_shift": max_shift,
        "random_seed": 42,
        "shifts_format": "YX order (dy, dx) for 2D or ZYX order (dz, dy, dx) for 3D",
        "is_3d": is_3d,
        "shifts": [list(shift) for shift in applied_shifts]
    }
    
    with open(shifts_file, 'w') as f:
        json.dump(shifts_data, f, indent=2)
    
    logger.info(f"Saved known shifts to {shifts_file}")
    logger.info(f"Successfully created synthetic drift image with {T} timepoints")
    
    return img_data, ground_truth_img_data, applied_shifts


def save_synthetic_datasets(img_data: np.ndarray, ground_truth_img_data: np.ndarray, 
                          applied_shifts: List[Tuple], metadata: Dict[str, Any], 
                          output_dir: str, dataset_name: str = "synthetic") -> None:
    """
    Save synthetic datasets and metadata to disk.
    
    Args:
        img_data: Video with applied drift (T,C,Z,Y,X)
        ground_truth_img_data: Reference video without drift (T,C,Z,Y,X)
        applied_shifts: List of shifts applied
        metadata: Dictionary with dataset metadata
        output_dir: Directory to save files
        dataset_name: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save videos
    drifted_path = os.path.join(output_dir, f"{dataset_name}.tif")
    ground_truth_path = os.path.join(output_dir, f"{dataset_name}_ground_truth.tif")
    
    logger.info(f"Saving drifted video to: {drifted_path}")
    rp.save_tczyx_image(img_data, drifted_path)
    
    logger.info(f"Saving ground truth to: {ground_truth_path}")
    rp.save_tczyx_image(ground_truth_img_data, ground_truth_path)
    
    # Save metadata
    T, C, Z, Y, X = img_data.shape
    is_3d = Z > 1
    
    full_metadata = {
        'dataset_info': {
            'name': dataset_name,
            'creation_date': str(np.datetime64('now')),
            'dimensions': {
                'T': int(T), 'C': int(C), 'Z': int(Z), 'Y': int(Y), 'X': int(X)
            },
            'is_3d': is_3d
        },
        'drift_info': {
            'applied_shifts': [list(shift) for shift in applied_shifts],
            'shifts_format': "YX order (dy, dx) for 2D or ZYX order (dz, dy, dx) for 3D",
            'max_drift_magnitude': float(np.max([np.sqrt(sum(coord**2 for coord in shift)) for shift in applied_shifts]))
        },
        'pattern_info': metadata
    }
    
    metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata.yaml")
    import yaml
    with open(metadata_path, 'w') as f:
        yaml.dump(full_metadata, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved metadata to: {metadata_path}")
    logger.info(f"Dataset '{dataset_name}' saved to: {output_dir}")



def test_all_generators():
    """
    Test all synthetic data generators and save outputs.
    """
    output_base_dir = "synthetic_datasets"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Test simple squares (2D)
    img_data, gt_data, shifts = create_simple_squares(T=5, C=1, Z=1, Y=100, X=100, square_size=10)
    print(img_data.shape, gt_data.shape, shifts)
    #metadata = {'pattern': 'simple_squares', 'description': 'Clean squares for drift testing'}
    #save_synthetic_datasets(img_data, gt_data, shifts, metadata, output_base_dir, "simple_squares_2d")
    
    # Test simple squares (3D)
    img_data, gt_data, shifts = create_simple_squares(T=5, C=1, Z=5, Y=100, X=100, square_size=10)
    print(img_data.shape, gt_data.shape, shifts)
    #metadata = {'pattern': 'simple_squares', 'description': 'Clean cubes for 3D drift testing'}
    #save_synthetic_datasets(img_data, gt_data, shifts, metadata, output_base_dir, "simple_squares_3d")
    
    # Test cell-like patterns (2D)
    img_data, gt_data, shifts = create_cell_like_video(T=8, C=1, Z=1, Y=256, X=256, num_cells=12, noise_level=8.0)
    print(img_data.shape, gt_data.shape, shifts)

    #metadata = {'pattern': 'cell_like', 'description': 'Realistic cell-like patterns with noise'}
    #save_synthetic_datasets(img_data, gt_data, shifts, metadata, output_base_dir, "cell_like_2d")
    
    # Test cell-like patterns (3D)
    img_data, gt_data, shifts = create_cell_like_video(T=8, C=1, Z=5, Y=128, X=128, num_cells=10, noise_level=5.0)
    print(img_data.shape, gt_data.shape, shifts)
    #metadata = {'pattern': 'cell_like', 'description': 'Realistic 3D cell-like patterns with noise'}
    #save_synthetic_datasets(img_data, gt_data, shifts, metadata, output_base_dir, "cell_like_3d")
    



def main():
    test_all_generators()



if __name__ == "__main__":
    main()