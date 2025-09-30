#!/usr/bin/env python3
"""
Synthetic data generators for drift correction testing.

This module provides two types of synthetic data:
1. Simple squares - Guaranteed 100% accurate reconstruction
2. Cell-like patterns - More realistic Gaussian blobs with noise

Both generate two datasets:
- Drifted video: Contains intentional drift to test correction
- Ground truth: Reference video without drift for validation
"""

import numpy as np
import sys
import os
from typing import Tuple, List, Dict, Any
# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import bioimage_pipeline_utils as rp

import bioimage_pipeline_utils as rp


def create_simple_squares(T: int = 5, Y: int = 100, X: int = 100, 
                         square_size: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Create synthetic video with simple square patterns for drift testing.
    
    This generator creates clean, sharp squares that should yield 100% perfect
    drift correction results. Ideal for validating algorithm accuracy.
    
    Args:
        T: Number of time frames
        Y: Image height  
        X: Image width
        square_size: Size of squares in pixels
        
    Returns:
        Tuple of (drifted_video, ground_truth_video, applied_shifts)
        - drifted_video: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_video: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of (dx, dy) shifts applied to each frame
    """
    video = np.zeros((T, 1, 1, Y, X), dtype=np.uint16)
    ground_truth = np.zeros((T, 1, 1, Y, X), dtype=np.uint16)
    
    # Define small, controlled shifts to avoid boundary issues
    true_shifts = [
        (0, 0),    # Frame 0: no shift (reference)
        (1, 2),    # Frame 1: shift right 1, down 2
        (2, -1),   # Frame 2: shift right 2, up 1
        (-1, 1),   # Frame 3: shift left 1, down 1
        (1, -1)    # Frame 4: shift right 1, up 1
    ]
    
    # Extend or truncate shifts to match T
    if T > len(true_shifts):
        # Repeat pattern for longer sequences
        true_shifts.extend([(np.random.randint(-2, 3), np.random.randint(-2, 3)) 
                           for _ in range(T - len(true_shifts))])
    else:
        true_shifts = true_shifts[:T]
    
    print(f"Creating simple squares video with {T} frames:")
    for t, (dx, dy) in enumerate(true_shifts):
        print(f"  Frame {t}: dx={dx}, dy={dy}")
    
    # Base pattern positions (centered squares with different intensities)
    base_y, base_x = Y//2, X//2
    
    for t in range(T):
        dx, dy = true_shifts[t]
        
        # Create ground truth (all frames identical to frame 0 - no drift)
        gt_dx, gt_dy = true_shifts[0]  # Frame 0 position (reference)
        
        # Square 1: top-left quadrant
        y1 = base_y//2 + dy
        x1 = base_x//2 + dx
        if 0 <= y1 <= Y-square_size and 0 <= x1 <= X-square_size:
            video[t, 0, 0, y1:y1+square_size, x1:x1+square_size] = 1000
        
        # Ground truth square 1 at reference position
        gt_y1 = base_y//2 + gt_dy
        gt_x1 = base_x//2 + gt_dx  
        if 0 <= gt_y1 <= Y-square_size and 0 <= gt_x1 <= X-square_size:
            ground_truth[t, 0, 0, gt_y1:gt_y1+square_size, gt_x1:gt_x1+square_size] = 1000
        
        # Square 2: bottom-right quadrant  
        y2 = base_y + base_y//4 + dy
        x2 = base_x + base_x//4 + dx
        if 0 <= y2 <= Y-square_size and 0 <= x2 <= X-square_size:
            video[t, 0, 0, y2:y2+square_size, x2:x2+square_size] = 2000
        
        # Ground truth square 2 at reference position
        gt_y2 = base_y + base_y//4 + gt_dy
        gt_x2 = base_x + base_x//4 + gt_dx
        if 0 <= gt_y2 <= Y-square_size and 0 <= gt_x2 <= X-square_size:
            ground_truth[t, 0, 0, gt_y2:gt_y2+square_size, gt_x2:gt_x2+square_size] = 2000
        
        # Square 3: top-right quadrant (different intensity)
        y3 = base_y//4 + dy
        x3 = base_x + base_x//3 + dx
        if 0 <= y3 <= Y-square_size and 0 <= x3 <= X-square_size:
            video[t, 0, 0, y3:y3+square_size, x3:x3+square_size] = 1500
        
        # Ground truth square 3 at reference position  
        gt_y3 = base_y//4 + gt_dy
        gt_x3 = base_x + base_x//3 + gt_dx
        if 0 <= gt_y3 <= Y-square_size and 0 <= gt_x3 <= X-square_size:
            ground_truth[t, 0, 0, gt_y3:gt_y3+square_size, gt_x3:gt_x3+square_size] = 1500
    
    return video, ground_truth, true_shifts


def create_simple_squares_3d(T: int = 5, Z: int = 8, Y: int = 100, X: int = 100, 
                            square_size: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """
    Create 3D synthetic video with simple square patterns for drift testing.
    
    This generator creates clean, sharp squares in 3D that should yield 100% perfect
    drift correction results. Ideal for validating 3D algorithm accuracy.
    
    Args:
        T: Number of time frames
        Z: Number of Z slices
        Y: Image height  
        X: Image width
        square_size: Size of squares in pixels
        
    Returns:
        Tuple of (drifted_video, ground_truth_video, applied_shifts)
        - drifted_video: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_video: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of (dx, dy, dz) shifts applied to each frame
    """
    video = np.zeros((T, 1, Z, Y, X), dtype=np.uint16)
    ground_truth = np.zeros((T, 1, Z, Y, X), dtype=np.uint16)
    
    # Define small, controlled 3D shifts to avoid boundary issues
    true_shifts = [
        (0, 0, 0),     # Frame 0: no shift (reference)
        (1, 2, 0),     # Frame 1: shift right 1, down 2, no Z
        (2, -1, 1),    # Frame 2: shift right 2, up 1, Z forward 1
        (-1, 1, -1),   # Frame 3: shift left 1, down 1, Z back 1
        (1, -1, 0)     # Frame 4: shift right 1, up 1, no Z
    ]
    
    # Extend or truncate shifts to match T
    if T > len(true_shifts):
        true_shifts.extend([(np.random.randint(-2, 3), np.random.randint(-2, 3), np.random.randint(-1, 2)) 
                           for _ in range(T - len(true_shifts))])
    else:
        true_shifts = true_shifts[:T]
    
    print(f"Creating 3D simple squares video with {T} frames, {Z} Z-slices:")
    for t, (dx, dy, dz) in enumerate(true_shifts):
        print(f"  Frame {t}: dx={dx}, dy={dy}, dz={dz}")
    
    # Base pattern positions (centered squares with different intensities)
    base_y, base_x, base_z = Y//2, X//2, Z//2
    
    for t in range(T):
        dx, dy, dz = true_shifts[t]
        
        # Create ground truth (all frames identical to frame 0 - no drift)
        gt_dx, gt_dy, gt_dz = true_shifts[0]  # Frame 0 position (reference)
        
        # Create 3D squares (cubes) in multiple Z-slices
        for z_offset in [-1, 0, 1]:  # Create cubes spanning 3 Z slices
            
            # Cube 1: top-left quadrant
            z1 = base_z + z_offset + dz
            y1 = base_y//2 + dy
            x1 = base_x//2 + dx
            if (0 <= z1 < Z and 0 <= y1 <= Y-square_size and 0 <= x1 <= X-square_size):
                video[t, 0, z1, y1:y1+square_size, x1:x1+square_size] = 1000
            
            # Ground truth cube 1 at reference position
            gt_z1 = base_z + z_offset + gt_dz
            gt_y1 = base_y//2 + gt_dy
            gt_x1 = base_x//2 + gt_dx  
            if (0 <= gt_z1 < Z and 0 <= gt_y1 <= Y-square_size and 0 <= gt_x1 <= X-square_size):
                ground_truth[t, 0, gt_z1, gt_y1:gt_y1+square_size, gt_x1:gt_x1+square_size] = 1000
            
            # Cube 2: bottom-right quadrant  
            z2 = base_z + z_offset + dz
            y2 = base_y + base_y//4 + dy
            x2 = base_x + base_x//4 + dx
            if (0 <= z2 < Z and 0 <= y2 <= Y-square_size and 0 <= x2 <= X-square_size):
                video[t, 0, z2, y2:y2+square_size, x2:x2+square_size] = 2000
            
            # Ground truth cube 2 at reference position
            gt_z2 = base_z + z_offset + gt_dz
            gt_y2 = base_y + base_y//4 + gt_dy
            gt_x2 = base_x + base_x//4 + gt_dx
            if (0 <= gt_z2 < Z and 0 <= gt_y2 <= Y-square_size and 0 <= gt_x2 <= X-square_size):
                ground_truth[t, 0, gt_z2, gt_y2:gt_y2+square_size, gt_x2:gt_x2+square_size] = 2000
            
            # Cube 3: top-right quadrant (different intensity)
            z3 = base_z + z_offset + dz
            y3 = base_y//4 + dy
            x3 = base_x + base_x//3 + dx
            if (0 <= z3 < Z and 0 <= y3 <= Y-square_size and 0 <= x3 <= X-square_size):
                video[t, 0, z3, y3:y3+square_size, x3:x3+square_size] = 1500
            
            # Ground truth cube 3 at reference position  
            gt_z3 = base_z + z_offset + gt_dz
            gt_y3 = base_y//4 + gt_dy
            gt_x3 = base_x + base_x//3 + gt_dx
            if (0 <= gt_z3 < Z and 0 <= gt_y3 <= Y-square_size and 0 <= gt_x3 <= X-square_size):
                ground_truth[t, 0, gt_z3, gt_y3:gt_y3+square_size, gt_x3:gt_x3+square_size] = 1500
    
    return video, ground_truth, true_shifts


def create_cell_like_pattern(height: int = 256, width: int = 256, 
                           num_cells: int = 8, noise_level: float = 5.0) -> np.ndarray:
    """
    Create a synthetic pattern resembling cells using Gaussian blobs.
    
    Args:
        height: Pattern height in pixels
        width: Pattern width in pixels  
        num_cells: Number of cell-like blobs to create
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        2D numpy array with cell-like pattern
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Generate random cell positions and sizes
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_cells):
        # Random position (avoid edges)
        margin = 30
        cx = np.random.randint(margin, width - margin)
        cy = np.random.randint(margin, height - margin)
        
        # Random size and intensity
        size = np.random.uniform(8, 20)
        intensity = np.random.uniform(80, 200)
        
        # Create Gaussian blob
        blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * size**2))
        pattern += blob * intensity
    
    # Add some background structures
    # Large background gradient
    gradient_x = np.linspace(0, 20, width)
    gradient_y = np.linspace(0, 15, height)
    gradient = np.outer(gradient_y, gradient_x)
    pattern += gradient
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, pattern.shape)
        pattern += noise
    
    # Ensure positive values
    pattern = np.maximum(pattern, 0)
    
    return pattern.astype(np.uint16)


def create_cell_like_pattern_3d(depth: int = 16, height: int = 128, width: int = 128, 
                               num_cells: int = 6, noise_level: float = 5.0) -> np.ndarray:
    """
    Create a 3D synthetic pattern resembling cells using 3D Gaussian blobs.
    
    Args:
        depth: Pattern depth in pixels (Z dimension)
        height: Pattern height in pixels (Y dimension)
        width: Pattern width in pixels (X dimension)
        num_cells: Number of cell-like blobs to create
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        3D numpy array with cell-like pattern (Z, Y, X)
    """
    pattern = np.zeros((depth, height, width), dtype=np.float32)
    
    # Create coordinate grids
    z, y, x = np.ogrid[:depth, :height, :width]
    
    # Generate random cell positions and sizes
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_cells):
        # Random position (avoid edges)
        margin_z = max(2, depth // 8)
        margin_xy = max(20, min(height, width) // 8)
        
        cx = np.random.randint(margin_xy, width - margin_xy)
        cy = np.random.randint(margin_xy, height - margin_xy)
        cz = np.random.randint(margin_z, depth - margin_z)
        
        # Random size and intensity (different for each dimension)
        size_xy = np.random.uniform(6, 15)  # XY size
        size_z = np.random.uniform(2, 6)    # Z size (typically smaller)
        intensity = np.random.uniform(80, 200)
        
        # Create 3D Gaussian blob
        blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * size_xy**2) - 
                      (z - cz)**2 / (2 * size_z**2))
        pattern += blob * intensity
    
    # Add some background structures
    # 3D gradient
    gradient_x = np.linspace(0, 15, width)
    gradient_y = np.linspace(0, 10, height)
    gradient_z = np.linspace(0, 8, depth)
    
    # Create 3D gradient
    grad_3d = np.zeros((depth, height, width))
    for z_idx in range(depth):
        grad_3d[z_idx] = np.outer(gradient_y, gradient_x) + gradient_z[z_idx]
    
    pattern += grad_3d
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, pattern.shape)
        pattern += noise
    
    # Ensure positive values
    pattern = np.maximum(pattern, 0)
    
    return pattern.astype(np.uint16)


def apply_shift_to_frame(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """
    Apply a known shift to a frame to create synthetic drift.
    
    Args:
        frame: 2D numpy array to shift
        shift_x: Horizontal shift in pixels (positive = right)
        shift_y: Vertical shift in pixels (positive = down)
        
    Returns:
        Shifted frame with same dtype as input
    """
    try:
        from scipy.ndimage import shift as scipy_shift
        # Apply shift using scipy (this creates the "drifted" version)
        shifted = scipy_shift(frame.astype(np.float32), [shift_y, shift_x], 
                            order=1, prefilter=False)
        return shifted.astype(frame.dtype)
    except ImportError:
        # Fallback to simple integer shifting if scipy not available
        print("Warning: scipy not available, using simple integer shift")
        shift_y_int = int(round(shift_y))
        shift_x_int = int(round(shift_x))
        
        shifted = np.zeros_like(frame)
        
        # Calculate valid regions
        src_y_start = max(0, -shift_y_int)
        src_y_end = min(frame.shape[0], frame.shape[0] - shift_y_int)
        src_x_start = max(0, -shift_x_int)
        src_x_end = min(frame.shape[1], frame.shape[1] - shift_x_int)
        
        dst_y_start = max(0, shift_y_int)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_x_int)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        # Copy shifted region
        if src_y_end > src_y_start and src_x_end > src_x_start:
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                frame[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return shifted


def apply_shift_to_volume(volume: np.ndarray, shift_x: float, shift_y: float, shift_z: float = 0.0) -> np.ndarray:
    """
    Apply a known 3D shift to a volume to create synthetic drift.
    
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
        # Apply 3D shift using scipy (this creates the "drifted" version)
        shifted = scipy_shift(volume.astype(np.float32), [shift_z, shift_y, shift_x], 
                            order=1, prefilter=False)
        return shifted.astype(volume.dtype)
    except ImportError:
        # Fallback to simple integer shifting if scipy not available
        print("Warning: scipy not available, using simple integer 3D shift")
        shift_z_int = int(round(shift_z))
        shift_y_int = int(round(shift_y))
        shift_x_int = int(round(shift_x))
        
        shifted = np.zeros_like(volume)
        Z, Y, X = volume.shape
        
        # Calculate valid regions
        src_z_start = max(0, -shift_z_int)
        src_z_end = min(Z, Z - shift_z_int)
        src_y_start = max(0, -shift_y_int)
        src_y_end = min(Y, Y - shift_y_int)
        src_x_start = max(0, -shift_x_int)
        src_x_end = min(X, X - shift_x_int)
        
        dst_z_start = max(0, shift_z_int)
        dst_z_end = dst_z_start + (src_z_end - src_z_start)
        dst_y_start = max(0, shift_y_int)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_x_int)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        # Copy shifted region
        if (src_z_end > src_z_start and src_y_end > src_y_start and src_x_end > src_x_start):
            shifted[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                volume[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        
        return shifted


def create_cell_like_video(T: int = 8, Y: int = 256, X: int = 256, 
                          num_cells: int = 12, noise_level: float = 8.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Create synthetic video with cell-like patterns for realistic drift testing.
    
    This generator creates more realistic cell-like patterns with noise, suitable
    for testing algorithm robustness under more challenging conditions.
    
    Args:
        T: Number of time frames
        Y: Image height  
        X: Image width
        num_cells: Number of cell-like structures per frame
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Tuple of (drifted_video, ground_truth_video, applied_shifts)
        - drifted_video: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_video: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of (dx, dy) shifts applied to each frame
    """
    # Create base cell pattern
    base_pattern = create_cell_like_pattern(Y, X, num_cells, noise_level)
    
    # Define realistic drift shifts (larger range for bigger images)
    max_shift = min(Y, X) // 20  # Adaptive to image size
    applied_shifts = [(0.0, 0.0)]  # Frame 0: no shift (reference)
    
    # Generate cumulative drift for remaining frames
    np.random.seed(123)  # For reproducible results
    for t in range(1, T):
        # Small random walk drift
        dx = np.random.uniform(-2.5, 2.5)
        dy = np.random.uniform(-2.5, 2.5)
        
        # Accumulate drift (but keep within bounds)
        prev_x, prev_y = applied_shifts[t-1]
        new_x = np.clip(prev_x + dx, -max_shift, max_shift)
        new_y = np.clip(prev_y + dy, -max_shift, max_shift)
        
        applied_shifts.append((new_x, new_y))
    
    print(f"Creating cell-like video with {T} frames:")
    for t, (dx, dy) in enumerate(applied_shifts):
        print(f"  Frame {t}: dx={dx:.2f}, dy={dy:.2f}")
    
    # Initialize video arrays
    drifted_video = np.zeros((T, 1, 1, Y, X), dtype=np.uint16)
    ground_truth_video = np.zeros((T, 1, 1, Y, X), dtype=np.uint16)
    
    for t in range(T):
        shift_x, shift_y = applied_shifts[t]
        
        # Ground truth: no shift applied (same base pattern for all frames)
        ground_truth_video[t, 0, 0] = base_pattern
        
        # Drifted version: apply the shift  
        drifted_video[t, 0, 0] = apply_shift_to_frame(base_pattern, shift_x, shift_y)
    
    return drifted_video, ground_truth_video, applied_shifts


def create_cell_like_video_3d(T: int = 8, Z: int = 16, Y: int = 128, X: int = 128, 
                             num_cells: int = 8, noise_level: float = 6.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float]]]:
    """
    Create 3D synthetic video with cell-like patterns for realistic drift testing.
    
    This generator creates more realistic 3D cell-like patterns with noise, suitable
    for testing algorithm robustness under challenging 3D conditions.
    
    Args:
        T: Number of time frames
        Z: Number of Z slices
        Y: Image height  
        X: Image width
        num_cells: Number of cell-like structures per frame
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Tuple of (drifted_video, ground_truth_video, applied_shifts)
        - drifted_video: 5D array (T,C,Z,Y,X) with intentional drift
        - ground_truth_video: 5D array (T,C,Z,Y,X) reference without drift  
        - applied_shifts: List of (dx, dy, dz) shifts applied to each frame
    """
    # Create base 3D cell pattern
    base_pattern = create_cell_like_pattern_3d(Z, Y, X, num_cells, noise_level)
    
    # Define realistic 3D drift shifts (adaptive to image size)
    max_shift_xy = min(Y, X) // 20
    max_shift_z = Z // 10  # Smaller Z shifts typically
    applied_shifts = [(0.0, 0.0, 0.0)]  # Frame 0: no shift (reference)
    
    # Generate cumulative drift for remaining frames
    np.random.seed(456)  # Different seed from 2D version
    for t in range(1, T):
        # Small random walk drift in 3D
        dx = np.random.uniform(-2.0, 2.0)
        dy = np.random.uniform(-2.0, 2.0)  
        dz = np.random.uniform(-1.0, 1.0)  # Smaller Z drift
        
        # Accumulate drift (but keep within bounds)
        prev_x, prev_y, prev_z = applied_shifts[t-1]
        new_x = np.clip(prev_x + dx, -max_shift_xy, max_shift_xy)
        new_y = np.clip(prev_y + dy, -max_shift_xy, max_shift_xy)
        new_z = np.clip(prev_z + dz, -max_shift_z, max_shift_z)
        
        applied_shifts.append((new_x, new_y, new_z))
    
    print(f"Creating 3D cell-like video with {T} frames, {Z} Z-slices:")
    for t, (dx, dy, dz) in enumerate(applied_shifts):
        print(f"  Frame {t}: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
    
    # Initialize video arrays
    drifted_video = np.zeros((T, 1, Z, Y, X), dtype=np.uint16)
    ground_truth_video = np.zeros((T, 1, Z, Y, X), dtype=np.uint16)
    
    for t in range(T):
        shift_x, shift_y, shift_z = applied_shifts[t]
        
        # Ground truth: no shift applied (same base pattern for all frames)
        ground_truth_video[t, 0] = base_pattern
        
        # Drifted version: apply the 3D shift  
        drifted_video[t, 0] = apply_shift_to_volume(base_pattern, shift_x, shift_y, shift_z)
    
    return drifted_video, ground_truth_video, applied_shifts


def save_synthetic_datasets(drifted_video: np.ndarray, ground_truth_video: np.ndarray, 
                          applied_shifts: List[Tuple], metadata: Dict[str, Any], 
                          output_dir: str, dataset_name: str = "synthetic") -> None:
    """
    Save synthetic datasets and metadata to disk.
    
    Args:
        drifted_video: Video with applied drift
        ground_truth_video: Reference video without drift
        applied_shifts: List of shifts applied
        metadata: Dictionary with dataset metadata
        output_dir: Directory to save files
        dataset_name: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save videos
    drifted_path = os.path.join(output_dir, f"{dataset_name}.tif")
    ground_truth_path = os.path.join(output_dir, f"{dataset_name}_ground_truth.tif")
    
    print(f"Saving drifted video to: {drifted_path}")
    rp.save_tczyx_image(drifted_video, drifted_path)
    
    print(f"Saving ground truth to: {ground_truth_path}")
    rp.save_tczyx_image(ground_truth_video, ground_truth_path)
    
    # Save metadata
    full_metadata = {
        'dataset_info': {
            'name': dataset_name,
            'creation_date': str(np.datetime64('now')),
            'dimensions': {
                'T': int(drifted_video.shape[0]),
                'C': int(drifted_video.shape[1]), 
                'Z': int(drifted_video.shape[2]),
                'Y': int(drifted_video.shape[3]),
                'X': int(drifted_video.shape[4])
            }
        },
        'drift_info': {
            'applied_shifts': applied_shifts,
            'max_drift_magnitude': float(np.max([np.sqrt(sum(coord**2 for coord in shift)) for shift in applied_shifts]))
        },
        'pattern_info': metadata
    }
    
    metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata.yaml")
    import yaml
    with open(metadata_path, 'w') as f:
        yaml.dump(full_metadata, f, default_flow_style=False, indent=2)
    
    print(f"Saved metadata to: {metadata_path}")
    print(f"Dataset '{dataset_name}' saved to: {output_dir}")

def create_drift_image_from_template(input_file: str, output_file: str, num_timepoints: int = 10, max_shift: float = 20.0) -> np.ndarray:
    """
    Create a synthetic drift image by taking the first timepoint from an input file
    and applying known shifts to simulate drift over time.
    
    Args:
        input_file (str): Path to input image file (any format supported by bioimage_pipeline_utils)
        output_file (str): Path to save synthetic drift image as TIFF
        num_timepoints (int): Number of timepoints to generate (default: 10)
        max_shift (float): Maximum shift in pixels (default: 20.0)
        
    Returns:
        np.ndarray: Array of applied shifts for each timepoint (T, 3) in ZYX order
    """
    logger.info(f"Creating synthetic drift image from {input_file}")
    
    # Load the input image
    img = rp.load_tczyx_image(input_file)
    input_data = img.data.astype(np.float32)
    T, C, Z, Y, X = input_data.shape
    
    logger.info(f"Input image shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    # Use the first timepoint as the base image for all synthetic timepoints
    base_timepoint = input_data[0:1]  # Keep as (1, C, Z, Y, X)
    
    # Generate known shifts for each timepoint
    # Create progressive drift pattern that's realistic
    applied_shifts = np.zeros((num_timepoints, 3), dtype=np.float32)  # ZYX order
    
    # Generate cumulative drift pattern (more realistic than random jumps)
    np.random.seed(42)  # For reproducible results
    
    for t in range(1, num_timepoints):
        # Generate small random drift increments
        # For single-slice images (Z=1), don't generate Z drift
        if Z > 1:
            dz = np.random.normal(0, max_shift / 20)  # Smaller Z drift (typical of microscopy)
        else:
            dz = 0.0  # No Z drift for single-slice images
            
        dy = np.random.normal(0, max_shift / 10)  # Y drift
        dx = np.random.normal(0, max_shift / 10)  # X drift
        
        # Cumulative drift (add to previous position)
        applied_shifts[t] = applied_shifts[t-1] + np.array([dz, dy, dx])
    
    # Limit maximum cumulative shift
    for i in range(3):
        applied_shifts[:, i] = np.clip(applied_shifts[:, i], -max_shift, max_shift)
    
    logger.info(f"Generated shifts ranging from {applied_shifts.min():.2f} to {applied_shifts.max():.2f} pixels")
    
    # Create synthetic drift stack
    synthetic_stack = np.zeros((num_timepoints, C, Z, Y, X), dtype=np.float32)
    
    # Track the actually applied shifts (after rounding to integers for np.roll)
    actually_applied_shifts = np.zeros_like(applied_shifts)
    
    for t in range(num_timepoints):
        dz, dy, dx = applied_shifts[t]
        
        # Round shifts to integers for np.roll application
        dz_int, dy_int, dx_int = int(round(dz)), int(round(dy)), int(round(dx))
        actually_applied_shifts[t] = [dz_int, dy_int, dx_int]
        
        # Apply shifts to each channel
        for c in range(C):
            # Get the base volume for this channel
            base_volume = base_timepoint[0, c]  # Shape: (Z, Y, X)
            
            # Apply integer shifts using numpy roll (for simplicity and speed)
            # Note: This creates periodic boundary conditions
            shifted_volume = base_volume.copy()
            
            # Only apply Z shift if Z > 1
            if Z > 1 and dz_int != 0:
                shifted_volume = np.roll(shifted_volume, dz_int, axis=0)
            
            # Apply Y and X shifts
            if dy_int != 0:
                shifted_volume = np.roll(shifted_volume, dy_int, axis=1)  
            if dx_int != 0:
                shifted_volume = np.roll(shifted_volume, dx_int, axis=2)
            
            synthetic_stack[t, c] = shifted_volume
    
    # Save the synthetic drift image
    logger.info(f"Saving synthetic drift image to {output_file}")
    rp.save_tczyx_image(synthetic_stack, output_file)
    
    # Save the known shifts as JSON file next to the image
    shifts_file = os.path.splitext(output_file)[0] + "_known_shifts.json"
    shifts_data = {
        "description": "Known shifts applied to create synthetic drift image",
        "input_file": input_file,
        "output_file": output_file,
        "num_timepoints": num_timepoints,
        "max_shift": max_shift,
        "random_seed": 42,
        "shifts_format": "ZYX order (timepoint, [dz, dy, dx])",
        "generated_shifts": applied_shifts.tolist(),  # Original fractional shifts generated
        "actually_applied_shifts": actually_applied_shifts.tolist(),  # Integer shifts actually applied
        "shifts": actually_applied_shifts.tolist()  # For compatibility, use actually applied shifts
    }
    
    with open(shifts_file, 'w') as f:
        json.dump(shifts_data, f, indent=2)
    
    logger.info(f"Saved known shifts to {shifts_file}")
    logger.info(f"Successfully created synthetic drift image with {num_timepoints} timepoints")
    logger.info(f"Generated shifts (ZYX): min={applied_shifts.min(axis=0)}, max={applied_shifts.max(axis=0)}")
    logger.info(f"Actually applied shifts (ZYX): min={actually_applied_shifts.min(axis=0)}, max={actually_applied_shifts.max(axis=0)}")
    
    return actually_applied_shifts  # Return the actually applied shifts for consistency




def main():
    """
    Demo function showing how to use both synthetic data generators.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic drift correction test data")
    parser.add_argument("--output-dir", default="E:\\Oyvind\\BIP-hub-test-data\\drift\\synthetic", 
                       help="Output directory for synthetic datasets")
    parser.add_argument("--squares-only", action="store_true",
                       help="Generate only simple squares dataset") 
    parser.add_argument("--cells-only", action="store_true",
                       help="Generate only cell-like dataset")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SYNTHETIC DRIFT CORRECTION DATA GENERATOR")
    print("=" * 70)
    
    if args.squares_only or (not args.cells_only):
        print("\n🟩 Generating Simple Squares Datasets (100% Accurate Reconstruction)")
        print("-" * 60)
        
        # Generate 2D simple squares dataset
        print("  📐 2D Simple Squares...")
        video_small_2d, gt_small_2d, shifts_small_2d = create_simple_squares(T=5, Y=100, X=100)
        
        metadata_small_2d = {
            'pattern_type': 'simple_squares_2d',
            'expected_accuracy': '100% perfect reconstruction',
            'square_size': 10,
            'intensities': [1000, 1500, 2000],
            'use_case': 'Algorithm validation and accuracy testing',
            'dimensions': '2D (single Z slice)'
        }
        
        save_synthetic_datasets(video_small_2d, gt_small_2d, shifts_small_2d, metadata_small_2d, 
                              args.output_dir, "squares_2d")
        
        # Generate 3D simple squares dataset
        print("  📦 3D Simple Squares...")
        video_small_3d, gt_small_3d, shifts_small_3d = create_simple_squares_3d(T=5, Z=8, Y=100, X=100)
        
        metadata_small_3d = {
            'pattern_type': 'simple_squares_3d',
            'expected_accuracy': '100% perfect reconstruction',
            'square_size': 10,
            'intensities': [1000, 1500, 2000],
            'use_case': 'Algorithm validation and accuracy testing',
            'dimensions': '3D (multiple Z slices with XYZ drift)'
        }
        
        save_synthetic_datasets(video_small_3d, gt_small_3d, shifts_small_3d, metadata_small_3d, 
                              args.output_dir, "squares_3d")
    
    if args.cells_only or (not args.squares_only):
        print("\n🟨 Generating Cell-like Datasets (Realistic Testing)")
        print("-" * 60)
        
        # Generate 2D cell-like dataset
        print("  🧬 2D Cell-like Patterns...")
        video_large_2d, gt_large_2d, shifts_large_2d = create_cell_like_video(T=8, Y=256, X=256)
        
        metadata_large_2d = {
            'pattern_type': 'cell_like_gaussian_blobs_2d',
            'expected_accuracy': 'Sub-pixel accuracy expected',
            'num_cells': 12,
            'noise_level': 8.0,
            'use_case': 'Robustness testing under realistic conditions',
            'dimensions': '2D (single Z slice)'
        }
        
        save_synthetic_datasets(video_large_2d, gt_large_2d, shifts_large_2d, metadata_large_2d,
                              args.output_dir, "cells_2d")
        
        # Generate 3D cell-like dataset
        print("  🧪 3D Cell-like Patterns...")
        video_large_3d, gt_large_3d, shifts_large_3d = create_cell_like_video_3d(T=6, Z=16, Y=128, X=128)
        
        metadata_large_3d = {
            'pattern_type': 'cell_like_gaussian_blobs_3d',
            'expected_accuracy': 'Sub-pixel accuracy expected',
            'num_cells': 8,
            'noise_level': 6.0,
            'use_case': 'Robustness testing under realistic 3D conditions',
            'dimensions': '3D (multiple Z slices with XYZ drift)'
        }
        
        save_synthetic_datasets(video_large_3d, gt_large_3d, shifts_large_3d, metadata_large_3d,
                              args.output_dir, "cells_3d")
    
    print("\n✅ Synthetic data generation complete!")
    print(f"\nFiles saved to: {args.output_dir}")
    
    datasets_generated = []
    if args.squares_only or (not args.cells_only):
        datasets_generated.extend(["squares_2d", "squares_3d"])
    if args.cells_only or (not args.squares_only):
        datasets_generated.extend(["cells_2d", "cells_3d"])
    
    print(f"\n📊 Generated datasets: {', '.join(datasets_generated)}")
    print("\nUse these datasets to:")
    print("  • Test 2D and 3D drift correction algorithms")
    print("  • Validate sub-pixel accuracy in both 2D and 3D") 
    print("  • Compare different correction methods")
    print("  • Benchmark performance improvements")
    print("  • Test XY-only vs XYZ drift correction")


if __name__ == "__main__":
    main()