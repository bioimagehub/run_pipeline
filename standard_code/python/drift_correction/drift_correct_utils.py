
import numpy as np
import json
import os   
import sys

import logging
logger = logging.getLogger(__name__)

# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import bioimage_pipeline_utils as rp


def drift_correction_score(image_path: str, channel: int = 0, reference: str = "first", # "first", "median", or "previous"
                           central_crop: float = 0.8, # fraction of XY kept (e.g., 0.8 keeps central 80%) 
                           z_project: str | None = "mean", # None, "mean", "max", or "median" 
                           ) -> float: 
    """
    Compute a single alignment score for a time series by measuring frame-to-frame similarity using z-normalized Pearson correlation, focusing on the central region.

    Args:
        image_path: Path to TCZYX image.
        channel: Channel index to analyze.
        reference: Which reference to compare each timepoint against.
            - "first": compare each frame to the first frame (default)
            - "median": compare each frame to the temporal median frame
            - "previous": compare each frame to the immediately previous frame
        central_crop: Fraction [0, 1] of the XY extent retained centered. 0.8 avoids edge artifacts.
        z_project: If not None, project along Z to 2D before scoring.
            - "mean", "max", "median" supported. None keeps 3D.

    Returns:
        A single scalar score in [-1, 1], where 1 indicates near-identical frames.
        If the series has <2 timepoints, returns NaN.
    """

    # Load image as T×C×Z×Y×X
    img = rp.load_tczyx_image(image_path)
    data = img.data
    if data.ndim != 5:
        raise ValueError(f"Expected TCZYX, got shape {data.shape}")
    T, C, Z, Y, X = data.shape
    if T < 2:
        return float("nan")
    if not (0 <= channel < C):
        raise IndexError(f"Channel {channel} out of range [0, {C-1}]")

    # Select channel
    series = data[:, channel, ...]  # shape T×Z×Y×X

    # Optional Z projection
    if z_project is not None:
        if z_project == "mean":
            series = series.mean(axis=1)      # T×Y×X
        elif z_project == "max":
            series = series.max(axis=1)       # T×Y×X
        elif z_project == "median":
            series = np.median(series, axis=1)
        else:
            raise ValueError(f"Unsupported z_project: {z_project}")

    # Central crop on XY
    def central_xy_crop(arr, frac):
        if frac >= 1.0:
            return arr
        # Support 2D (Y×X) or 3D (Z×Y×X)
        if arr.ndim == 3:
            Z_, Y_, X_ = arr.shape
            cy = int(Y_ * (1 - frac) / 2)
            cx = int(X_ * (1 - frac) / 2)
            return arr[:, cy:Y_-cy, cx:X_-cx]
        elif arr.ndim == 2:
            Y_, X_ = arr.shape
            cy = int(Y_ * (1 - frac) / 2)
            cx = int(X_ * (1 - frac) / 2)
            return arr[cy:Y_-cy, cx:X_-cx]
        else:
            raise ValueError("Unexpected array dimensionality for cropping.")

    series = np.array([central_xy_crop(frame, central_crop) for frame in series], dtype=np.float32)

    # Build reference(s)
    if reference == "first":
        ref = series[0]
        compare_pairs = [(ref, series[t]) for t in range(1, T)]
    elif reference == "median":
        ref = np.median(series, axis=0)
        compare_pairs = [(ref, series[t]) for t in range(T)]
    elif reference == "previous":
        compare_pairs = [(series[t-1], series[t]) for t in range(1, T)]
    else:
        raise ValueError(f"Unsupported reference: {reference}")

    # Fast Pearson correlation on flattened, z-normalized data
    def pearson_centered(x, y):
        x = x.ravel().astype(np.float64)
        y = y.ravel().astype(np.float64)
        x -= x.mean()
        y -= y.mean()
        # Avoid division by zero
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        if x_norm == 0 or y_norm == 0:
            return np.nan
        return float(np.dot(x, y) / (x_norm * y_norm))

    corrs = []
    for ref_frame, cur_frame in compare_pairs:
        corrs.append(pearson_centered(ref_frame, cur_frame))

    # Final score: mean correlation across comparisons
    score = float(np.nanmean(corrs)) if len(corrs) > 0 else float("nan")
    return score


# ==================== SYNTHETIC DATA GENERATION ====================
def create_synthetic_drift_image(input_file: str, output_file: str, num_timepoints: int = 10, max_shift: float = 20.0) -> np.ndarray:
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

