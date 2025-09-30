
import numpy as np
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

#TODO add gaussian blur functon here




# ==================== SYNTHETIC DATA GENERATION ====================
# see synthetic_data_generators.py for related functions