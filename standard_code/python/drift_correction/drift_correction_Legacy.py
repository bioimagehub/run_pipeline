"""
#TODO
This will be changed to a wrapper module for all drift correction algorithms located in `drift_correction` folder.


Modular drift correction library with standardized algorithm interfaces.

This module provides a clean, modular approach to XY translation drift correction with:
- Unified function interfaces across all algorithms
- Standardized input/output formats  
- Separation of correction estimation and application
- Library-based naming convention for clarity
- Fast CuPy-accelerated shift application for all methods
- **XY TRANSLATION ONLY**: No rotation, scaling, or other transformations

CORRECTION SCOPE: This library corrects sample movement in X and Y directions over time.
It does NOT correct rotation, scaling, shearing, or Z-drift. All methods are configured 
for pure translational drift correction only.

Supported algorithms:
- skimage: phase_cross_correlation (translation estimation)
- pystackreg: TurboReg/StackReg algorithms (translation mode only)  
- imreg_dft: DFT-based registration (translation component extraction)
- pyGPUreg: GPU-accelerated phase correlation (translation)
- MVRegFus: Multi-view registration (translation, when available)
- NanoPyx: Advanced drift correction with quality metrics (translation)

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

from typing import Optional, Tuple, Any, Dict, List, Union
import numpy as np
import logging
import time
import traceback

# Local helper used throughout the repo
import bioimage_pipeline_utils as rp

# Module logger
logger = logging.getLogger(__name__)

# ------------------------------
# Core utility functions
# ------------------------------
#
# KEY DESIGN: XY TRANSLATION DRIFT CORRECTION ONLY
# All algorithms are configured to estimate and apply ONLY XY translations.
# This corrects sample movement/drift in X and Y directions over time.
# NO rotation, scaling, shearing, or Z-drift correction is performed.
#
# All methods use the same fast CuPy apply_shift() for translation application:
# - Consistent performance regardless of algorithm choice
# - GPU acceleration when available, CPU fallback when not
# - Unified interface for all XY translation applications
# - Pure translation correction maintains biological accuracy
#

def phase_cross_correlation(a: np.ndarray, b: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """
    Pure phase correlation for estimating translation between two images.
    
    Renamed from phase_cross_correlation_cupy for standardization.
    Uses CuPy if available, falls back to NumPy/SciPy.
    
    Args:
        a: Reference image (2D)
        b: Moving image to register to 'a' (2D) 
        upsample_factor: Subpixel refinement factor
        
    Returns:
        tuple: (shift_yx, error, global_phase)
            - shift_yx: Translation to apply to 'b' to align with 'a' (skimage convention)
            - error: Registration error estimate (1.0 - correlation peak)
            - global_phase: Global phase difference (usually 0.0)
    """
    try:
        import cupy as cp
        return _phase_cross_correlation_cupy(a, b, upsample_factor)
    except ImportError:
        return _phase_cross_correlation_numpy(a, b, upsample_factor)


def _phase_cross_correlation_cupy(a: np.ndarray, b: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """CuPy-accelerated phase correlation implementation."""
    import cupy as cp
    
    a = cp.asarray(a, dtype=cp.float32)
    b = cp.asarray(b, dtype=cp.float32)
    H, W = a.shape

    # Normalize to reduce bias from intensity variations
    a_mean = cp.mean(a)
    b_mean = cp.mean(b)
    a_norm = a - a_mean
    b_norm = b - b_mean
    
    # Apply window function to reduce edge effects (critical for accuracy)
    if H > 32 and W > 32:  # Only for reasonably sized images
        wy = cp.hanning(H)[:, cp.newaxis]
        wx = cp.hanning(W)[cp.newaxis, :]
        window = wy * wx
        a_norm = a_norm * window
        b_norm = b_norm * window

    # Cross power spectrum with better normalization
    Fa = cp.fft.fft2(a_norm)
    Fb = cp.fft.fft2(b_norm)
    R = Fa * cp.conj(Fb)
    # More robust normalization
    R_abs = cp.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    R = R / R_abs
    r = cp.fft.ifft2(R).real

    # Coarse peak (wrap-aware)
    idx = int(cp.argmax(r))
    py = idx // W
    px = idx % W
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    if upsample_factor <= 1:
        return (cp.asnumpy(cp.asarray([float(py), float(px)])), 1.0 - float(r.max()), 0.0)

    # Enhanced subpixel estimation using 5x5 neighborhood for better accuracy
    def _improved_subpixel_fit(r_local, center_y, center_x, H, W):
        """Improved subpixel fitting using larger neighborhood and weighted fitting"""
        # Get 5x5 neighborhood around peak
        neighborhood = cp.zeros((5, 5), dtype=cp.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny = (center_y + dy + H) % H
                nx = (center_x + dx + W) % W
                neighborhood[dy+2, dx+2] = r_local[ny, nx]
        
        # Find the maximum in the neighborhood
        max_idx = cp.argmax(neighborhood)
        max_y, max_x = divmod(int(max_idx), 5)
        
        # If peak is at edge, fall back to simple parabolic fit
        if max_y == 0 or max_y == 4 or max_x == 0 or max_x == 4:
            # Simple parabolic fit
            y0, x0 = (center_y + H) % H, (center_x + W) % W
            y1, y2 = (y0 - 1) % H, (y0 + 1) % H
            x1, x2 = (x0 - 1) % W, (x0 + 1) % W
            
            cy = r_local[y0, x0]
            dy_off = 0.5 * (r_local[y1, x0] - r_local[y2, x0]) / (r_local[y1, x0] - 2*cy + r_local[y2, x0] + 1e-12)
            dx_off = 0.5 * (r_local[y0, x1] - r_local[y0, x2]) / (r_local[y0, x1] - 2*cy + r_local[y0, x2] + 1e-12)
            
            return float(center_y + dy_off), float(center_x + dx_off)
        
        # Use center of mass for subpixel estimation (more robust)
        total_weight = cp.sum(neighborhood)
        if total_weight > 1e-12:
            y_indices, x_indices = cp.meshgrid(cp.arange(5, dtype=cp.float32) - 2, 
                                               cp.arange(5, dtype=cp.float32) - 2, indexing='ij')
            cm_y = cp.sum(y_indices * neighborhood) / total_weight
            cm_x = cp.sum(x_indices * neighborhood) / total_weight
            return float(center_y + cm_y), float(center_x + cm_x)
        else:
            return float(center_y), float(center_x)
    
    dy_sub, dx_sub = _improved_subpixel_fit(r, py, px, H, W)
    
    # Better error estimate
    max_val = float(r.max())
    err = max(0.0, 1.0 - max_val)
    
    return (cp.asnumpy(cp.asarray([dy_sub, dx_sub])), err, 0.0)


def _phase_cross_correlation_numpy(a: np.ndarray, b: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """NumPy/SciPy fallback implementation."""
    try:
        from scipy import ndimage
        from skimage.registration import phase_cross_correlation as skimage_pcc
        # Use skimage implementation as fallback
        shift, error, phase_diff = skimage_pcc(a, b, upsample_factor=upsample_factor)
        return shift, error, phase_diff
    except ImportError:
        # Basic NumPy-only implementation
        logger.warning("SciPy/skimage not available, using basic NumPy phase correlation")
        
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        H, W = a.shape
        
        # Basic normalization
        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)
        
        # Cross power spectrum
        Fa = np.fft.fft2(a_norm)
        Fb = np.fft.fft2(b_norm)
        R = Fa * np.conj(Fb)
        R_abs = np.abs(R)
        R_abs[R_abs < 1e-12] = 1e-12
        R = R / R_abs
        r = np.fft.ifft2(R).real
        
        # Find peak
        idx = np.argmax(r)
        py = idx // W
        px = idx % W
        if py > H // 2: py -= H
        if px > W // 2: px -= W
        
        error = 1.0 - r.max()
        return np.array([float(py), float(px)]), error, 0.0

def extract_translation_from_matrix(tmat: np.ndarray) -> np.ndarray:
    """
    Extract translation components from a transformation matrix.
    
    Args:
        tmat: 3x3 transformation matrix (homogeneous coordinates)
        
    Returns:
        Translation vector [dy, dx]
    """
    if tmat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 transformation matrix, got shape {tmat.shape}")
    
    # Translation components are in the last column
    dy = tmat[1, 2]  # Y translation
    dx = tmat[0, 2]  # X translation
    
    return np.array([dy, dx], dtype=np.float32)

def apply_shift(image: np.ndarray, shift_yx: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """
    Apply translation shift to an image.
    
    Args:
        image: Input image (2D or higher)
        shift_yx: Translation vector [dy, dx]
        mode: Boundary handling mode (default 'constant' = zero-fill)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Shifted image with same shape and dtype as input
    """
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cndi
        return _apply_shift_cupy(image, shift_yx, mode, order)
    except ImportError:
        return _apply_shift_numpy(image, shift_yx, mode, order)

def _apply_shift_cupy(image: np.ndarray, shift_yx: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """CuPy-accelerated shift application."""
    import cupy as cp
    from cupyx.scipy import ndimage as cndi
    
    # Convert to CuPy array
    image_cp = cp.asarray(image, dtype=cp.float32)
    shift = cp.asarray(shift_yx, dtype=cp.float32)
    
    # Apply shift
    shifted_cp = cndi.shift(image_cp, shift=shift, order=order, mode=mode, cval=0.0, prefilter=True)
    shifted_np = cp.asnumpy(shifted_cp)
    
    # Convert back to original dtype
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        shifted_np = np.clip(shifted_np, info.min, info.max).astype(image.dtype)
    else:
        shifted_np = shifted_np.astype(image.dtype)
        
    return shifted_np

def _apply_shift_numpy(image: np.ndarray, shift_yx: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """NumPy/SciPy fallback shift application."""
    try:
        from scipy import ndimage
        
        shifted = ndimage.shift(image.astype(np.float32), shift=shift_yx, order=order, mode=mode, cval=0.0, prefilter=True)
        
        # Convert back to original dtype
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            shifted = np.clip(shifted, info.min, info.max).astype(image.dtype)
        else:
            shifted = shifted.astype(image.dtype)
            
        return shifted
    except ImportError:
        logger.error("SciPy not available for shift application")
        return image  # Return unchanged if no shift capability

# ------------------------------
# Algorithm-specific implementations
# ------------------------------

def _phase_corr_and_psr_2d(ref: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fast NumPy phase correlation between two 2D arrays plus PSR value.
    Returns (shift_yx, psr)."""
    a = ref.astype(np.float32)
    b = img.astype(np.float32)
    H, W = a.shape
    if H > 32 and W > 32:
        wy = np.hanning(H)[:, None]
        wx = np.hanning(W)[None, :]
        win = wy * wx
        a = a * win
        b = b * win
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R_abs = np.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    Rn = R / R_abs
    r = np.fft.ifft2(Rn).real
    idx = np.argmax(r)
    py = int(idx // W)
    px = int(idx % W)
    peak = float(r[py, px])
    if py > H // 2:
        py -= H
    if px > W // 2:
        px -= W
    # PSR
    mask = np.ones_like(r, dtype=bool)
    y0 = (py % H)
    x0 = (px % W)
    y1 = max(0, y0 - 2); y2 = min(H, y0 + 3)
    x1 = max(0, x0 - 2); x2 = min(W, x0 + 3)
    mask[y1:y2, x1:x2] = False
    side = r[mask]
    mu = float(side.mean()) if side.size > 0 else 0.0
    sigma = float(side.std(ddof=1)) if side.size > 1 else 1.0
    psr = (peak - mu) / (sigma + 1e-12)
    return np.array([float(py), float(px)], dtype=np.float32), psr


def greedy_psr_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    reference_frame: str = 'previous',  # 'previous' (sequential) or 'mean'
    step_px: int = 1,                   # integer pixel step for search
    max_iters: int = 10,                # max greedy steps per frame
    neighborhood: str = '8',            # '4' or '8' connectivity
    thumb_max_size: int = 256           # thumbnail size for scoring
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Greedy drift correction using PSR to evaluate 8-neighborhood candidate shifts per frame.
    Only integer pixel exploration on thumbnails for speed; final application uses subpixel shift.
    """
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    T, C, Z, Y, X = video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")

    # Prepare reference stack (max-Z projections)
    stack = np.max(video[:, drift_channel, :, :, :], axis=1).astype(np.float32)  # (T, Y, X)

    # Compute uniform integer scale for thumbnails across the stack
    scale = max(1, int(np.ceil(max(Y, X) / float(thumb_max_size))))
    # Create thumbnails using stride-based downsampling (fast, aliasing-safe for coarse search)
    thumbs = stack[:, ::scale, ::scale]

    # Reference per strategy
    if reference_frame == 'mean':
        ref_thumb = thumbs.mean(axis=0)
    elif reference_frame == 'previous':
        ref_thumb = None  # will be prior aligned frame
    else:
        try:
            idx = int(reference_frame)
            ref_thumb = thumbs[idx]
        except Exception:
            ref_thumb = thumbs.mean(axis=0)

    # Neighborhood moves (integer)
    if neighborhood == '4':
        moves = [(0, step_px), (0, -step_px), (step_px, 0), (-step_px, 0)]
    else:
        s = step_px
        moves = [(-s, -s), (-s, 0), (-s, s), (0, -s), (0, s), (s, -s), (s, 0), (s, s)]

    # Track cumulative subpixel shifts (float)
    shifts = np.zeros((T, 2), dtype=np.float32)

    def _roll_int(img2d: np.ndarray, dy: int, dx: int) -> np.ndarray:
        return np.roll(np.roll(img2d, dy, axis=0), dx, axis=1)

    # Greedy per frame relative to previous aligned (or global ref)
    for t in range(T):
        if t == 0:
            continue  # first frame as reference
        cur = thumbs[t]
        if reference_frame == 'previous':
            ref = thumbs[t-1]
        else:
            ref = ref_thumb
        # Safety guard for type-checkers and edge cases
        if ref is None:
            ref = thumbs[t-1] if t > 0 else cur
        # Start with no additional move
        best_score = _phase_corr_and_psr_2d(ref, cur)[1]
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            candidate_scores: List[Tuple[float, Tuple[int, int]]] = []
            for dy, dx in moves:
                cand = _roll_int(cur, dy, dx)
                score = _phase_corr_and_psr_2d(ref, cand)[1]
                candidate_scores.append((score, (dy, dx)))
            # Choose best move
            candidate_scores.sort(reverse=True, key=lambda x: x[0])
            top_score, (dy_best, dx_best) = candidate_scores[0]
            if top_score > best_score + 1e-3:  # small improvement threshold
                # Accept move on thumbnail and accumulate (integer to float shift)
                cur = _roll_int(cur, dy_best, dx_best)
                shifts[t, 0] += float(dy_best)
                shifts[t, 1] += float(dx_best)
                best_score = top_score
                improved = True

    # Apply cumulative shifts to full-resolution video (interpolated)
    corrected = np.zeros_like(video)
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                # Scale thumbnail-derived integer shifts back to full resolution and apply (zero-fill borders)
                scaled_shift = shifts[t] * float(scale)
                corrected[t, c, z] = apply_shift(video[t, c, z], scaled_shift, mode='constant', order=3)

    info = {
        'method': 'greedy_psr',
        'reference_frame': reference_frame,
        'step_px': step_px,
        'max_iters': max_iters,
        'neighborhood': neighborhood,
        'thumb_max_size': thumb_max_size,
        'thumb_scale': scale,
        'shifts': shifts,
    }
    return corrected, info

def skimage_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    reference_frame: str = 'mean',  # 'mean', 'first', 'previous', or frame index
    upsample_factor: int = 50,
    gaussian_filter: Optional[float] = None,  # Gaussian smoothing sigma (helps with noise)
    registration_strategy: str = 'to_mean',  # 'to_mean', 'sequential', 'running_mean'
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drift correction using scikit-image phase cross-correlation.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        reference_frame: Reference for registration ('mean', 'first', 'previous', or frame index)
        upsample_factor: Subpixel refinement factor (1-100, higher = more precise but slower)
        gaussian_filter: Gaussian smoothing sigma for noise reduction (None to disable)
        registration_strategy: 'to_mean', 'sequential', or 'running_mean'
        
    Returns:
        tuple: (corrected_video, shifts)
    """
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")
    
    # Create Z-max projections for drift estimation
    ref_stack = np.max(video[:, drift_channel, :, :, :], axis=1).astype(np.float32)
    
    # Apply Gaussian filtering if requested (helps with noisy data)
    if gaussian_filter is not None and gaussian_filter > 0:
        try:
            from scipy import ndimage
            for t in range(T):
                ref_stack[t] = ndimage.gaussian_filter(ref_stack[t], sigma=gaussian_filter)
            logger.info(f"Applied Gaussian filtering (sigma={gaussian_filter})")
        except ImportError:
            logger.warning("SciPy not available for Gaussian filtering, skipping")
    
    # Estimate shifts based on registration strategy
    shifts = np.zeros((T, 2), dtype=np.float32)
    
    if registration_strategy == 'sequential':
        # Register each frame to the previous frame (can accumulate errors but good for fast drift)
        logger.info("Using sequential registration strategy")
        for t in range(1, T):  # Skip first frame (reference)
            shift_yx, error, _ = phase_cross_correlation(ref_stack[t-1], ref_stack[t], upsample_factor)
            shifts[t] = shifts[t-1] + shift_yx  # Accumulate shifts
            
    elif registration_strategy == 'running_mean':
        # Use running mean as reference (adaptive to changing conditions)
        logger.info("Using running mean registration strategy")
        window_size = min(5, T // 2)  # Adaptive window size
        for t in range(T):
            # Create running mean reference
            start_idx = max(0, t - window_size//2)
            end_idx = min(T, t + window_size//2 + 1)
            if t == 0:
                reference = ref_stack[0]
            else:
                reference = np.mean(ref_stack[start_idx:end_idx], axis=0)
            
            shift_yx, error, _ = phase_cross_correlation(reference, ref_stack[t], upsample_factor)
            shifts[t] = shift_yx
            
    else:  # 'to_mean' strategy (default)
        logger.info("Using to-mean registration strategy")
        # Determine reference frame
        if reference_frame == 'mean':
            reference = np.mean(ref_stack, axis=0)
        elif reference_frame == 'first':
            reference = ref_stack[0]
        elif reference_frame == 'previous':
            # This doesn't make sense for to_mean strategy, fall back to mean
            logger.warning("'previous' reference_frame not compatible with 'to_mean' strategy, using 'mean'")
            reference = np.mean(ref_stack, axis=0)
        else:
            reference = ref_stack[int(reference_frame)]
        
        # Register all frames to the same reference
        for t in range(T):
            shift_yx, error, _ = phase_cross_correlation(reference, ref_stack[t], upsample_factor)
            shifts[t] = shift_yx
    
    # Apply correction to all channels and Z-slices
    corrected = np.zeros_like(video)
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                corrected[t, c, z] = apply_shift(video[t, c, z], shifts[t])
    
    return corrected, shifts


def pystackreg_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    transformation: str = 'translation',  # FORCED to 'translation' - only XY translations supported
    reference_frame: str = 'mean',
    **kwargs
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    XY translation drift correction using pyStackReg (TurboReg/StackReg algorithms).
    
    TRANSLATION ONLY: This function is hardcoded to use translation-only mode.
    Only XY sample drift/movement is corrected. No rotation, scaling, or other 
    transformations are estimated or applied.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        transformation: IGNORED - always uses 'translation' mode for XY drift only
        reference_frame: Reference for registration
        
    Returns:
        tuple: (corrected_video, {'transformation_matrices': tmats, 'translations': shifts})
    """
    try:
        from pystackreg import StackReg
    except ImportError:
        raise ImportError("pystackreg not available. Install with: pip install pystackreg")
    
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    
    # FORCE TRANSLATION-ONLY MODE: Only XY translations, no rotation/scaling
    if transformation != 'translation':
        logger.warning(f"pystackreg_drift_correction: transformation '{transformation}' ignored. "
                      f"Only 'translation' mode supported for XY drift correction.")
    
    # Initialize StackReg in translation-only mode
    sr = StackReg(StackReg.TRANSLATION)
    
    # Create Z-max projections
    ref_stack = np.max(video[:, drift_channel, :, :, :], axis=1)
    
    # Register stack
    tmats = sr.register_stack(ref_stack, reference=reference_frame, verbose=False, axis=0)
    
    # Extract translations from transformation matrices and apply using fast CuPy shift
    corrected = np.zeros_like(video)
    translations = np.zeros((T, 2), dtype=np.float32)
    
    for t in range(T):
        # Extract translation component from transformation matrix
        translations[t] = extract_translation_from_matrix(tmats[t])
        
        # Apply translation to all channels and Z-slices using fast CuPy method
        for c in range(C):
            for z in range(Z):
                corrected[t, c, z] = apply_shift(video[t, c, z], translations[t], mode='constant', order=3)
    
    return corrected, {'transformation_matrices': tmats, 'translations': translations}


def imreg_dft_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    reference_frame: str = 'mean',
    estimate_rotation: bool = False,  # IGNORED - only translation supported
    estimate_scale: bool = False,     # IGNORED - only translation supported
    **kwargs
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    XY translation drift correction using imreg_dft.
    
    TRANSLATION ONLY: This function only performs XY translation correction.
    Rotation and scale estimation parameters are ignored - only sample movement
    in X and Y directions over time is corrected.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        reference_frame: Reference for registration
        estimate_rotation: IGNORED - only XY translation supported
        estimate_scale: IGNORED - only XY translation supported
        
    Returns:
        tuple: (corrected_video, transformation_parameters)
    """
    try:
        import imreg_dft as ird
    except ImportError:
        raise ImportError("imreg_dft not available. Install with: pip install imreg_dft")
    
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    
    # Create Z-max projections
    ref_stack = np.max(video[:, drift_channel, :, :, :], axis=1).astype(np.float32)
    
    # Determine reference frame
    if reference_frame == 'mean':
        reference = np.mean(ref_stack, axis=0)
    elif reference_frame == 'first':
        reference = ref_stack[0]
    else:
        reference = ref_stack[int(reference_frame)]
    
    # Estimate transformations
    transformations = []
    corrected = np.zeros_like(video)
    
    for t in range(T):
        # FORCE TRANSLATION-ONLY: Always use translation mode for XY drift correction
        if estimate_rotation or estimate_scale:
            logger.warning(f"imreg_dft_drift_correction: rotation/scale estimation ignored. "
                          f"Only XY translation drift correction supported.")
        
        # Always use translation-only mode
        result = ird.translation(reference, ref_stack[t])
        
        transformations.append({
            'frame': t,
            'tvec': result['tvec'] if 'tvec' in result else None,
            'angle': result.get('angle', 0.0),
            'scale': result.get('scale', 1.0),
            'success': result.get('success', True)
        })
        
        # Extract translation component and apply using fast CuPy method
        if 'tvec' in result and result['tvec'] is not None:
            # Use only translation component for fast processing
            # Note: Rotation and scale are ignored to maintain speed - use pystackreg for full transformations
            translation = np.array([result['tvec'][1], result['tvec'][0]], dtype=np.float32)  # [dy, dx]
            
            # Translation-only correction (no rotation/scale warning needed since forced above)
            
            for c in range(C):
                for z in range(Z):
                    corrected[t, c, z] = apply_shift(video[t, c, z], translation, mode='constant', order=3)
        else:
            # No transformation - copy original
            for c in range(C):
                for z in range(Z):
                    corrected[t, c, z] = video[t, c, z]
    
    return corrected, transformations


def pygpureg_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    reference_frame: str = 'mean',
    gpu_memory_limit: float = 0.8,  # Use up to 80% of GPU memory
    batch_size: int = 0,  # 0 = auto-detect optimal batch size
    correlation_threshold: float = 0.1,  # Minimum correlation for valid registration
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhanced GPU-accelerated drift correction using pyGPUreg with memory management.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        reference_frame: Reference for registration ('mean', 'first', or frame index)
        gpu_memory_limit: Fraction of GPU memory to use (0.1-0.9)
        batch_size: Number of frames to process simultaneously (0 = auto)
        correlation_threshold: Minimum correlation for accepting registration
        
    Returns:
        tuple: (corrected_video, registration_info)
    """
    logger.info("Initializing pyGPUreg-based drift correction")
    
    try:
        import pyGPUreg as reg  # type: ignore
        import cupy as cp  # type: ignore
        gpu_available = True
        logger.info("pyGPUreg and CuPy available, using GPU acceleration")
    except ImportError as e:
        logger.warning(f"pyGPUreg/CuPy not available ({e}). Falling back to skimage method.")
        corrected, shifts = skimage_drift_correction(video, drift_channel, reference_frame, **kwargs)
        return corrected, {'method': 'skimage_fallback', 'shifts': shifts}
    
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")
    
    # GPU memory management
    mempool = None
    try:
        mempool = cp.get_default_memory_pool()
        gpu_mem_total = cp.cuda.Device().mem_info[1]  # Total GPU memory
        gpu_mem_available = int(gpu_mem_total * gpu_memory_limit)
        logger.info(f"GPU memory: {gpu_mem_total // (1024**3):.1f} GB total, "
                   f"using {gpu_mem_available // (1024**3):.1f} GB ({gpu_memory_limit:.1%})")
    except Exception as e:
        logger.warning(f"GPU memory detection failed: {e}")
        gpu_mem_available = 2 * 1024**3  # Assume 2GB as fallback
    
    # Create reference stack (max projection across Z for each timepoint)
    logger.info("Creating reference stack (max projection)")
    ref_stack = np.max(video[:, drift_channel, :, :, :], axis=1)  # (T, Y, X)
    
    # Determine reference frame
    if reference_frame == 'mean':
        template = ref_stack.mean(axis=0).astype(np.float32)
        logger.info("Using mean frame as reference")
    elif reference_frame == 'first':
        template = ref_stack[0].astype(np.float32)
        logger.info("Using first frame as reference")
    else:
        try:
            ref_idx = int(reference_frame)
            if 0 <= ref_idx < T:
                template = ref_stack[ref_idx].astype(np.float32)
                logger.info(f"Using frame {ref_idx} as reference")
            else:
                logger.warning(f"Reference frame {ref_idx} out of range. Using mean instead.")
                template = ref_stack.mean(axis=0).astype(np.float32)
        except ValueError:
            logger.warning(f"Invalid reference frame '{reference_frame}'. Using mean instead.")
            template = ref_stack.mean(axis=0).astype(np.float32)
    
    # Optimize image size for GPU (power of 2 for FFT efficiency)
    def _next_pow2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    size = _next_pow2(max(Y, X))
    logger.info(f"Padded image size: {size}x{size} (original: {Y}x{X})")
    
    # Initialize pyGPUreg
    try:
        reg.init()
        logger.info("pyGPUreg initialized successfully")
    except Exception as e:
        logger.warning(f"pyGPUreg init failed: {e}. Falling back to skimage method.")
        corrected, shifts = skimage_drift_correction(video, drift_channel, reference_frame, **kwargs)
        return corrected, {'method': 'skimage_fallback', 'shifts': shifts, 'error': str(e)}
    
    # Auto-detect optimal batch size based on GPU memory
    if batch_size == 0:
        # Estimate memory per frame (padded image size + overhead)
        mem_per_frame = size * size * 4 * 4  # 4 bytes per float32, 4x overhead for FFT
        max_batch = max(1, gpu_mem_available // mem_per_frame)
        batch_size = min(max_batch, T, 16)  # Cap at reasonable batch size
        logger.info(f"Auto-detected batch size: {batch_size} frames")
    
    # Initialize arrays
    shifts = np.zeros((T, 2), dtype=np.float32)
    correlation_scores = np.zeros(T, dtype=np.float32)
    registration_success = np.zeros(T, dtype=bool)
    
    # Helper functions for padding/cropping
    def _pad_center(img2d: np.ndarray, size: int) -> Tuple[np.ndarray, int, int]:
        h, w = img2d.shape
        pad_img = np.zeros((size, size), dtype=np.float32)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        pad_img[y0:y0 + h, x0:x0 + w] = img2d.astype(np.float32)
        return pad_img, y0, x0
    
    # Prepare padded template
    template_pad, ty0, tx0 = _pad_center(template, size)
    
    # Process frames in batches
    logger.info(f"Processing {T} frames in batches of {batch_size}")
    
    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        batch_indices = range(batch_start, batch_end)
        
        try:
            # Clear GPU memory at start of each batch
            if gpu_available and mempool is not None:
                try:
                    mempool.free_all_blocks()
                except Exception:
                    pass
            
            # Process each frame in the current batch
            for i, t in enumerate(batch_indices):
                ref_img = ref_stack[t].astype(np.float32)
                ref_pad, ry0, rx0 = _pad_center(ref_img, size)
                
                try:
                    # Perform registration; API may return (img, shift) or just shift
                    reg_result = reg.register(template_pad, ref_pad, apply_shift=True)
                    registered_img = None
                    shift = None
                    if isinstance(reg_result, tuple) and len(reg_result) >= 2:
                        registered_img, shift = reg_result[0], reg_result[1]
                    else:
                        shift = reg_result

                    if isinstance(shift, (list, tuple, np.ndarray)) and len(shift) >= 2:
                        shifts[t, 0] = float(shift[0])
                        shifts[t, 1] = float(shift[1])
                        # Quality metric: correlation if registered_img available
                        try:
                            if isinstance(registered_img, np.ndarray):
                                corr = np.corrcoef(template_pad.ravel(), registered_img.ravel())[0, 1]
                                if not np.isnan(corr):
                                    correlation_scores[t] = max(0.0, float(corr))
                                    registration_success[t] = corr >= correlation_threshold
                                else:
                                    correlation_scores[t] = 0.0
                                    registration_success[t] = False
                            else:
                                correlation_scores[t] = 0.0
                                registration_success[t] = False
                        except Exception:
                            correlation_scores[t] = 0.0
                            registration_success[t] = False
                    else:
                        logger.warning(f"Invalid shift returned for frame {t}")
                        shifts[t] = np.array([0.0, 0.0])
                        correlation_scores[t] = 0.0
                        registration_success[t] = False
                        
                except Exception as e:
                    logger.warning(f"pyGPUreg registration failed at frame {t}: {e}")
                    shifts[t] = np.array([0.0, 0.0])
                    correlation_scores[t] = 0.0
                    registration_success[t] = False
            
        except Exception as e:
            logger.error(f"Batch processing failed for frames {batch_start}-{batch_end}: {e}")
            # Set failed frames to zero shift
            for t in batch_indices:
                shifts[t] = np.array([0.0, 0.0])
                correlation_scores[t] = 0.0
                registration_success[t] = False
    
    # Apply drift correction using optimized shift function
    logger.info("Applying drift correction to all channels and Z-slices")
    corrected_video = np.zeros_like(video)
    
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                corrected_video[t, c, z] = apply_shift(video[t, c, z], shifts[t])
    
    # Cleanup GPU resources
    try:
        if gpu_available and mempool is not None:
            try:
                mempool.free_all_blocks()
            except Exception:
                pass
        try:
            if hasattr(reg, 'cleanup') and callable(getattr(reg, 'cleanup', None)):
                reg.cleanup()
        except Exception:
            pass
        logger.info("GPU resources cleaned up")
    except Exception as e:
        logger.warning(f"GPU cleanup warning: {e}")
    
    # Calculate quality metrics
    success_rate = np.sum(registration_success) / T
    mean_correlation = np.mean(correlation_scores[registration_success])
    shift_magnitudes = np.sqrt(np.sum(shifts**2, axis=1))
    
    # Prepare comprehensive registration info
    registration_info = {
        'method': 'pygpureg_gpu',
        'shifts': shifts,
        'correlation_scores': correlation_scores,
        'registration_success': registration_success,
        'success_rate': success_rate,
        'mean_correlation': mean_correlation if success_rate > 0 else 0.0,
        'correlation_threshold': correlation_threshold,
        'reference_frame_type': reference_frame,
        'batch_size_used': batch_size,
        'gpu_memory_limit': gpu_memory_limit,
        'padded_size': size,
        'original_size': (Y, X),
        'frames_processed': T,
        'z_slices_processed': Z,
        'mean_shift_magnitude': np.mean(shift_magnitudes),
        'max_shift_magnitude': np.max(shift_magnitudes),
        'failed_frames': np.where(~registration_success)[0].tolist()
    }
    
    logger.info(f"pyGPUreg completed: {success_rate:.1%} success rate, "
               f"mean correlation: {mean_correlation:.3f}, "
               f"mean shift: {np.mean(shift_magnitudes):.2f}px")
    
    return corrected_video, registration_info


def mvregfus_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    fusion_strategy: str = 'weighted_average',  # 'weighted_average', 'median', 'best_quality'
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Multi-view registration and fusion using multiple algorithms with result fusion.
    
    This implements a fusion approach that runs multiple registration methods
    and combines their results for improved robustness and accuracy.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        fusion_strategy: How to combine results ('weighted_average', 'median', 'best_quality')
        
    Returns:
        tuple: (corrected_video, registration_info)
    """
    logger.info(f"Running multi-view registration fusion with strategy: {fusion_strategy}")
    
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")
    
    # Run multiple registration methods
    methods = []
    results = []
    errors = []
    
    # Method 1: skimage phase correlation
    try:
        logger.info("MVRegFus: Running skimage phase correlation")
        corrected1, shifts1 = skimage_drift_correction(
            video, drift_channel, 
            registration_strategy='to_mean', 
            upsample_factor=20,
            **kwargs
        )
        methods.append('skimage_phase')
        results.append((corrected1, shifts1))
        
        # Calculate registration quality (consistency metric)
        shift_magnitudes = np.sqrt(np.sum(shifts1**2, axis=1))
        consistency1 = 1.0 / (1.0 + np.std(shift_magnitudes))
        errors.append(1.0 - consistency1)
        
    except Exception as e:
        logger.warning(f"MVRegFus: skimage method failed: {e}")
        methods.append('skimage_phase')
        results.append(None)
        errors.append(1.0)
    
    # Method 2: pystackreg if available
    try:
        logger.info("MVRegFus: Running pystackreg")
        corrected2, result2 = pystackreg_drift_correction(
            video, drift_channel, 
            reference_frame='mean',
            **kwargs
        )
        shifts2 = result2['translations']
        methods.append('pystackreg')
        results.append((corrected2, shifts2))
        
        # Calculate consistency
        shift_magnitudes = np.sqrt(np.sum(shifts2**2, axis=1))
        consistency2 = 1.0 / (1.0 + np.std(shift_magnitudes))
        errors.append(1.0 - consistency2)
        
    except Exception as e:
        logger.warning(f"MVRegFus: pystackreg method failed: {e}")
        methods.append('pystackreg')
        results.append(None)
        errors.append(1.0)
    
    # Method 3: sequential skimage for comparison
    try:
        logger.info("MVRegFus: Running skimage sequential")
        corrected3, shifts3 = skimage_drift_correction(
            video, drift_channel,
            registration_strategy='sequential',
            upsample_factor=10,
            **kwargs
        )
        methods.append('skimage_sequential')
        results.append((corrected3, shifts3))
        
        # Calculate consistency
        shift_magnitudes = np.sqrt(np.sum(shifts3**2, axis=1))
        consistency3 = 1.0 / (1.0 + np.std(shift_magnitudes))
        errors.append(1.0 - consistency3)
        
    except Exception as e:
        logger.warning(f"MVRegFus: sequential method failed: {e}")
        methods.append('skimage_sequential')
        results.append(None)
        errors.append(1.0)
    
    # Filter out failed methods
    valid_results = [(methods[i], results[i], errors[i]) 
                    for i in range(len(results)) if results[i] is not None]
    
    if not valid_results:
        raise RuntimeError("All registration methods failed in MVRegFus")
    
    logger.info(f"MVRegFus: {len(valid_results)} methods succeeded out of {len(methods)}")
    
    # Fusion strategy
    weights_list = None  # for reporting later
    if fusion_strategy == 'best_quality':
        # Use the method with lowest error (highest consistency)
        best_idx = min(range(len(valid_results)), key=lambda i: valid_results[i][2])
        best_method, (corrected_final, shifts_final), error = valid_results[best_idx]
        logger.info(f"MVRegFus: Selected best quality method: {best_method} (error: {error:.4f})")
        
    elif fusion_strategy == 'median':
        # Use median of all shift estimates
        logger.info("MVRegFus: Computing median fusion of shifts")
        all_shifts = [result[1] for _, result, _ in valid_results if result is not None]
        shifts_stack = np.stack(all_shifts, axis=0)  # (n_methods, T, 2)
        shifts_final = np.median(shifts_stack, axis=0)  # (T, 2)
        
        # Apply median shifts to original video
        corrected_final = np.zeros_like(video)
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    corrected_final[t, c, z] = apply_shift(video[t, c, z], shifts_final[t])
        
    else:  # 'weighted_average'
        logger.info("MVRegFus: Computing weighted average fusion")
        # Weight by inverse error (higher consistency = higher weight)
        weights = np.array([1.0 / (error + 1e-6) for _, _, error in valid_results])
        weights /= np.sum(weights)  # Normalize
        weights_list = weights
        
        # Weighted average of shifts
        all_shifts = [result[1] for _, result, _ in valid_results if result is not None]
        shifts_stack = np.stack(all_shifts, axis=0)  # (n_methods, T, 2)
        shifts_final = np.sum(shifts_stack * weights.reshape(-1, 1, 1), axis=0)  # (T, 2)
        
        # Apply weighted average shifts to original video
        corrected_final = np.zeros_like(video)
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    corrected_final[t, c, z] = apply_shift(video[t, c, z], shifts_final[t])
        
    if weights_list is not None:
        try:
            logger.info(f"MVRegFus: Used weights: {dict(zip([m for m, _, _ in valid_results], list(weights_list))) }")
        except Exception:
            logger.info("MVRegFus: Used weights: <unavailable>")
    
    # Prepare registration info
    registration_info = {
        'fusion_strategy': fusion_strategy,
        'methods_used': [method for method, _, _ in valid_results],
        'method_errors': [error for _, _, error in valid_results],
        'final_shifts': shifts_final,
        'n_methods_succeeded': len(valid_results),
        'n_methods_total': len(methods)
    }
    
    # Add fusion weights if using weighted average
    if weights_list is not None:
        try:
            registration_info['fusion_weights'] = dict(zip([m for m, _, _ in valid_results], weights_list))
        except Exception:
            pass
    
    return corrected_final, registration_info


def nanopyx_drift_correction(
    video: np.ndarray,
    drift_channel: int = 0,
    subpixel_precision: float = 0.01,  # Nano-scale precision
    use_gradient_correlation: bool = True,
    adaptive_window: bool = True,
    quality_metrics: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    NanoPyx-inspired drift correction with nano-scale precision and optimization.
    
    Implements high-precision drift correction optimized for nanoscopy data
    with adaptive windowing and gradient-based correlation for enhanced accuracy.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        drift_channel: Channel to use for drift estimation
        subpixel_precision: Target precision for subpixel estimation (pixels)
        use_gradient_correlation: Use gradient-enhanced correlation
        adaptive_window: Use adaptive correlation windows based on image content
        quality_metrics: Whether to compute quality analysis metrics
        
    Returns:
        tuple: (corrected_video, registration_info)
    """
    logger.info(f"Running NanoPyx-style drift correction with {subpixel_precision:.3f}px precision")
    
    # Input validation
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video (TCZYX), got {video.ndim}D")
    
    T, C, Z, Y, X = video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")
    
    # Extract drift channel for all timepoints and Z-slices
    drift_data = video[:, drift_channel, :, :, :].astype(np.float32)
    
    # Create reference image (mean across time for each Z-slice)
    reference_images = np.mean(drift_data, axis=0)  # (Z, Y, X)
    
    # Initialize shift arrays
    shifts = np.zeros((T, 2), dtype=np.float32)  # Only XY translations
    correlation_peaks = np.zeros(T, dtype=np.float32)
    
    # High-precision upsample factor for nano-scale accuracy
    upsample_factor = max(10, int(1.0 / subpixel_precision))
    logger.info(f"NanoPyx: Using upsample factor {upsample_factor} for {subpixel_precision:.3f}px precision")
    
    # Adaptive window sizing based on image content
    window_mask = None
    if adaptive_window:
        # Analyze image gradients to determine optimal correlation window
        middle_z = Z // 2 if Z > 1 else 0
        grad_x = np.gradient(reference_images[middle_z], axis=1)
        grad_y = np.gradient(reference_images[middle_z], axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find regions with high gradient content for correlation windows
        grad_threshold = np.percentile(gradient_magnitude, 75)
        high_grad_mask = gradient_magnitude > grad_threshold
        
        if np.sum(high_grad_mask) > 0.1 * high_grad_mask.size:
            # Use high-gradient regions for correlation
            window_mask = high_grad_mask
            logger.info("NanoPyx: Using adaptive gradient-based correlation windows")
        else:
            # Fall back to center region
            window_mask = None
            logger.info("NanoPyx: Using full-frame correlation (insufficient gradient content)")
    
    # Process each timepoint
    for t in range(T):
        try:
            # Current images for this timepoint
            current_images = drift_data[t]  # (Z, Y, X)
            
            # Initialize per-Z-slice shifts for this timepoint
            z_shifts = []
            z_peaks = []
            
            # Process each Z-slice
            for z in range(Z):
                current_img = current_images[z]
                reference_img = reference_images[z]
                
                # Apply window mask if using adaptive windowing
                if adaptive_window and window_mask is not None:
                    # Create masked versions for correlation
                    current_masked = current_img * window_mask
                    reference_masked = reference_img * window_mask
                else:
                    current_masked = current_img
                    reference_masked = reference_img
                
                # Gradient-enhanced correlation
                if use_gradient_correlation:
                    # Compute gradients
                    curr_grad_x = np.gradient(current_masked, axis=1)
                    curr_grad_y = np.gradient(current_masked, axis=0)
                    ref_grad_x = np.gradient(reference_masked, axis=1) 
                    ref_grad_y = np.gradient(reference_masked, axis=0)
                    
                    # Combine intensity and gradient information
                    current_enhanced = np.stack([current_masked, curr_grad_x, curr_grad_y])
                    reference_enhanced = np.stack([reference_masked, ref_grad_x, ref_grad_y])
                    
                    # Multi-channel phase correlation
                    total_shift = np.zeros(2)
                    total_peak = 0.0
                    
                    for ch in range(3):  # intensity + 2 gradient channels
                        try:
                            shift, error, _ = phase_cross_correlation(
                                reference_enhanced[ch],
                                current_enhanced[ch],
                                upsample_factor=upsample_factor
                            )
                            conf = max(0.0, 1.0 - float(error))
                            total_shift += shift * conf
                            total_peak += conf
                        except Exception as e:
                            logger.warning(f"NanoPyx: Gradient channel {ch} correlation failed: {e}")
                            continue
                    
                    # Average weighted shifts
                    if total_peak > 0:
                        z_shift = total_shift / 3.0  # Average across channels
                        z_peak = total_peak / 3.0
                    else:
                        z_shift = np.array([0.0, 0.0])
                        z_peak = 0.0
                        
                else:
                    # Standard phase correlation
                    try:
                        z_shift, error, _ = phase_cross_correlation(
                            reference_masked,
                            current_masked,
                            upsample_factor=upsample_factor
                        )
                        z_peak = max(0.0, 1.0 - float(error))
                    except Exception as e:
                        logger.warning(f"NanoPyx: Phase correlation failed for t={t}, z={z}: {e}")
                        z_shift = np.array([0.0, 0.0])
                        z_peak = 0.0
                
                z_shifts.append(z_shift)
                z_peaks.append(z_peak)
            
            # Combine Z-slice results (weighted by correlation peak)
            z_shifts = np.array(z_shifts)  # (Z, 2)
            z_peaks = np.array(z_peaks)   # (Z,)
            
            if np.sum(z_peaks) > 0:
                # Weighted average across Z-slices
                weights = z_peaks / np.sum(z_peaks)
                final_shift = np.sum(z_shifts * weights.reshape(-1, 1), axis=0)
                final_peak = np.mean(z_peaks)
            else:
                final_shift = np.array([0.0, 0.0])
                final_peak = 0.0
            
            shifts[t] = final_shift
            correlation_peaks[t] = final_peak
            
        except Exception as e:
            logger.error(f"NanoPyx: Failed to process frame {t}: {e}")
            shifts[t] = np.array([0.0, 0.0])
            correlation_peaks[t] = 0.0
    
    # Apply drift correction
    logger.info("NanoPyx: Applying nano-precision drift correction")
    corrected_video = np.zeros_like(video)
    
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                corrected_video[t, c, z] = apply_shift(video[t, c, z], shifts[t])
    
    # Calculate nano-scale precision metrics
    shift_magnitudes = np.sqrt(np.sum(shifts**2, axis=1))
    avg_precision = np.mean(shift_magnitudes[shift_magnitudes > 0])
    
    # Prepare registration info
    registration_info = {
        'method': 'nanopyx_precision',
        'subpixel_precision_target': subpixel_precision,
        'actual_precision_achieved': min(avg_precision, subpixel_precision) if avg_precision > 0 else subpixel_precision,
        'upsample_factor': upsample_factor,
        'use_gradient_correlation': use_gradient_correlation,
        'adaptive_window': adaptive_window,
        'shifts': shifts,
        'correlation_peaks': correlation_peaks,
        'mean_correlation_peak': np.mean(correlation_peaks),
        'frames_processed': T,
        'z_slices_processed': Z
    }
    
    # Add quality metrics if requested
    if quality_metrics:
        try:
            # Compute FRC-like metric using correlation peaks
            registration_info['frc_metric'] = np.mean(correlation_peaks)
            
            # SQUIRREL-like metric: consistency of shifts
            shift_consistency = 1.0 / (1.0 + np.std(shift_magnitudes))
            registration_info['squirrel_metric'] = shift_consistency
            
            # Decorrelation metric: temporal smoothness
            shift_diffs = np.diff(shifts, axis=0)
            temporal_smoothness = 1.0 / (1.0 + np.mean(np.sqrt(np.sum(shift_diffs**2, axis=1))))
            registration_info['decorrelation_metric'] = temporal_smoothness
            
        except Exception as e:
            logger.warning(f"NanoPyx: Quality metrics calculation failed: {e}")
            registration_info['frc_metric'] = None
            registration_info['squirrel_metric'] = None
            registration_info['decorrelation_metric'] = None
    
    logger.info(f"NanoPyx: Completed with avg precision {avg_precision:.4f}px, "
               f"mean correlation peak: {np.mean(correlation_peaks):.3f}")
    
    return corrected_video, registration_info


# ------------------------------
# Drift correction quality evaluation
# ------------------------------

def evaluate_drift_correction_quality(
    input_video: np.ndarray,           # Original TCZYX
    corrected_video: np.ndarray,       # Drift-corrected TCZYX  
    drift_channel: int = 0,            # Channel used for drift estimation
    method: str = "variance",          # "variance", "mad", "iqr", "psr_residual" (fast, discriminative)
    spatial_weight: str = "gradient",  # "uniform", "gradient", "center"
    exclude_background: bool = True,   # Mask low-intensity regions
    patch_size: Optional[int] = None,  # For patch-based analysis
    return_details: bool = False,      # Return breakdown of metrics
    thumb_max_size: Optional[int] = None  # Max dimension for thumbnail downsampling (psr_residual)
) -> Union[float, Dict[str, Any]]:
    """
    Evaluate drift correction quality using pixel variance analysis across time.
    
    The core idea: if we're imaging the same object over time and it's drifting,
    pixel variance across time will be high with drift and low after correction.
    
    Returns:
    - drift_quality (float): Improvement ratio (input_variance / output_variance)
      Values > 1.0 indicate improvement, < 1.0 indicate degradation
    - If return_details=True: Dict with comprehensive metrics breakdown
    """
    # Input validation
    if input_video.shape != corrected_video.shape:
        raise ValueError(f"Input and corrected video shapes must match: {input_video.shape} vs {corrected_video.shape}")
    
    if input_video.ndim != 5:
        raise ValueError(f"Videos must be 5D (TCZYX), got {input_video.ndim}D")
    
    T, C, Z, Y, X = input_video.shape
    if not (0 <= drift_channel < C):
        raise ValueError(f"drift_channel {drift_channel} out of bounds for {C} channels")
    
    # Extract drift channel and create max projections (T, Y, X)
    input_proj = np.max(input_video[:, drift_channel, :, :, :], axis=1).astype(np.float32)
    corrected_proj = np.max(corrected_video[:, drift_channel, :, :, :], axis=1).astype(np.float32)

    # Fast, discriminative method based on PSR and residual drift on thumbnails
    if method.lower() in {"psr_residual", "fast", "psr"}:
        def _downsample_stack(stack: np.ndarray, target_max: int = 256) -> np.ndarray:
            """Downsample (T, Y, X) by integer stride so max(Y, X) <= target_max (fast, no deps)."""
            T_, Y_, X_ = stack.shape
            scale = max(1, int(np.ceil(max(Y_, X_) / float(target_max))))
            if scale <= 1:
                return stack
            return stack[:, ::scale, ::scale]

        def _phase_corr_and_psr(ref: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, float]:
            """
            Compute translation via phase correlation and PSR on correlation surface.
            Returns (shift_yx, psr). Implemented with NumPy for speed on thumbnails.
            """
            a = ref.astype(np.float32)
            b = img.astype(np.float32)
            H, W = a.shape
            # Windowing reduces edge leakage
            if H > 32 and W > 32:
                wy = np.hanning(H)[:, None]
                wx = np.hanning(W)[None, :]
                win = wy * wx
                a = a * win
                b = b * win
            Fa = np.fft.fft2(a)
            Fb = np.fft.fft2(b)
            R = Fa * np.conj(Fb)
            R_abs = np.abs(R)
            R_abs[R_abs < 1e-12] = 1e-12
            Rn = R / R_abs
            r = np.fft.ifft2(Rn).real
            # Peak and shift (wrap-aware)
            idx = np.argmax(r)
            py = idx // W
            px = idx % W
            peak = r[py, px]
            if py > H // 2:
                py -= H
            if px > W // 2:
                px -= W
            # PSR: exclude 5x5 window around peak
            mask = np.ones_like(r, dtype=bool)
            y0 = (py % H)
            x0 = (px % W)
            y1 = max(0, y0 - 2); y2 = min(H, y0 + 3)
            x1 = max(0, x0 - 2); x2 = min(W, x0 + 3)
            mask[y1:y2, x1:x2] = False
            side = r[mask]
            mu = float(side.mean()) if side.size > 0 else 0.0
            sigma = float(side.std(ddof=1)) if side.size > 1 else 1.0
            psr = (float(peak) - mu) / (sigma + 1e-12)
            return np.array([float(py), float(px)], dtype=np.float32), psr

        # Build thumbnails
        target_dim = int(thumb_max_size) if (thumb_max_size is not None and thumb_max_size > 32) else 256
        input_thumb = _downsample_stack(input_proj, target_dim)
        corrected_thumb = _downsample_stack(corrected_proj, target_dim)
        T_ = input_thumb.shape[0]
        # Reference = mean frame of input (same reference for both before/after)
        ref = input_thumb.mean(axis=0)
        # Compute PSR before/after and residual shifts before/after (between consecutive frames)
        psr_before = np.zeros(T_, dtype=np.float32)
        psr_after = np.zeros(T_, dtype=np.float32)
        resid_before = []
        resid_after = []
        prev_b = None
        prev_a = None
        for t in range(T_):
            _, psr_b = _phase_corr_and_psr(ref, input_thumb[t])
            _, psr_a = _phase_corr_and_psr(ref, corrected_thumb[t])
            psr_before[t] = psr_b
            psr_after[t] = psr_a
            if t > 0:
                # residual between consecutive frames (should be ~0 after good correction)
                shift_b, _ = _phase_corr_and_psr(input_thumb[t-1], input_thumb[t])
                shift_a, _ = _phase_corr_and_psr(corrected_thumb[t-1], corrected_thumb[t])
                resid_before.append(float(np.linalg.norm(shift_b)))
                resid_after.append(float(np.linalg.norm(shift_a)))

        resid_before = np.array(resid_before) if resid_before else np.array([0.0])
        resid_after = np.array(resid_after) if resid_after else np.array([0.0])

        # Metrics
        alignment_conf_after = float(np.median(psr_after))            # absolute quality
        alignment_conf_before = float(np.median(psr_before))
        improvement_psr = float(np.median(psr_after - psr_before))     # delta quality
        median_resid_before = float(np.median(resid_before))
        median_resid_after = float(np.median(resid_after))
        residual_reduction = median_resid_before / (median_resid_after + 1e-6)

        # Normalize to a drift_quality-like score >1 is good
        # Combine PSR improvement and residual reduction; clamp to sensible range
        drift_quality = max(0.0, 0.5 * (1.0 + improvement_psr / (abs(alignment_conf_before) + 5.0))
                                 + 0.5 * min(3.0, residual_reduction))

        # Edge and temporal improvements can be approximated cheaply on thumbnails
        # Use simple gradient energy ratio as edge proxy
        def _grad_energy(stack: np.ndarray) -> float:
            gy, gx = np.gradient(stack, axis=(1, 2))
            return float(np.mean(np.sqrt(gy**2 + gx**2)))
        edge_improvement = _grad_energy(corrected_thumb) / (_grad_energy(input_thumb) + 1e-12)
        # Temporal consistency: lower residual after is better -> map to >1 better
        temporal_improvement = (median_resid_before + 1e-6) / (median_resid_after + 1e-6)

        composite_score = (
            0.6 * drift_quality +
            0.25 * edge_improvement +
            0.15 * temporal_improvement
        )

        if return_details:
            details = {
                'drift_quality': float(drift_quality),
                'alignment_confidence_after_psr_median': alignment_conf_after,
                'alignment_confidence_before_psr_median': alignment_conf_before,
                'psr_improvement_median': improvement_psr,
                'residual_before_median_px': median_resid_before,
                'residual_after_median_px': median_resid_after,
                'residual_reduction_ratio': float(residual_reduction),
                'edge_sharpness_improvement': float(edge_improvement),
                'temporal_consistency_improvement': float(temporal_improvement),
                'composite_score': float(composite_score),
                'method': method,
                'spatial_weight': 'uniform',
                'background_pixels_excluded': 0,
                'total_pixels_analyzed': int(input_thumb.shape[1] * input_thumb.shape[2]),
                'patch_analysis': None,
                'quality_interpretation': _interpret_quality_score(float(drift_quality), float(composite_score))
            }
            # Add regime classification to distinguish good-stable vs bad-stable
            # Heuristics: PSR absolute threshold and residuals
            psr_thresh_good = 20.0
            psr_thresh_poor = 8.0
            if alignment_conf_before >= psr_thresh_good and median_resid_before < 0.5 and improvement_psr < 1.0:
                details['regime'] = 'already_aligned_good_signal'
                details['regime_note'] = 'Signal is strong and already stable; little to correct.'
            elif alignment_conf_before < psr_thresh_poor and alignment_conf_after < psr_thresh_poor and improvement_psr < 1.0:
                details['regime'] = 'uninformative_or_low_signal'
                details['regime_note'] = 'Both before/after show low alignment confidence; drift assessment unreliable.'
            elif residual_reduction > 1.5 and alignment_conf_after >= max(psr_thresh_poor, alignment_conf_before):
                details['regime'] = 'successful_correction'
                details['regime_note'] = 'Residual drift reduced and alignment confidence improved.'
            elif median_resid_after >= median_resid_before * 1.1:
                details['regime'] = 'potential_failure'
                details['regime_note'] = 'Residual drift did not decrease; review parameters/method.'
            else:
                details['regime'] = 'inconclusive'
                details['regime_note'] = 'Mixed signals; try alternative method or parameters.'
            return details
        else:
            return float(drift_quality)
    
    # Create background mask (exclude low-intensity regions)
    background_mask = None
    if exclude_background:
        # Use mean intensity across time to identify background
        mean_intensity = np.mean(input_proj, axis=0)
        # Threshold at mean + 0.5 * std to exclude noise/background
        threshold = np.mean(mean_intensity) + 0.5 * np.std(mean_intensity)
        background_mask = mean_intensity > threshold
        if np.sum(background_mask) < 0.1 * Y * X:  # If too restrictive, use lower threshold
            threshold = np.mean(mean_intensity)
            background_mask = mean_intensity > threshold
    
    # Create spatial weighting map
    spatial_weights = np.ones((Y, X), dtype=np.float32)
    
    if spatial_weight == "gradient":
        # Weight by local gradient (edge pixels are more informative)
        mean_frame = np.mean(input_proj, axis=0)
        gy, gx = np.gradient(mean_frame)
        gradient_mag = np.sqrt(gy**2 + gx**2)
        # Normalize and add small constant to avoid zero weights
        spatial_weights = (gradient_mag / (np.max(gradient_mag) + 1e-6)) + 0.1
        
    elif spatial_weight == "center":
        # Weight central pixels higher (drift effects more visible at edges)
        cy, cx = Y // 2, X // 2
        yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
        dist_from_center = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        spatial_weights = 1.0 - (dist_from_center / max_dist) * 0.8  # Range [0.2, 1.0]
    
    # Apply background mask to weights
    if background_mask is not None:
        spatial_weights *= background_mask.astype(np.float32)
    
    # Calculate variance metrics
    def _calculate_weighted_variance(proj_stack, weights, method_name):
        """Calculate weighted variance across time for each pixel."""
        if method_name == "variance":
            # Standard variance
            temporal_var = np.var(proj_stack, axis=0, ddof=1)
        elif method_name == "mad":
            # Median Absolute Deviation (more robust to outliers)
            temporal_median = np.median(proj_stack, axis=0)
            temporal_var = np.median(np.abs(proj_stack - temporal_median[None, :, :]), axis=0)
        elif method_name == "iqr":
            # Interquartile Range
            q75 = np.percentile(proj_stack, 75, axis=0)
            q25 = np.percentile(proj_stack, 25, axis=0)
            temporal_var = q75 - q25
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Apply spatial weighting
        weighted_var = temporal_var * weights
        
        # Return weighted average (most common pixel variance equivalent)
        total_weight = np.sum(weights)
        if total_weight > 0:
            return np.sum(weighted_var) / total_weight
        else:
            return 0.0
    
    # Calculate patch-based analysis if requested
    patch_results = None
    if patch_size is not None and patch_size > 10:
        patch_results = []
        n_patches_y = max(1, Y // patch_size)
        n_patches_x = max(1, X // patch_size)
        
        for py in range(n_patches_y):
            for px in range(n_patches_x):
                y1 = py * patch_size
                y2 = min((py + 1) * patch_size, Y)
                x1 = px * patch_size
                x2 = min((px + 1) * patch_size, X)
                
                patch_weights = spatial_weights[y1:y2, x1:x2]
                input_patch_var = _calculate_weighted_variance(
                    input_proj[:, y1:y2, x1:x2], patch_weights, method
                )
                corrected_patch_var = _calculate_weighted_variance(
                    corrected_proj[:, y1:y2, x1:x2], patch_weights, method
                )
                
                patch_quality = input_patch_var / (corrected_patch_var + 1e-12)
                patch_results.append({
                    'patch_coords': (y1, y2, x1, x2),
                    'input_variance': input_patch_var,
                    'corrected_variance': corrected_patch_var,
                    'quality_ratio': patch_quality
                })
    
    # Calculate main metrics
    input_variance = _calculate_weighted_variance(input_proj, spatial_weights, method)
    corrected_variance = _calculate_weighted_variance(corrected_proj, spatial_weights, method)
    
    # Main drift quality metric (improvement ratio)
    drift_quality = input_variance / (corrected_variance + 1e-12)
    
    # Additional complementary metrics
    def _edge_sharpness_improvement():
        """Measure how much sharper edges become after correction."""
        try:
            from scipy import ndimage
            
            # Calculate edge strength using Sobel filter
            def edge_strength(img_stack):
                edge_strengths = []
                for t in range(T):
                    sx = ndimage.sobel(img_stack[t], axis=1)
                    sy = ndimage.sobel(img_stack[t], axis=0)
                    edge_mag = np.sqrt(sx**2 + sy**2)
                    edge_strengths.append(np.mean(edge_mag))
                return np.mean(edge_strengths)
            
            input_sharpness = edge_strength(input_proj)
            corrected_sharpness = edge_strength(corrected_proj)
            return corrected_sharpness / (input_sharpness + 1e-12)
        except ImportError:
            return 1.0  # Neutral if scipy not available
    
    def _temporal_consistency():
        """Measure smoothness of intensity profiles over time."""
        # Calculate standard deviation of frame-to-frame differences
        input_diffs = np.diff(input_proj, axis=0)
        corrected_diffs = np.diff(corrected_proj, axis=0)
        
        input_consistency = np.mean(np.std(input_diffs, axis=0))
        corrected_consistency = np.mean(np.std(corrected_diffs, axis=0))
        
        return input_consistency / (corrected_consistency + 1e-12)
    
    edge_improvement = _edge_sharpness_improvement()
    temporal_improvement = _temporal_consistency()
    
    # Composite score (weighted combination)
    composite_score = (
        0.6 * drift_quality +           # Main metric (60%)
        0.25 * edge_improvement +       # Edge sharpness (25%) 
        0.15 * temporal_improvement     # Temporal consistency (15%)
    )
    
    if return_details:
        details = {
            'drift_quality': float(drift_quality),
            'input_variance': float(input_variance),
            'corrected_variance': float(corrected_variance),
            'edge_sharpness_improvement': float(edge_improvement),
            'temporal_consistency_improvement': float(temporal_improvement),
            'composite_score': float(composite_score),
            'method': method,
            'spatial_weight': spatial_weight,
            'background_pixels_excluded': int(np.sum(~background_mask)) if background_mask is not None else 0,
            'total_pixels_analyzed': int(np.sum(spatial_weights > 0)),
            'patch_analysis': patch_results,
            'quality_interpretation': _interpret_quality_score(drift_quality, composite_score)
        }
        return details
    else:
        return float(drift_quality)


def _interpret_quality_score(drift_quality: float, composite_score: float) -> Dict[str, str]:
    """Provide human-readable interpretation of quality scores."""
    
    def _categorize_score(score: float) -> str:
        if score > 2.0:
            return "Excellent improvement"
        elif score > 1.5:
            return "Good improvement"
        elif score > 1.2:
            return "Modest improvement"
        elif score > 0.95:
            return "Minimal change"
        else:
            return "Degradation or no improvement"
    
    return {
        'drift_quality_category': _categorize_score(drift_quality),
        'composite_score_category': _categorize_score(composite_score),
        'recommendation': (
            "Drift correction appears successful" if composite_score > 1.2
            else "Consider different correction parameters or method" if composite_score > 0.8
            else "Drift correction may have failed - investigate parameters"
        )
    }


# ------------------------------
# Unified interface
# ------------------------------

def drift_correct(
    video: np.ndarray,
    method: str = 'skimage',
    drift_channel: int = 0,
    **kwargs
) -> Tuple[np.ndarray, Any]:
    """
    Unified drift correction interface.
    
    Args:
        video: Input video (T, C, Z, Y, X)
        method: Algorithm to use ('skimage', 'pystackreg', 'imreg_dft', 'pygpureg', 'mvregfus', 'nanopyx')
        drift_channel: Channel to use for drift estimation
        **kwargs: Algorithm-specific parameters
        #TODO add which kwargs for each method
        skimage: {



        
    Returns:
        tuple: (corrected_video, algorithm_specific_output)
    """
    method = method.lower()
    
    if method == 'skimage':
        return skimage_drift_correction(video, drift_channel, **kwargs)
    elif method == 'pystackreg':
        return pystackreg_drift_correction(video, drift_channel, **kwargs)
    elif method == 'imreg_dft':
        return imreg_dft_drift_correction(video, drift_channel, **kwargs)
    elif method == 'pygpureg':
        return pygpureg_drift_correction(video, drift_channel, **kwargs)
    elif method == 'mvregfus':
        return mvregfus_drift_correction(video, drift_channel, **kwargs)
    elif method == 'nanopyx':
        return nanopyx_drift_correction(video, drift_channel, **kwargs)
    elif method == 'greedy_psr':
        return greedy_psr_drift_correction(video, drift_channel, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Available: skimage, pystackreg, imreg_dft, pygpureg, mvregfus, nanopyx")


# ------------------------------
# Command line interface
# ------------------------------

def main():
    """Command line interface for drift correction."""
    import argparse
    import os
    
    # Configure logging to ensure messages are displayed
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Modular drift correction with multiple algorithms")
    parser.add_argument('--input', required=True, help='Input image file')
    parser.add_argument('--output-dir', help='Output directory (default: next to input)')
    parser.add_argument('--output-file-name-extension', type=str, default='', 
                       help='Extension to add to output filename (for multiple experiments)')
    parser.add_argument('--method', default='skimage', 
                       choices=['skimage', 'pystackreg', 'imreg_dft', 'pygpureg', 'mvregfus', 'nanopyx', 'greedy_psr'],
                       help='Drift correction algorithm')
    parser.add_argument('--drift-channel', type=int, default=0, help='Channel for drift estimation')
    parser.add_argument('--reference-frame', default='mean', help='Reference frame (mean/first/index)')
    
    # Algorithm-specific options
    parser.add_argument('--upsample-factor', type=int, default=20, help='Subpixel refinement (skimage, 1-100)')
    parser.add_argument('--gaussian-filter', type=float, help='Gaussian smoothing sigma for noise reduction (skimage)')
    parser.add_argument('--registration-strategy', default='to_mean', 
                       choices=['to_mean', 'sequential', 'running_mean'],
                       help='Registration strategy (skimage)')
    parser.add_argument('--transformation', default='translation', help='IGNORED - always uses translation mode for XY drift only')
    parser.add_argument('--estimate-rotation', action='store_true', help='IGNORED - only XY translation supported')
    parser.add_argument('--estimate-scale', action='store_true', help='IGNORED - only XY translation supported')
    
    # MVRegFus-specific options
    parser.add_argument('--fusion-strategy', default='weighted_average',
                       choices=['weighted_average', 'median', 'best_quality'],
                       help='Multi-view fusion strategy (mvregfus)')
    
    # NanoPyx-specific options  
    parser.add_argument('--subpixel-precision', type=float, default=0.01,
                       help='Target subpixel precision in pixels (nanopyx)')
    parser.add_argument('--use-gradient-correlation', type=bool, default=True,
                       help='Use gradient-enhanced correlation (nanopyx)')
    parser.add_argument('--adaptive-window', type=bool, default=True,
                       help='Use adaptive correlation windows (nanopyx)')
    
    # PyGPUreg-specific options
    parser.add_argument('--gpu-memory-limit', type=float, default=0.8,
                       help='Fraction of GPU memory to use (pygpureg)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size for GPU processing, 0=auto (pygpureg)')
    parser.add_argument('--correlation-threshold', type=float, default=0.1,
                       help='Minimum correlation for valid registration (pygpureg)')

    # Greedy PSR-specific options
    parser.add_argument('--greedy-step-px', type=int, default=1,
                       help='Integer step size in pixels for greedy neighborhood search (greedy_psr)')
    parser.add_argument('--greedy-max-iters', type=int, default=10,
                       help='Max greedy iterations per frame (greedy_psr)')
    parser.add_argument('--greedy-neighborhood', type=str, default='8', choices=['4','8'],
                       help='Neighborhood connectivity for greedy search (greedy_psr)')
    parser.add_argument('--greedy-thumb-max-size', type=int, default=256,
                       help='Thumbnail max dimension for PSR scoring (greedy_psr)')
    
    # Quality evaluation options
    parser.add_argument('--evaluate-quality', action='store_true', help='Evaluate drift correction quality')
    parser.add_argument('--quality-method', default='variance', choices=['variance', 'mad', 'iqr', 'psr_residual'], 
                       help='Quality evaluation metric')
    parser.add_argument('--thumb-max-size', type=int, default=None,
                       help='Max dimension for thumbnail downsampling in psr_residual method (default 256)')

    # Optional ML scorer
    parser.add_argument('--ml-train-csv', type=str, default=None,
                       help='CSV of features/labels to train a lightweight scorer (columns: features + label)')
    parser.add_argument('--ml-model-out', type=str, default=None,
                       help='Path to save trained ML model (joblib)')
    parser.add_argument('--ml-model-in', type=str, default=None,
                       help='Path to load an existing ML model to score the current run')
    parser.add_argument('--spatial-weight', default='gradient', choices=['uniform', 'gradient', 'center'],
                       help='Spatial weighting for quality evaluation')
    parser.add_argument('--patch-size', type=int, help='Patch size for regional quality analysis')
    
    args = parser.parse_args()
    
    # Load image
    logger.info(f"Loading image: {args.input}")
    bio_image = rp.load_tczyx_image(args.input)
    video = bio_image.data  # Extract numpy array from BioImage object
    
    # Prepare output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), "_drift_corrected")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare algorithm-specific kwargs
    kwargs = {
        'reference_frame': args.reference_frame
    }
    
    if args.method == 'skimage':
        kwargs['upsample_factor'] = args.upsample_factor
        kwargs['gaussian_filter'] = args.gaussian_filter
        kwargs['registration_strategy'] = args.registration_strategy
    elif args.method == 'pystackreg':
        kwargs['transformation'] = args.transformation
    elif args.method == 'imreg_dft':
        kwargs['estimate_rotation'] = args.estimate_rotation
        kwargs['estimate_scale'] = args.estimate_scale
    elif args.method == 'mvregfus':
        kwargs['fusion_strategy'] = args.fusion_strategy
    elif args.method == 'nanopyx':
        kwargs['subpixel_precision'] = args.subpixel_precision
        kwargs['use_gradient_correlation'] = args.use_gradient_correlation
        kwargs['adaptive_window'] = args.adaptive_window
    elif args.method == 'pygpureg':
        kwargs['gpu_memory_limit'] = args.gpu_memory_limit
        kwargs['batch_size'] = args.batch_size
        kwargs['correlation_threshold'] = args.correlation_threshold
    elif args.method == 'greedy_psr':
        kwargs['step_px'] = args.greedy_step_px
        kwargs['max_iters'] = args.greedy_max_iters
        kwargs['neighborhood'] = args.greedy_neighborhood
        kwargs['thumb_max_size'] = args.greedy_thumb_max_size
    
    # Run drift correction
    logger.info(f"Running drift correction with method: {args.method}")
    start_time = time.time()
    
    corrected, output_data = drift_correct(
        video, 
        method=args.method, 
        drift_channel=args.drift_channel, 
        **kwargs
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Drift correction completed in {elapsed:.2f} seconds")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_drift_corrected_{args.method}.tif")
    
    logger.info(f"Saving corrected image: {output_path}")
    rp.save_tczyx_image(corrected, output_path)
    
    # Save algorithm-specific outputs
    if isinstance(output_data, np.ndarray):
        output_npy = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_drift_data_{args.method}.npy")
        np.save(output_npy, output_data)
        logger.info(f"Saved drift data: {output_npy}")
    
    # Evaluate quality if requested
    if args.evaluate_quality:
        logger.info("Evaluating drift correction quality...")
        quality_start = time.time()
        
        quality_results = evaluate_drift_correction_quality(
            video, corrected, 
            drift_channel=args.drift_channel,
            method=args.quality_method,
            spatial_weight=args.spatial_weight,
            patch_size=args.patch_size,
            thumb_max_size=args.thumb_max_size,
            return_details=True
        )
        
        quality_elapsed = time.time() - quality_start
        logger.info(f"Quality evaluation completed in {quality_elapsed:.2f} seconds")
        
        # Report quality results (quality_results is guaranteed to be Dict due to return_details=True)
        if isinstance(quality_results, dict):
            # Add timing information to quality results
            quality_results['drift_correction_duration_seconds'] = elapsed
            quality_results['quality_evaluation_duration_seconds'] = quality_elapsed
            
            interp = quality_results['quality_interpretation']
            
            # Log comprehensive quality metrics
            logger.info("=== Drift Correction Quality Assessment ===")
            logger.info(f"Drift quality: {quality_results['drift_quality']:.3f} ({interp['drift_quality_category']})")
            if 'input_variance' in quality_results and 'corrected_variance' in quality_results:
                logger.info(f"  - Input variance: {quality_results['input_variance']:.1f}")
                logger.info(f"  - Corrected variance: {quality_results['corrected_variance']:.1f}")
            else:
                # psr_residual summary fields if available
                if 'alignment_confidence_before_psr_median' in quality_results:
                    logger.info(
                        f"  - PSR before/after (median): "
                        f"{quality_results.get('alignment_confidence_before_psr_median'):.2f} -> "
                        f"{quality_results.get('alignment_confidence_after_psr_median'):.2f}"
                    )
                if 'residual_before_median_px' in quality_results and 'residual_after_median_px' in quality_results:
                    logger.info(
                        f"  - Residual drift (median px): "
                        f"{quality_results.get('residual_before_median_px'):.3f} -> "
                        f"{quality_results.get('residual_after_median_px'):.3f}"
                    )
                if 'residual_reduction_ratio' in quality_results:
                    logger.info(f"  - Residual reduction ratio: {quality_results.get('residual_reduction_ratio'):.3f}")
            logger.info(f"Edge sharpness improvement: {quality_results['edge_sharpness_improvement']:.3f}")
            logger.info(f"Temporal consistency improvement: {quality_results['temporal_consistency_improvement']:.3f}")
            logger.info(f"Composite score: {quality_results['composite_score']:.3f} ({interp['composite_score_category']})")
            logger.info(f"Pixels analyzed: {quality_results['total_pixels_analyzed']:,} (excluded {quality_results['background_pixels_excluded']:,} background)")
            logger.info(f"Drift correction duration: {elapsed:.2f}s")
            logger.info(f"Quality evaluation duration: {quality_elapsed:.2f}s")
            logger.info(f"Recommendation: {interp['recommendation']}")
            logger.info("=========================================")
            
            # Save detailed quality results
            quality_json = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_quality_{args.method}.json")
            import json
            with open(quality_json, 'w') as f:
                json.dump(quality_results, f, indent=2)
            logger.info(f"Saved quality analysis: {quality_json}")

            # Also compute and surface a compact PSR/residual summary alongside, for consistency across methods
            psr_json = None
            try:
                psr_summary = evaluate_drift_correction_quality(
                    video, corrected,
                    drift_channel=args.drift_channel,
                    method='psr_residual',
                    return_details=True,
                    thumb_max_size=args.thumb_max_size
                )
                if isinstance(psr_summary, dict):
                    # Merge essential PSR fields into main quality JSON object for easy consumption
                    merged = dict(quality_results)
                    merged.update({
                        'psr_alignment_confidence_after_median': psr_summary.get('alignment_confidence_after_psr_median'),
                        'psr_alignment_confidence_before_median': psr_summary.get('alignment_confidence_before_psr_median'),
                        'psr_improvement_median': psr_summary.get('psr_improvement_median'),
                        'residual_before_median_px': psr_summary.get('residual_before_median_px'),
                        'residual_after_median_px': psr_summary.get('residual_after_median_px'),
                        'residual_reduction_ratio': psr_summary.get('residual_reduction_ratio'),
                        'regime': psr_summary.get('regime'),
                        'regime_note': psr_summary.get('regime_note'),
                    })
                    with open(quality_json, 'w') as f:
                        json.dump(merged, f, indent=2)
                    # Also save a standalone PSR summary JSON for tooling
                    psr_json = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_quality_psr_residual_summary_{args.method}.json")
                    with open(psr_json, 'w') as f:
                        json.dump(psr_summary, f, indent=2)
                    logger.info(f"Saved PSR/residual summary and merged into main quality JSON: {psr_json}")
            except Exception as e:
                logger.warning(f"PSR/residual summary computation failed: {e}")

            # Optional: ML training or scoring
            try:
                if args.ml_train_csv is not None:
                    report = _ml_train_scorer_from_csv(args.ml_train_csv, args.ml_model_out)
                    report_json = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_ml_train_report.json")
                    with open(report_json, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Saved ML training report: {report_json}")
                if args.ml_model_in is not None:
                    # Build feature vector from available quality metrics
                    feat = {}
                    # Core features from variance/composite path
                    for k in ['drift_quality','edge_sharpness_improvement','temporal_consistency_improvement','composite_score']:
                        if k in quality_results and isinstance(quality_results[k], (int, float)):
                            feat[k] = float(quality_results[k])
                    # Add psr summary if available
                    if 'alignment_confidence_after_psr_median' in quality_results:
                        # Already a psr_residual run
                        psr_feat = quality_results
                    else:
                        # Try to load the psr summary we just wrote
                        psr_feat = {}
                        try:
                            if psr_json is not None:
                                with open(psr_json, 'r') as f:
                                    psr_feat = json.load(f)
                        except Exception:
                            pass
                    for k in [
                        'alignment_confidence_after_psr_median',
                        'alignment_confidence_before_psr_median',
                        'psr_improvement_median',
                        'residual_before_median_px',
                        'residual_after_median_px',
                        'residual_reduction_ratio'
                    ]:
                        if k in psr_feat and isinstance(psr_feat[k], (int, float)):
                            feat[k] = float(psr_feat[k])
                    ml_score = _ml_score_current_run(args.ml_model_in, feat)
                    ml_json = os.path.join(args.output_dir, f"{base_name}{args.output_file_name_extension}_ml_score_{args.method}.json")
                    with open(ml_json, 'w') as f:
                        json.dump({'features': feat, **ml_score}, f, indent=2)
                    logger.info(f"Saved ML score: {ml_json}")
            except Exception as e:
                logger.warning(f"ML scorer step failed: {e}")
        else:
            logger.info(f"Drift correction quality score: {quality_results:.3f}")
    
    logger.info("Drift correction completed successfully!")


def test_drift_quality_on_files(
    input_file: str,
    corrected_file: str,
    drift_channel: int = 0,
    show_details: bool = True
) -> Union[float, Dict[str, Any]]:
    """
    Standalone function to evaluate drift correction quality between two files.
    
    Usage example:
        quality = test_drift_quality_on_files(
            "original.tif", 
            "drift_corrected.tif",
            drift_channel=0
        )
        print(f"Drift quality: {quality['drift_quality']:.2f}")
    """
    # Load both videos
    logger.info(f"Loading input video: {input_file}")
    input_bio = rp.load_tczyx_image(input_file)
    input_video = input_bio.data
    
    logger.info(f"Loading corrected video: {corrected_file}")
    corrected_bio = rp.load_tczyx_image(corrected_file)
    corrected_video = corrected_bio.data
    
    # Evaluate quality - always request details for proper reporting
    logger.info("Evaluating drift correction quality...")
    quality_results = evaluate_drift_correction_quality(
        input_video, corrected_video, drift_channel, return_details=True
    )
    
    if show_details and isinstance(quality_results, dict):
        interp = quality_results['quality_interpretation']
        logger.info(f"Drift correction quality: {quality_results['drift_quality']:.3f} ({interp['drift_quality_category']})")
        logger.info(f"Composite score: {quality_results['composite_score']:.3f} ({interp['composite_score_category']})")
        logger.info(f"Recommendation: {interp['recommendation']}")
        
        logger.info(f"Edge sharpness improvement: {quality_results['edge_sharpness_improvement']:.3f}")
        logger.info(f"Temporal consistency improvement: {quality_results['temporal_consistency_improvement']:.3f}")
    
    return quality_results if show_details else quality_results['drift_quality'] if isinstance(quality_results, dict) else quality_results


# ------------------------------
# Lightweight ML scorer utilities (optional)
# ------------------------------

def _ml_train_scorer_from_csv(csv_path: str, model_out: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a tiny baseline scorer (LogisticRegression) from a CSV containing features and a label column.
    Expected columns: any numeric features plus a 'label' column with values {0,1} (improved or not).
    Returns training report and optionally saves the model via joblib.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score
        import joblib
    except Exception as e:
        raise RuntimeError(f"scikit-learn/joblib not available: {e}")

    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column with binary labels {0,1}")
    y = df['label'].astype(int).to_numpy()
    X = df.drop(columns=['label'])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0.0).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    yprob = clf.predict_proba(Xte)[:, 1]
    ypred = (yprob >= 0.5).astype(int)
    report = {
        'roc_auc': float(roc_auc_score(yte, yprob)),
        'accuracy': float(accuracy_score(yte, ypred)),
        'n_train': int(len(ytr)),
        'n_test': int(len(yte))
    }
    if model_out:
        joblib.dump(clf, model_out)
        report['model_saved'] = os.path.abspath(model_out)
    return report


def _ml_score_current_run(model_in: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Load a saved model and score the current run using provided features dict."""
    import numpy as np
    import joblib
    clf = joblib.load(model_in)
    # Consistent feature ordering: sort keys
    keys = sorted([k for k, v in features.items() if isinstance(v, (int, float))])
    X = np.array([[float(features[k]) for k in keys]], dtype=float)
    prob = float(clf.predict_proba(X)[0, 1])
    return {'ml_score_probability_improved': prob, 'feature_keys': keys}


if __name__ == "__main__":
    main()