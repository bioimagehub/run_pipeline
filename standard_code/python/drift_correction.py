import argparse
import os
import time
import logging
from typing import Optional, Tuple, Any, Dict, Iterable, List, Union

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Local helper used throughout the repo
import bioimage_pipeline_utils as rp

# CPU registration (StackReg)
from pystackreg import StackReg
from bioio.writers import OmeTiffWriter  # type: ignore[import]

# Module logger
logger = logging.getLogger(__name__)

# ------------------------------
# CuPy preset configuration
# ------------------------------
CUPY_PRESETS: Dict[str, Dict[str, Optional[float]]] = {
    # Speed focused
    "go-fast": {
        "downsample": 4,
        "refine_upsample": 10,
        "highpass_pix": 3.0,
        "lowpass_pix": 96.0,
        "use_running_ref": True,
    },
    # Balanced default
    "balanced": {
        "downsample": 2,
        "refine_upsample": 20,
        "highpass_pix": 3.0,
        "lowpass_pix": 128.0,
        "use_running_ref": True,
    },
    # Quality focused
    "max-accuracy": {
        "downsample": 1,
        "refine_upsample": 50,
        "highpass_pix": None,
        "lowpass_pix": None,
        "use_running_ref": True,
    },
}

def _resolve_cupy_params(
    *,
    preset: Optional[str],
    downsample: Optional[int],
    refine_upsample: Optional[int],
    highpass_pix: Optional[float],
    lowpass_pix: Optional[float],
    use_running_ref: Optional[bool],
) -> Dict[str, Any]:
    """Merge preset with explicit overrides and return effective parameters."""
    # Start from preset values
    eff: Dict[str, Any] = {}
    if preset:
        key = preset.lower()
        if key not in CUPY_PRESETS:
            raise ValueError(f"Unknown CuPy preset: {preset}. Choose from {list(CUPY_PRESETS)}")
        eff.update(CUPY_PRESETS[key])
    # Apply explicit overrides if provided
    if downsample is not None:
        eff["downsample"] = int(downsample)
    if refine_upsample is not None:
        eff["refine_upsample"] = int(refine_upsample)
    if highpass_pix is not None or (highpass_pix is None and "highpass_pix" in eff):
        eff["highpass_pix"] = highpass_pix
    if lowpass_pix is not None or (lowpass_pix is None and "lowpass_pix" in eff):
        eff["lowpass_pix"] = lowpass_pix
    if use_running_ref is not None:
        eff["use_running_ref"] = bool(use_running_ref)

    # Defaults if still missing (max-accuracy)
    eff.setdefault("downsample", 1)
    eff.setdefault("refine_upsample", 50)
    eff.setdefault("highpass_pix", None)
    eff.setdefault("lowpass_pix", None)
    eff.setdefault("use_running_ref", True)

    return eff


def drift_correct_xy_parallel(
    video: np.ndarray,
    drift_correct_channel: int = 0,
    transform_threads: int = 0,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if logger is None:
        import logging
        logger = logging
    # --- Input validation and shape normalization ---
    arr = video
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input video must be a numpy ndarray, got {type(arr)}")
    if arr.ndim < 5:
        arr = arr[(None,) * (5 - arr.ndim)]
    if arr.ndim != 5:
        raise ValueError(f"Input video must be 5D (TCZYX), got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Input video must have a numeric dtype, got {arr.dtype}")
    T, C, Z, _, _ = arr.shape
    if not (0 <= drift_correct_channel < C):
        raise ValueError(f"drift_correct_channel {drift_correct_channel} is out of bounds for {C} channels.")
    video = arr
    """
    CPU baseline using pystackreg (translation only), identical to convert_to_tif.py.
    video shape: (T, C, Z, Y, X)
    Returns: corrected_video (same shape), tmats (T, 3, 3)
    """
    T, C, Z, _, _ = video.shape
    corrected_video = np.zeros_like(video)

    sr = StackReg(StackReg.TRANSLATION)

    # Max-projection along Z for drift correction
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)

    tmats = sr.register_stack(ref_stack, reference='mean', verbose=False, axis=0)

    # Optionally parallelize per-plane transform for speed
    def _transform_one(args):
        t, c, z = args
        return c, z, sr.transform(video[t, c, z, :, :], tmats[t])

    for t in range(T):
        if transform_threads and transform_threads > 1:
            tasks = [(t, c, z) for c in range(C) for z in range(Z)]
            results_gen = Parallel(
                n_jobs=transform_threads, prefer="threads"
            )(delayed(_transform_one)(a) for a in tasks)
            results_list: List[Tuple[int, int, np.ndarray]] = list(results_gen)  # type: ignore[assignment]
            for c, z, plane in results_list:
                corrected_video[t, c, z, :, :] = plane
        else:
            for c in range(C):
                for z in range(Z):
                    corrected_video[t, c, z, :, :] = sr.transform(
                        video[t, c, z, :, :], tmats[t]
                    )

    return corrected_video, tmats


def drift_correct_xy_pygpureg(
    video: np.ndarray,
    drift_correct_channel: int = 0,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    if logger is None:
        import logging
        logger = logging
    # --- Input validation and shape normalization ---
    arr = video
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input video must be a numpy ndarray, got {type(arr)}")
    if arr.ndim < 5:
        arr = arr[(None,) * (5 - arr.ndim)]
    if arr.ndim != 5:
        raise ValueError(f"Input video must be 5D (TCZYX), got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Input video must have a numeric dtype, got {arr.dtype}")
    T, C, Z, _, _ = arr.shape
    if not (0 <= drift_correct_channel < C):
        raise ValueError(f"drift_correct_channel {drift_correct_channel} is out of bounds for {C} channels.")
    video = arr
    """
    GPU alternative using pyGPUreg if available.
    Strategy:
    - Build a template from the mean of the Z-max projections (across time) of the drift channel.
    - For each timepoint, register the timepoint's Z-max projection to the template on GPU to estimate shift.
    - Apply registration per-plane (C,Z) for that timepoint by calling register with apply_shift=True.

    Returns: corrected video and an array of shifts per timepoint (Tx2) if available, else None.
    """
    errors = []
    try:
        import pyGPUreg as reg  # type: ignore
    except Exception as e:
        logger.warning(f"pyGPUreg not available: {e}. Returning original video WITHOUT drift correction.")
        return video, None, [{'type': 'import_failed', 'error': str(e), 'message': 'No drift correction applied - original video returned'}]

    T, C, Z, Y, X = video.shape
    corrected = np.zeros_like(video)

    # Reference stack (T, Y, X) and template
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)
    template = ref_stack.mean(axis=0).astype(np.float32)

    # Use a Hann window ONLY for estimating shifts (frequency-domain registration)
    # to reduce edge artifacts. We will apply the resulting shifts to the ORIGINAL
    # (unwindowed) planes to avoid visible vignetting in outputs.
    def _hann2d(h: int, w: int) -> np.ndarray:
        if h <= 1 or w <= 1:
            return np.ones((h, w), dtype=np.float32)
        y = np.hanning(h).astype(np.float32)
        x = np.hanning(w).astype(np.float32)
        return np.outer(y, x)

    win = _hann2d(Y, X)
    template_win = (template * win).astype(np.float32)

    # Helper: next power of two (required by pyGPUreg)
    def _next_pow2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    # Initialize pyGPUreg with nearest pow2
    size = _next_pow2(max(Y, X))
    try:
        reg.init()
    except Exception as e:
        logger.warning(f"pyGPUreg init failed for size {size}: {e}. Skipping GPU drift correction.")
        return video, None, [{"type": "init_failed", "error": str(e)}]

    # Optional enums (fallback to library defaults if unavailable)
    # Prefer more accurate options when available
    SUBPIX = (
        getattr(reg, "SUBPIXEL_MODE_PARABOLIC", None)
        if hasattr(reg, "SUBPIXEL_MODE_PARABOLIC")
        else getattr(reg, "SUBPIXEL_MODE_COM", None)
    )
    EDGE = (
        getattr(reg, "EDGE_MODE_MIRROR", None)
        if hasattr(reg, "EDGE_MODE_MIRROR")
        else getattr(reg, "EDGE_MODE_CLAMP", None)
        if hasattr(reg, "EDGE_MODE_CLAMP")
        else getattr(reg, "EDGE_MODE_ZERO", None)
    )
    INTERP = (
        getattr(reg, "INTERPOLATION_MODE_CUBIC", None)
        if hasattr(reg, "INTERPOLATION_MODE_CUBIC")
        else getattr(reg, "INTERPOLATION_MODE_LINEAR", None)
    )

    shifts = np.zeros((T, 2), dtype=np.float32)
    
    # Pre-pad the template once
    def _pad_center(img2d: np.ndarray, size: int) -> Tuple[np.ndarray, int, int]:
        h, w = img2d.shape
        pad_img = np.zeros((size, size), dtype=np.float32)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        pad_img[y0:y0 + h, x0:x0 + w] = img2d.astype(np.float32)
        return pad_img, y0, x0

    def _crop_center(img2d: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return img2d[y0:y0 + h, x0:x0 + w]

    # Window before padding
    template_pad, ty0, tx0 = _pad_center(template_win, size)

    # Prefer using a direct apply_shift if provided by pyGPUreg
    apply_shift_fn = None
    for name in ("apply_shift", "shift", "applyTranslation"):
        if hasattr(reg, name):
            apply_shift_fn = getattr(reg, name)
            break

    try:
        for t in range(T):
            ref_img = (ref_stack[t].astype(np.float32) * win)
            ref_pad, ry0, rx0 = _pad_center(ref_img, size)
            # Estimate one shift per timepoint from ref to template
            try:
                reg_kwargs: dict[str, Any] = {"apply_shift": True}
                if isinstance(SUBPIX, int):
                    reg_kwargs["subpixel_mode"] = SUBPIX
                if isinstance(EDGE, int):
                    reg_kwargs["edge_mode"] = EDGE
                if isinstance(INTERP, int):
                    reg_kwargs["interpolation_mode"] = INTERP
                _, shift = reg.register(template_pad, ref_pad, **reg_kwargs)
                if isinstance(shift, (list, tuple, np.ndarray)) and len(shift) >= 2:
                    shifts[t, 0] = float(shift[0])
                    shifts[t, 1] = float(shift[1])
            except Exception as e:
                logger.warning(f"pyGPUreg shift estimate failed at t={t}: {e}")
                errors.append({"type": "shift_estimate", "t": t, "error": str(e)})

            for c in range(C):
                for z in range(Z):
                    plane = video[t, c, z].astype(np.float32)
                    plane_pad, py0, px0 = _pad_center(plane, size)
                    try:
                        reg_plane = None
                        if apply_shift_fn is not None:
                            try:
                                if (shifts[t, 0] == 0.0 and shifts[t, 1] == 0.0):
                                    reg_plane = plane_pad
                                else:
                                    kwargs2: dict[str, Any] = {}
                                    if isinstance(EDGE, int):
                                        kwargs2["edge_mode"] = EDGE
                                    if isinstance(INTERP, int):
                                        kwargs2["interpolation_mode"] = INTERP
                                    reg_plane = apply_shift_fn(
                                        plane_pad,
                                        (float(shifts[t, 0]), float(shifts[t, 1])),
                                        **kwargs2,
                                    )
                            except Exception as e:
                                reg_plane = None
                                errors.append({"type": "apply_shift_fn", "t": t, "c": c, "z": z, "error": str(e)})
                        if reg_plane is None:
                            try:
                                reg_kwargs: dict[str, Any] = {"apply_shift": True}
                                if isinstance(SUBPIX, int):
                                    reg_kwargs["subpixel_mode"] = SUBPIX
                                if isinstance(EDGE, int):
                                    reg_kwargs["edge_mode"] = EDGE
                                if isinstance(INTERP, int):
                                    reg_kwargs["interpolation_mode"] = INTERP
                                reg_plane, _ = reg.register(template_pad, plane_pad, **reg_kwargs)
                            except Exception as e:
                                errors.append({"type": "register_fallback", "t": t, "c": c, "z": z, "error": str(e)})
                                reg_plane = plane_pad
                        reg_plane = np.asarray(reg_plane, dtype=np.float32)
                        reg_plane = _crop_center(reg_plane, py0, px0, Y, X)
                        if np.issubdtype(video.dtype, np.integer):
                            reg_plane = np.clip(reg_plane, np.iinfo(video.dtype).min, np.iinfo(video.dtype).max)
                            corrected[t, c, z] = reg_plane.astype(video.dtype)
                        else:
                            corrected[t, c, z] = reg_plane.astype(video.dtype)
                    except Exception as e:
                        logger.debug(f"Plane apply failed at t={t}, c={c}, z={z}: {e}")
                        errors.append({"type": "plane_apply", "t": t, "c": c, "z": z, "error": str(e)})
                        corrected[t, c, z] = video[t, c, z]
        return corrected, shifts, errors if errors else None
    except Exception as e:
        logger.error(f"pyGPUreg path failed ({e}); returning original frames.")
        return video, None, [{"type": "fatal", "error": str(e)}]

def phase_cross_correlation_cupy(a, b, upsample_factor=1):
    """
    Pure-CuPy phase correlation.
    Returns: (shift_yx, error, global_phase)
    SHIFT IS THE AMOUNT TO APPLY TO `b` TO ALIGN IT TO `a` (skimage convention).
    


    Suggested parameter presets
    “Go-fast”: downsample=4, refine_upsample=10, highpass_pix=3, lowpass_pix=96
    “Balanced”: downsample=2, refine_upsample=20, highpass_pix=3, lowpass_pix=128
    “Max accuracy”: downsample=1, refine_upsample=50, highpass_pix=None, lowpass_pix=None
    """
    import cupy as cp

    a = a.astype(cp.float32, copy=False)
    b = b.astype(cp.float32, copy=False)
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
        return (cp.asarray([float(py), float(px)]), 1.0 - float(r.max()), 0.0)

    # --- Enhanced subpixel estimation using 5x5 neighborhood for better accuracy ---
    def _improved_subpixel_fit(r_local, center_y, center_x, H, W):
        """
        Improved subpixel fitting using larger neighborhood and weighted fitting
        """
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
    
    return (cp.asarray([dy_sub, dx_sub]), err, 0.0)


def drift_correct_xy_cupy(
    video: np.ndarray,
    *,
    drift_correct_channel: int = 0,
    downsample: int = 1,
    refine_upsample: int = 50,
    use_running_ref: bool = True,
    highpass_pix: Optional[float] = None,
    lowpass_pix: Optional[float] = None,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    if logger is None:
        import logging
        logger = logging
    # --- Input validation and shape normalization ---
    arr = video
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input video must be a numpy ndarray, got {type(arr)}")
    if arr.ndim < 5:
        arr = arr[(None,) * (5 - arr.ndim)]
    if arr.ndim != 5:
        raise ValueError(f"Input video must be 5D (TCZYX), got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Input video must have a numeric dtype, got {arr.dtype}")
    T, C, Z, _, _ = arr.shape
    if not (0 <= drift_correct_channel < C):
        raise ValueError(f"drift_correct_channel {drift_correct_channel} is out of bounds for {C} channels.")
    video = arr
    """
    Drift-correct using CuPy phase correlation on Z-max projections.

    Parameters
    - video: numpy array with shape (T, C, Z, Y, X)
    - drift_correct_channel: channel used to estimate shifts
    - downsample: spatial downsample factor for speed (1, 2, 4, ...)
    - refine_upsample: subpixel refinement factor passed to phase correlation (>=1)
    - use_running_ref: if True, estimate incremental shifts to previous frame and accumulate
                       else, estimate to a single mean template
    - highpass_pix / lowpass_pix: optional Gaussian HP/LP filtering (in pixels) for robustness

    Returns
    - corrected video (numpy array, same dtype/shape)
    - shifts (numpy float32 array of shape (T, 2)) or None if CuPy path unavailable
    """
    errors = []
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy import ndimage as cndi  # type: ignore
    except Exception as e:
        logger.warning(f"CuPy/cuCIM not available: {e}. Returning original video WITHOUT drift correction.")
        return video, None, [{'type': 'import_failed', 'error': str(e), 'message': 'No drift correction applied - original video returned'}]

    try:
        T, C, Z, Y, X = video.shape
        corrected = np.zeros_like(video)

        ref_stack_np = np.max(video[:, drift_correct_channel, :, :, :], axis=1).astype(np.float32, copy=False)

        def _preprocess_2d(img2d_cp: "cp.ndarray") -> "cp.ndarray":
            x = cp.asarray(img2d_cp, dtype=cp.float32)
            if highpass_pix is not None and float(highpass_pix) > 0.0:
                x = x - cndi.gaussian_filter(x, sigma=float(highpass_pix))
            if lowpass_pix is not None and float(lowpass_pix) > 0.0:
                x = cndi.gaussian_filter(x, sigma=float(lowpass_pix))
            ds = int(downsample) if isinstance(downsample, int) else 1
            if ds > 1:
                x = x[::ds, ::ds]  # type: ignore[index]
            return x

        shifts = np.zeros((T, 2), dtype=np.float32)

        if use_running_ref:
            prev_proc = _preprocess_2d(cp.asarray(ref_stack_np[0]))
            cum = np.array([0.0, 0.0], dtype=np.float32)
            shifts[0] = cum
            for t in range(1, T):
                cur_proc = _preprocess_2d(cp.asarray(ref_stack_np[t]))
                try:
                    shift_yx, err, _ = phase_cross_correlation_cupy(prev_proc, cur_proc, upsample_factor=max(1, int(refine_upsample)))
                    sh = cp.asnumpy(shift_yx).astype(np.float32) * float(max(1, downsample))
                except Exception as e:
                    logger.debug(f"CuPy phase corr failed at t={t}: {e}")
                    errors.append({"type": "phase_corr", "t": t, "error": str(e)})
                    sh = np.array([0.0, 0.0], dtype=np.float32)
                cum += sh
                shifts[t] = cum
                prev_proc = cur_proc
        else:
            template_np = ref_stack_np.mean(axis=0)
            template_proc = _preprocess_2d(cp.asarray(template_np))
            for t in range(T):
                cur_proc = _preprocess_2d(cp.asarray(ref_stack_np[t]))
                try:
                    shift_yx, err, _ = phase_cross_correlation_cupy(template_proc, cur_proc, upsample_factor=max(1, int(refine_upsample)))
                    sh = cp.asnumpy(shift_yx).astype(np.float32) * float(max(1, downsample))
                except Exception as e:
                    logger.debug(f"CuPy phase corr failed at t={t}: {e}")
                    errors.append({"type": "phase_corr", "t": t, "error": str(e)})
                    sh = np.array([0.0, 0.0], dtype=np.float32)
                shifts[t] = sh

        mode = 'reflect'
        order = 3
        for t in range(T):
            dy, dx = float(shifts[t, 0]), float(shifts[t, 1])
            do_shift = not (abs(dy) < 1e-6 and abs(dx) < 1e-6)
            for c in range(C):
                for z in range(Z):
                    plane_np = video[t, c, z]
                    if do_shift:
                        plane_cp = cp.asarray(plane_np, dtype=cp.float32)
                        try:
                            shifted_cp = cndi.shift(plane_cp, shift=(dy, dx), order=order, mode=mode, prefilter=True)
                            out_np = cp.asnumpy(shifted_cp)
                        except Exception as e:
                            logger.debug(f"CuPy apply shift failed at t={t}, c={c}, z={z}: {e}")
                            errors.append({"type": "apply_shift", "t": t, "c": c, "z": z, "error": str(e)})
                            out_np = plane_np.astype(np.float32, copy=False)
                    else:
                        out_np = plane_np.astype(np.float32, copy=False)

                    if np.issubdtype(video.dtype, np.integer):
                        info = np.iinfo(video.dtype)
                        out_np = np.clip(out_np, info.min, info.max).astype(video.dtype, copy=False)
                    else:
                        out_np = out_np.astype(video.dtype, copy=False)
                    corrected[t, c, z] = out_np

        # Release GPU memory after processing
        try:
            mempool = cp.get_default_memory_pool()
            freed = mempool.used_bytes()
            mempool.free_bytes()
            logger.info(f"CuPy memory pool released {freed} bytes after drift correction.")
        except Exception as e:
            logger.debug(f"Could not release CuPy memory: {e}")
        # Validate drift correction quality
        validation_info = _validate_drift_correction(ref_stack_np, shifts)
        if errors:
            errors.extend(validation_info.get('warnings', []))
        else:
            errors = validation_info.get('warnings', None)
            
        # Calculate drift correction quality using pixel variance analysis
        try:
            quality_metrics = evaluate_drift_correction_quality(
                video, corrected, drift_correct_channel, return_details=True
            )
            validation_info['pixel_variance_quality'] = quality_metrics
            
            logger.info(f"CuPy drift correction completed. Max shift: {validation_info['max_shift']:.2f}px, "
                       f"Mean shift: {validation_info['mean_shift']:.2f}px, "
                       f"Shift quality: {validation_info['quality_score']:.3f}, "
                       f"Drift quality: {quality_metrics['drift_quality']:.2f} "
                       f"({quality_metrics['quality_interpretation']['drift_quality_category']})")
            
            if validation_info['quality_score'] < 0.5:
                logger.warning("Poor shift pattern detected. Consider using CPU method or different parameters.")
            if quality_metrics['drift_quality'] < 1.1:
                logger.warning(f"Limited drift correction improvement detected (quality: {quality_metrics['drift_quality']:.2f}). "
                              f"{quality_metrics['quality_interpretation']['recommendation']}")
        except Exception as e:
            logger.warning(f"Could not calculate drift quality metrics: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            logger.info(f"CuPy drift correction completed. Max shift: {validation_info['max_shift']:.2f}px, "
                       f"Mean shift: {validation_info['mean_shift']:.2f}px, "
                       f"Shift quality: {validation_info['quality_score']:.3f} (pixel variance analysis failed)")
        
        return corrected, shifts, errors if errors else None
    except Exception as e:
        logger.error(f"CuPy drift correction path failed ({e}); returning original frames.")
        return video, None, [{"type": "fatal", "error": str(e)}]

def evaluate_drift_correction_quality(
    input_video: np.ndarray,           # Original TCZYX
    corrected_video: np.ndarray,       # Drift-corrected TCZYX  
    drift_channel: int = 0,            # Channel used for drift estimation
    method: str = "variance",          # "variance", "mad", "iqr"
    spatial_weight: str = "gradient",  # "uniform", "gradient", "center"
    exclude_background: bool = True,   # Mask low-intensity regions
    patch_size: Optional[int] = None,  # For patch-based analysis
    return_details: bool = False       # Return breakdown of metrics
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


def _validate_drift_correction(ref_stack: np.ndarray, shifts: np.ndarray) -> Dict[str, Any]:
    """
    Validate the quality of drift correction by analyzing shift patterns and consistency.
    
    Returns dict with quality metrics and warnings.
    """
    validation_info = {
        'warnings': [],
        'max_shift': 0.0,
        'mean_shift': 0.0,
        'quality_score': 1.0
    }
    
    if shifts is None or len(shifts) == 0:
        validation_info['warnings'].append({
            'type': 'no_shifts', 
            'message': 'No shift data available for validation'
        })
        validation_info['quality_score'] = 0.0
        return validation_info
    
    # Calculate shift magnitudes
    shift_magnitudes = np.sqrt(np.sum(shifts**2, axis=1))
    validation_info['max_shift'] = float(np.max(shift_magnitudes))
    validation_info['mean_shift'] = float(np.mean(shift_magnitudes))
    
    # Check for unreasonably large shifts
    if validation_info['max_shift'] > min(ref_stack.shape[-2:]) * 0.25:  # >25% of image size
        validation_info['warnings'].append({
            'type': 'large_shifts',
            'message': f"Very large shifts detected (max: {validation_info['max_shift']:.1f}px). Results may be unreliable."
        })
        validation_info['quality_score'] *= 0.3
    
    # Check for sudden jumps in shifts (indicating potential failures)
    if len(shifts) > 1:
        shift_diffs = np.diff(shift_magnitudes)
        max_jump = float(np.max(np.abs(shift_diffs)))
        if max_jump > validation_info['mean_shift'] * 3 and max_jump > 5.0:
            validation_info['warnings'].append({
                'type': 'sudden_jumps',
                'message': f"Sudden shift jumps detected (max: {max_jump:.1f}px). Registration may have failed for some frames."
            })
            validation_info['quality_score'] *= 0.5
    
    # Check shift consistency (should be relatively smooth for real drift)
    if len(shifts) > 5:
        smoothness = np.std(shift_diffs) / (validation_info['mean_shift'] + 0.1)
        if smoothness > 2.0:
            validation_info['warnings'].append({
                'type': 'irregular_shifts',
                'message': f"Irregular shift pattern detected (smoothness: {smoothness:.2f}). Consider different parameters."
            })
            validation_info['quality_score'] *= 0.7
    
    return validation_info


def load_first_T_timepoints(input_file: str, max_T: Optional[int] = None) -> np.ndarray:
    """
    Loads the image via rp.load_tczyx_image and returns the first max_T timepoints as a numpy array.
    Assumes shape TCZYX. If data is a Dask array, compute only the needed slice.
    """
    img = rp.load_tczyx_image(input_file)
    if img is None:
        raise RuntimeError(f"Could not load image: {input_file}")

    data = img.data
    try:
        T = int(data.shape[0])
    except Exception as e:
        raise RuntimeError(f"Unexpected data shape for {input_file}: {getattr(data, 'shape', None)}; error: {e}")

    tsel = T if max_T is None else min(max_T, T)
    subset = data[:tsel, ...]
    compute_fn = getattr(subset, 'compute', None)
    if callable(compute_fn):
        try:
            subset_np = compute_fn()
        except Exception:
            subset_np = np.asarray(subset)
    else:
        subset_np = np.asarray(subset)
    # Guarantee a NumPy ndarray type for downstream logic
    subset_np = np.asarray(subset_np)
    if subset_np.ndim != 5:
        raise RuntimeError(f"Expected 5D data (TCZYX), got shape {subset_np.shape}")
    return subset_np


def correct_image(
    *,
    input_path: str,
    output_dir: Optional[str] = None,
    drift_channel: int = 0,
    method: str = "auto",
    timepoints: Optional[int] = None,
    cpu_threads: int = 0,
    save_shifts: bool = True,
    cupy_preset: Optional[str] = None,
    cupy_downsample: Optional[int] = None,
    cupy_refine_upsample: Optional[int] = None,
    cupy_highpass_pix: Optional[float] = None,
    cupy_lowpass_pix: Optional[float] = None,
    cupy_static_template: bool = False,
) -> Dict[str, Any]:
    """
    Correct XY drift on a single image file.

    Parameters:
    - input_path: path to image to correct (expects TCZYX)
    - output_dir: where to write outputs (defaults next to input)
    - drift_channel: channel index to estimate drift from
    - method: 'cpu' | 'gpu' | 'cupy' | 'auto' (gpu/cupy then fallback to cpu)
    - timepoints: if provided, limit to first T timepoints
    - cpu_threads: per-plane transform threads for CPU implementation
    - save_shifts: save tmats/shifts npy alongside TIFF
    - cupy_*: parameters/preset to tweak CuPy path

    Returns dict with output paths and basic metadata.
    """
    t0 = time.perf_counter()
    data = load_first_T_timepoints(input_path, max_T=timepoints)
    T, C, Z, Y, X = data.shape
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = output_dir or os.path.join(os.path.dirname(input_path), "_output")
    os.makedirs(out_dir, exist_ok=True)

    chosen = method.lower()
    corrected: Optional[np.ndarray] = None
    info: Dict[str, Any] = {}
    npy_path = None
    eff: Optional[Dict[str, Any]] = None

    if chosen not in {"cupy", "cpu", "gpu", "auto"}:
        raise ValueError("method must be one of 'cpu', 'gpu', 'cupy', or 'auto'")

    def _save_outputs(arr: np.ndarray, suffix: str) -> str:
        out_tif = os.path.join(out_dir, f"{base}_T{T}_drift_{suffix}.tif")
        rp.save_tczyx_image(arr, out_tif, dim_order="TCZYX")
        return out_tif

    # Try cupy first, then gpu then cpu if requested/auto
    # gpu_failed = False
    gpu_used = False
    
    if chosen in {"cupy", "auto"} and not info.get("method"):  # try GPU CuPy if auto
        eff = _resolve_cupy_params(
            preset=cupy_preset,
            downsample=cupy_downsample,
            refine_upsample=cupy_refine_upsample,
            highpass_pix=cupy_highpass_pix,
            lowpass_pix=cupy_lowpass_pix,
            use_running_ref=not bool(cupy_static_template),
        )
        logger.info(
            "Attempting GPU drift correction (CuPy/cuCIM) with: preset=%s, downsample=%s, refine_upsample=%s, highpass=%s, lowpass=%s, running_ref=%s",
            cupy_preset or "(none)", eff["downsample"], eff["refine_upsample"], eff["highpass_pix"], eff["lowpass_pix"], eff["use_running_ref"],
        )
        corrected_gpu, gpu_shifts, gpu_errors = drift_correct_xy_cupy(
            data,
            drift_correct_channel=drift_channel,
            downsample=int(eff["downsample"]),
            refine_upsample=int(eff["refine_upsample"]),
            use_running_ref=bool(eff["use_running_ref"]),
            highpass_pix=eff["highpass_pix"],
            lowpass_pix=eff["lowpass_pix"],
        )
        if gpu_shifts is not None:
            corrected = corrected_gpu
            out_tif = _save_outputs(corrected, "CuPy")
            info["output_tif"] = out_tif
            if save_shifts:
                npy_path = os.path.join(out_dir, f"{base}_T{T}_drift_CuPy_shifts.npy")
                np.save(npy_path, gpu_shifts)
                info["shifts_npy"] = npy_path
            if gpu_errors:
                info["drift_errors"] = gpu_errors
                # Check for critical errors that indicate failure
                critical_errors = [e for e in gpu_errors if e.get('type') in ['import_failed', 'fatal']]
                if critical_errors:
                    logger.error(f"CuPy drift correction had critical errors: {critical_errors}")
            info["method"] = "cupy"
            info["time_sec_gpu_cupy"] = time.perf_counter() - t0
            logger.info(f"✅ CuPy drift correction completed successfully in {info['time_sec_gpu_cupy']:.2f}s")
            gpu_used = True  
        else:
            if chosen == "cupy":
                logger.error("❌ CuPy method specifically requested but failed. No fallback attempted.")
                raise RuntimeError("CuPy drift correction failed and no fallback allowed for method='cupy'")
            else:
                logger.warning("⚠️ CuPy/cuCIM path unavailable; will attempt CPU fallback.")

    elif chosen in {"gpu", "auto"} and not gpu_used:
        logger.info("Attempting GPU drift correction (pyGPUreg)...")
        corrected_gpu, gpu_shifts, gpu_errors = drift_correct_xy_pygpureg(data, drift_correct_channel=drift_channel)
        if gpu_shifts is None:
            gpu_failed = True
            logger.info("GPU drift correction unavailable or failed; will fallback if allowed.")
        else:
            gpu_used = True
            corrected = corrected_gpu
            out_tif = _save_outputs(corrected, "GPU")
            info["output_tif"] = out_tif
            if save_shifts:
                npy_path = os.path.join(out_dir, f"{base}_T{T}_drift_GPU_shifts.npy")
                np.save(npy_path, gpu_shifts)
                info["shifts_npy"] = npy_path
            if gpu_errors:
                info["drift_errors"] = gpu_errors
            info["method"] = "gpu"
            info["time_sec_gpu"] = time.perf_counter() - t0

    # CPU path if requested or GPU failed in auto
    if (chosen == "cpu") or (chosen == "auto" and not gpu_used):
        if chosen == "auto" and not gpu_used:
            logger.info("🔄 GPU methods failed/unavailable, falling back to CPU drift correction (pystackreg)...")
        else:
            logger.info("Running CPU drift correction (pystackreg)...")
        t1 = time.perf_counter()
        corrected_cpu, tmats = drift_correct_xy_parallel(
            data, drift_correct_channel=drift_channel, transform_threads=cpu_threads
        )
        corrected = corrected_cpu
        out_tif = _save_outputs(corrected, "CPU")
        info["output_tif"] = out_tif
        if save_shifts:
            npy_path = os.path.join(out_dir, f"{base}_T{T}_drift_CPU_tmats.npy")
            np.save(npy_path, tmats)
            info["tmats_npy"] = npy_path
        info["method"] = "cpu" if not gpu_used else info.get("method", "cpu")
        info["time_sec_cpu"] = time.perf_counter() - t1
        logger.info(f"✅ CPU drift correction completed successfully in {info['time_sec_cpu']:.2f}s")
    
    # Final validation check
    if not info.get("output_tif"):
        logger.error("❌ No drift correction method succeeded!")
        raise RuntimeError("All drift correction methods failed")

    # Calculate comprehensive drift correction quality for final report
    if corrected is not None:
        try:
            final_quality = evaluate_drift_correction_quality(
                data, corrected, drift_channel, return_details=True
            )
            info["drift_quality_analysis"] = final_quality
            
            logger.info(f"📊 Drift Quality Analysis:")
            logger.info(f"   • Drift Quality: {final_quality['drift_quality']:.2f} ({final_quality['quality_interpretation']['drift_quality_category']})")
            logger.info(f"   • Edge Sharpness: {final_quality['edge_sharpness_improvement']:.2f}x improvement")
            logger.info(f"   • Temporal Consistency: {final_quality['temporal_consistency_improvement']:.2f}x improvement")
            logger.info(f"   • Composite Score: {final_quality['composite_score']:.2f}")
            logger.info(f"   • Recommendation: {final_quality['quality_interpretation']['recommendation']}")
            
        except Exception as e:
            logger.warning(f"Could not calculate final drift quality analysis: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.warning("No corrected video available for quality analysis")

    info.update({
        "input_path": input_path,
        "output_dir": out_dir,
        "T": T,
        "shape": (T, C, Z, Y, X),
        "drift_channel": drift_channel,
        "cupy_effective": eff if chosen in {"cupy", "auto"} else None,
    })
    return info


def correct_directory(
    *,
    input_search_pattern: str,
    output_dir: Optional[str] = None,
    drift_channel: int = 0,
    method: str = "auto",
    timepoints: Optional[int] = None,
    cpu_threads: int = 0,
    save_shifts: bool = True,
    parallel: bool = True,
    # CuPy options
    cupy_preset: Optional[str] = None,
    cupy_downsample: Optional[int] = None,
    cupy_refine_upsample: Optional[int] = None,
    cupy_highpass_pix: Optional[float] = None,
    cupy_lowpass_pix: Optional[float] = None,
    cupy_static_template: bool = False,
) -> Iterable[Dict[str, Any]]:
    """
    Batch drift correction over a glob search pattern.
    Follows the organization pattern of segment_nellie.py.
    """
    import glob as _glob

    image_files = sorted(_glob.glob(input_search_pattern))
    # If wildcard used, optionally filter to common tif/tiff
    if any(ch in input_search_pattern for ch in ["*", "?", "["]):
        image_files = [p for p in image_files if p.lower().endswith((".tif", ".tiff"))]

    is_batch = len(image_files) > 1
    if output_dir is None and is_batch:
        output_dir = os.path.dirname(input_search_pattern)
    if is_batch and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _one(img_path: str) -> Dict[str, Any]:
        try:
            return correct_image(
                input_path=img_path,
                output_dir=output_dir,
                drift_channel=drift_channel,
                method=method,
                timepoints=timepoints,
                cpu_threads=cpu_threads,
                save_shifts=save_shifts,
                cupy_preset=cupy_preset,
                cupy_downsample=cupy_downsample,
                cupy_refine_upsample=cupy_refine_upsample,
                cupy_highpass_pix=cupy_highpass_pix,
                cupy_lowpass_pix=cupy_lowpass_pix,
                cupy_static_template=cupy_static_template,
            )
        except Exception as e:
            logger.error(f"Failed to correct {img_path}: {e}")
            return {"input_path": img_path, "error": str(e)}

    if parallel and is_batch:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(_one, p) for p in image_files]
            for f in tqdm(as_completed(futures), total=len(image_files), desc="Drift correcting"):
                yield f.result()
    else:
        for p in tqdm(image_files, desc="Drift correcting"):
            yield _one(p)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct XY drift using CuPy/cuCIM, pyGPUreg, or CPU (pystackreg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern or single file path.')
    parser.add_argument('--output-dir', default=None, help='Where to place outputs')
    parser.add_argument('--drift-channel', type=int, default=0, help='Channel index used for drift estimation')
    parser.add_argument('--method', default='auto', choices=['cpu', 'gpu', 'cupy', 'auto'], help='Which method to use')
    parser.add_argument('--cpu-threads', type=int, default=0, help='Threads for per-plane transform in CPU baseline (0/1=serial)')
    parser.add_argument('--save-shifts', action='store_true', help='Save tmats/shifts npy next to outputs')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel batch processing')
    # CuPy tuning options
    parser.add_argument('--cupy-preset', default=None, choices=list(CUPY_PRESETS.keys()), help='CuPy parameter preset to use')
    parser.add_argument('--cupy-downsample', type=int, default=None, help='Override downsample factor (1,2,4,...)')
    parser.add_argument('--cupy-refine', dest='cupy_refine_upsample', type=int, default=None, help='Subpixel refinement factor (e.g., 10-50)')
    parser.add_argument('--cupy-highpass', dest='cupy_highpass_pix', type=float, default=None, help='Highpass sigma in pixels (None to disable)')
    parser.add_argument('--cupy-lowpass', dest='cupy_lowpass_pix', type=float, default=None, help='Lowpass sigma in pixels (None to disable)')
    parser.add_argument('--cupy-static-template', action='store_true', help='Use static average template instead of running reference')
    parser.add_argument('--log-level', default=None, help='Explicit logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
    args = parser.parse_args(argv)

    # Logging setup: use provided --log-level or env DRIFT_CORRECT_LOG_LEVEL (default INFO)
    env_level = os.getenv('DRIFT_CORRECT_LOG_LEVEL')
    chosen = (args.log_level or env_level or 'INFO').upper()
    level = getattr(logging, chosen, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.debug(f"Logging initialized. level={logging.getLevelName(level)}")

    # Batch runner
    results = []
    for res in correct_directory(
        input_search_pattern=args.input_search_pattern,
        output_dir=args.output_dir,
        drift_channel=args.drift_channel,
        method=args.method,
        cpu_threads=args.cpu_threads,
        save_shifts=args.save_shifts,
        parallel=not args.no_parallel,
        # pass CuPy options
        cupy_preset=args.cupy_preset,
        cupy_downsample=args.cupy_downsample,
        cupy_refine_upsample=args.cupy_refine_upsample,
        cupy_highpass_pix=args.cupy_highpass_pix,
        cupy_lowpass_pix=args.cupy_lowpass_pix,
        cupy_static_template=args.cupy_static_template,
    ):
        results.append(res)
        if 'error' in res:
            logger.error(f"Error: {res['error']} ({res.get('input_path')})")
        else:
            logger.info(f"Done: {res.get('output_tif')} ({res.get('method')})")

    return 0


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
    # Use the same helper functions import as the rest of the module
    # import run_pipeline_helper_functions as rp  # Already imported as rp at module level
    
    # Load both videos
    logger.info(f"Loading input video: {input_file}")
    input_video = rp.load_bioio(input_file)
    
    logger.info(f"Loading corrected video: {corrected_file}")
    corrected_video = rp.load_bioio(corrected_file)
    
    # Evaluate quality - always request details for proper reporting
    logger.info("Evaluating drift correction quality...")
    quality_results = evaluate_drift_correction_quality(
        input_video, corrected_video, drift_channel, return_details=True
    )
    
    if show_details and isinstance(quality_results, dict):
        logger.info("📊 Drift Correction Quality Results:")
        logger.info(f"   • Drift Quality: {quality_results['drift_quality']:.3f}")
        logger.info(f"   • Category: {quality_results['quality_interpretation']['drift_quality_category']}")
        logger.info(f"   • Edge Sharpness Improvement: {quality_results['edge_sharpness_improvement']:.3f}")
        logger.info(f"   • Temporal Consistency: {quality_results['temporal_consistency_improvement']:.3f}")
        logger.info(f"   • Composite Score: {quality_results['composite_score']:.3f}")
        logger.info(f"   • Recommendation: {quality_results['quality_interpretation']['recommendation']}")
        logger.info(f"   • Pixels analyzed: {quality_results['total_pixels_analyzed']}")
        
        if quality_results.get('patch_analysis'):
            patch_qualities = [p['quality_ratio'] for p in quality_results['patch_analysis']]
            logger.info(f"   • Patch analysis: {len(patch_qualities)} patches, "
                       f"quality range [{min(patch_qualities):.2f}, {max(patch_qualities):.2f}]")
    elif isinstance(quality_results, (int, float)):
        logger.info(f"📊 Drift Quality: {quality_results:.3f}")
    
    return quality_results


if __name__ == "__main__":
    raise SystemExit(main())
