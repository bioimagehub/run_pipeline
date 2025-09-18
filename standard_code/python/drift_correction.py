
import argparse
import os
import time
from typing import Optional, Tuple, Any

import numpy as np
from joblib import Parallel, delayed

# Local helper used throughout the repo
import run_pipeline_helper_functions as rp

# CPU registration (StackReg)
from pystackreg import StackReg
from bioio.writers import OmeTiffWriter


def drift_correct_xy_parallel(
    video: np.ndarray,
    drift_correct_channel: int = 0,
    transform_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
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
            results = Parallel(n_jobs=transform_threads, prefer="threads")(
                delayed(_transform_one)(a) for a in tasks
            )
            for c, z, plane in results:
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    GPU alternative using pyGPUreg if available.
    Strategy:
    - Build a template from the mean of the Z-max projections (across time) of the drift channel.
    - For each timepoint, register the timepoint's Z-max projection to the template on GPU to estimate shift.
    - Apply registration per-plane (C,Z) for that timepoint by calling register with apply_shift=True.

    Returns: corrected video and an array of shifts per timepoint (Tx2) if available, else None.
    """
    try:
        import pyGPUreg as reg  # type: ignore
    except Exception as e:
        print(f"pyGPUreg not available: {e}. Skipping GPU drift correction.")
        return video, None

    T, C, Z, Y, X = video.shape
    corrected = np.zeros_like(video)

    # Reference stack (T, Y, X) and template
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)
    template = ref_stack.mean(axis=0).astype(np.float32)

    # Use a Hann window on images prior to FFT-based registration to reduce edge artifacts.
    # This tends to improve peak localization and match CPU (StackReg) behavior better.
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
        print(f"pyGPUreg init failed for size {size}: {e}. Skipping GPU drift correction.")
        return video, None

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
                # Register ref to template (returns ref aligned to template and the (dy,dx) shift)
                _, shift = reg.register(template_pad, ref_pad, **reg_kwargs)
                # shift is expected as (dy, dx) or similar
                if isinstance(shift, (list, tuple, np.ndarray)) and len(shift) >= 2:
                    shifts[t, 0] = float(shift[0])
                    shifts[t, 1] = float(shift[1])
            except Exception as e:
                # If estimation fails, leave shift zeros and still try per-plane register
                print(f"pyGPUreg shift estimate failed at t={t}: {e}")

            # Apply the SAME estimated shift to every plane for this timepoint to match CPU behavior.
            for c in range(C):
                for z in range(Z):
                    # Windowing before padding improves accuracy near borders
                    plane = (video[t, c, z].astype(np.float32) * win)
                    plane_pad, py0, px0 = _pad_center(plane, size)
                    try:
                        # If we have a fast path to apply a known shift, use it.
                        reg_plane = None
                        if apply_shift_fn is not None and (shifts[t, 0] != 0.0 or shifts[t, 1] != 0.0):
                            try:
                                kwargs2: dict[str, Any] = {}
                                if isinstance(EDGE, int):
                                    kwargs2["edge_mode"] = EDGE
                                if isinstance(INTERP, int):
                                    kwargs2["interpolation_mode"] = INTERP
                                # Accept (dy, dx) as shift
                                reg_plane = apply_shift_fn(plane_pad, (float(shifts[t, 0]), float(shifts[t, 1])), **kwargs2)
                            except Exception:
                                reg_plane = None
                        if reg_plane is None:
                            # Fallback: re-register this plane (slower/less consistent)
                            reg_kwargs: dict[str, Any] = {"apply_shift": True}
                            if isinstance(SUBPIX, int):
                                reg_kwargs["subpixel_mode"] = SUBPIX
                            if isinstance(EDGE, int):
                                reg_kwargs["edge_mode"] = EDGE
                            if isinstance(INTERP, int):
                                reg_kwargs["interpolation_mode"] = INTERP
                            reg_plane, _ = reg.register(template_pad, plane_pad, **reg_kwargs)
                        # Ensure ndarray then crop back to original image size
                        reg_plane = np.asarray(reg_plane, dtype=np.float32)
                        reg_plane = _crop_center(reg_plane, py0, px0, Y, X)
                        # Cast back to original dtype if possible
                        if np.issubdtype(video.dtype, np.integer):
                            reg_plane = np.clip(reg_plane, np.iinfo(video.dtype).min, np.iinfo(video.dtype).max)
                            corrected[t, c, z] = reg_plane.astype(video.dtype)
                        else:
                            corrected[t, c, z] = reg_plane.astype(video.dtype)
                    except Exception as e:
                        # On failure, fall back to original plane
                        corrected[t, c, z] = video[t, c, z]
        return corrected, shifts
    except Exception as e:
        print(f"pyGPUreg path failed ({e}); returning original frames.")
        return video, None


def load_first_T_timepoints(input_file: str, max_T: int = 10) -> np.ndarray:
    """
    Loads the image via rp.load_bioio and returns the first max_T timepoints as a numpy array.
    Assumes shape TCZYX. If data is a Dask array, compute only the needed slice.
    """
    img = rp.load_bioio(input_file)
    if img is None:
        raise RuntimeError(f"Could not load image: {input_file}")

    data = img.data
    try:
        T = int(data.shape[0])
    except Exception as e:
        raise RuntimeError(f"Unexpected data shape for {input_file}: {getattr(data, 'shape', None)}; error: {e}")

    tsel = min(max_T, T)
    subset = data[:tsel, ...]
    compute_fn = getattr(subset, 'compute', None)
    if callable(compute_fn):
        try:
            subset_np = compute_fn()
        except Exception:
            subset_np = np.asarray(subset)
    else:
        subset_np = np.asarray(subset)
    if subset_np.ndim != 5:
        raise RuntimeError(f"Expected 5D data (TCZYX), got shape {subset_np.shape}")
    return subset_np


def main():
    parser = argparse.ArgumentParser(
        description="Time CPU (StackReg) vs GPU (pyGPUreg) drift correction on first T timepoints"
    )
    parser.add_argument("--input-file", type=str, default="", help="Path to input image file")
    parser.add_argument("--drift-channel", type=int, default=0, help="Channel index used for drift estimation")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Threads for per-plane transform in CPU baseline (0/1=serial)")
    parser.add_argument("--timepoints", type=int, default=10, help="Number of initial timepoints to test")
    args = parser.parse_args()

    input_file = args.input_file
    if not input_file:
        print("No input file provided (--input-file). Fill in the path to run the test.")
        return
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    print(f"Loading first {args.timepoints} timepoints from: {input_file}")
    data = load_first_T_timepoints(input_file, max_T=args.timepoints)
    print(f"Data subset shape: {data.shape}, dtype: {data.dtype}")

    # Prepare output folder and filenames
    out_dir = os.path.join(os.path.dirname(input_file), "_output")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]
    tsel = data.shape[0]
    cpu_tif = os.path.join(out_dir, f"{base}_T{tsel}_drift_CPU.tif")
    cpu_tmats_npy = os.path.join(out_dir, f"{base}_T{tsel}_drift_CPU_tmats.npy")
    gpu_shifts_npy = os.path.join(out_dir, f"{base}_T{tsel}_drift_GPU_shifts.npy")
    gpu_tif = os.path.join(out_dir, f"{base}_T{tsel}_drift_GPU.tif")

    # CPU timing
    t0 = time.perf_counter()
    cpu_out, cpu_tmats = drift_correct_xy_parallel(
        data, drift_correct_channel=args.drift_channel, transform_threads=args.cpu_threads
    )
    cpu_dt = time.perf_counter() - t0
    print(f"CPU (StackReg) time: {cpu_dt:.2f}s")
    # Save CPU outputs
    try:
        OmeTiffWriter.save(cpu_out, cpu_tif, dim_order="TCZYX")
        print(f"Saved CPU-corrected TIFF: {cpu_tif}")
    except Exception as e:
        print(f"Failed to save CPU TIFF: {e}")
    try:
        # tmats can be large but useful for inspection
        np.save(cpu_tmats_npy, cpu_tmats)
        print(f"Saved CPU transforms: {cpu_tmats_npy}")
    except Exception as e:
        print(f"Failed to save CPU transforms: {e}")

    # GPU timing (pyGPUreg)
    t1 = time.perf_counter()
    gpu_out, gpu_shifts = drift_correct_xy_pygpureg(
        data, drift_correct_channel=args.drift_channel
    )
    gpu_dt = time.perf_counter() - t1
    if gpu_out is data:
        print(f"GPU (pyGPUreg) skipped or placeholder. Elapsed: {gpu_dt:.2f}s")
    else:
        print(f"GPU (pyGPUreg) time: {gpu_dt:.2f}s")
    # Save GPU outputs (even if placeholder) for easy side-by-side inspection
    try:
        OmeTiffWriter.save(gpu_out, gpu_tif, dim_order="TCZYX")
        print(f"Saved GPU-corrected TIFF: {gpu_tif}")
    except Exception as e:
        print(f"Failed to save GPU TIFF: {e}")

    # Save GPU shifts and report accuracy vs CPU translation components
    try:
        if gpu_shifts is not None and cpu_tmats is not None:
            np.save(gpu_shifts_npy, gpu_shifts)
            print(f"Saved GPU shifts: {gpu_shifts_npy}")
            # Extract CPU (dy, dx) from 3x3 matrices per timepoint
            try:
                cpu_shifts = np.column_stack([cpu_tmats[:, 1, 2], cpu_tmats[:, 0, 2]])
                # Align shapes in case of slight T mismatch
                tmin = min(cpu_shifts.shape[0], gpu_shifts.shape[0])
                d = gpu_shifts[:tmin] - cpu_shifts[:tmin]
                rms = float(np.sqrt(np.mean(d**2)))
                mad = float(np.max(np.abs(d)))
                print(f"GPU vs CPU shift error: RMS={rms:.3f} px, MaxAbs={mad:.3f} px over T={tmin}")
            except Exception as e:
                print(f"Failed to compute CPU/GPU shift comparison: {e}")
    except Exception as e:
        print(f"Failed to save GPU shifts: {e}")


if __name__ == "__main__":
    main()
