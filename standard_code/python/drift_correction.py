
import argparse
import os
import time
import logging
from typing import Optional, Tuple, Any, Dict, Iterable, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Local helper used throughout the repo
import run_pipeline_helper_functions as rp

# CPU registration (StackReg)
from pystackreg import StackReg
from bioio.writers import OmeTiffWriter

# Module logger
logger = logging.getLogger(__name__)


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
            _res = Parallel(n_jobs=transform_threads, prefer="threads")(
                delayed(_transform_one)(a) for a in tasks
            )
            results_list = _res if isinstance(_res, list) else (list(_res) if _res is not None else [])
            for c, z, plane in (results_list or []):
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
        logger.info(f"pyGPUreg not available: {e}. Skipping GPU drift correction.")
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
        logger.warning(f"pyGPUreg init failed for size {size}: {e}. Skipping GPU drift correction.")
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
                logger.warning(f"pyGPUreg shift estimate failed at t={t}: {e}")

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
                        logger.debug(f"Plane apply failed at t={t}, c={c}, z={z}: {e}")
                        corrected[t, c, z] = video[t, c, z]
        return corrected, shifts
    except Exception as e:
        logger.error(f"pyGPUreg path failed ({e}); returning original frames.")
        return video, None


def load_first_T_timepoints(input_file: str, max_T: Optional[int] = None) -> np.ndarray:
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
) -> Dict[str, Any]:
    """
    Correct XY drift on a single image file.

    Parameters:
    - input_path: path to image to correct (expects TCZYX)
    - output_dir: where to write outputs (defaults next to input)
    - drift_channel: channel index to estimate drift from
    - method: 'cpu' | 'gpu' | 'auto' (gpu then fallback to cpu)
    - timepoints: if provided, limit to first T timepoints
    - cpu_threads: per-plane transform threads for CPU implementation
    - save_shifts: save tmats/shifts npy alongside TIFF

    Returns dict with output paths and basic metadata.
    """
    t0 = time.perf_counter()
    data = load_first_T_timepoints(input_path, max_T=timepoints)
    T, C, Z, Y, X = data.shape
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = output_dir or os.path.join(os.path.dirname(input_path), "_output")
    os.makedirs(out_dir, exist_ok=True)

    chosen = method.lower()
    corrected: np.ndarray
    info: Dict[str, Any] = {}
    npy_path = None

    if chosen not in {"cpu", "gpu", "auto"}:
        raise ValueError("method must be one of 'cpu', 'gpu', or 'auto'")

    def _save_outputs(arr: np.ndarray, suffix: str) -> str:
        out_tif = os.path.join(out_dir, f"{base}_T{T}_drift_{suffix}.tif")
        OmeTiffWriter.save(arr, out_tif, dim_order="TCZYX")
        return out_tif

    # Try GPU first if requested/auto
    gpu_failed = False
    gpu_used = False
    if chosen in {"gpu", "auto"}:
        logger.info("Attempting GPU drift correction (pyGPUreg)...")
        corrected_gpu, gpu_shifts = drift_correct_xy_pygpureg(data, drift_correct_channel=drift_channel)
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
            info["method"] = "gpu"
            info["time_sec_gpu"] = time.perf_counter() - t0

    # CPU path if requested or GPU failed in auto
    if (chosen == "cpu") or (chosen == "auto" and (gpu_failed or not gpu_used)):
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

    info.update({
        "input_path": input_path,
        "output_dir": out_dir,
        "T": T,
        "shape": (T, C, Z, Y, X),
        "drift_channel": drift_channel,
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
            )
        except Exception as e:
            logger.error(f"Failed to correct {img_path}: {e}")
            return {"input_path": img_path, "error": str(e)}

    if parallel and is_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(_one, p) for p in image_files]
            for f in tqdm(as_completed(futures), total=len(image_files), desc="Drift correcting"):
                yield f.result()
    else:
        for p in tqdm(image_files, desc="Drift correcting"):
            yield _one(p)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct XY drift using CPU (pystackreg) or GPU (pyGPUreg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern or single file path.')
    parser.add_argument('--output-dir', default=None, help='Where to place outputs')
    parser.add_argument('--drift-channel', type=int, default=0, help='Channel index used for drift estimation')
    parser.add_argument('--method', default='auto', choices=['cpu', 'gpu', 'auto'], help='Which method to use')
    parser.add_argument('--cpu-threads', type=int, default=0, help='Threads for per-plane transform in CPU baseline (0/1=serial)')
    parser.add_argument('--save-shifts', action='store_true', help='Save tmats/shifts npy next to outputs')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel batch processing')
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
    ):
        results.append(res)
        if 'error' in res:
            logger.error(f"Error: {res['error']} ({res.get('input_path')})")
        else:
            logger.info(f"Done: {res.get('output_tif')} ({res.get('method')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
