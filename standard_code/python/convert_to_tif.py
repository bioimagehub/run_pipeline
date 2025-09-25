import os
import sys
import argparse
import logging
from typing import Optional, Any, Tuple
import numpy as np

from bioio.writers import OmeTiffWriter  # type: ignore
from bioio import BioImage

# Local helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bioimage_pipeline_utils as rp  


def get_physical_pixel_sizes_safe(image: BioImage, src_path: str, out_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        pps = image.physical_pixel_sizes if image.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        logging.warning(f"Error retrieving physical pixel sizes: {e} for file {src_path}. Using None.")
        # Persist an error sidecar for traceability
        try:
            with open(os.path.splitext(out_path)[0] + "_error.txt", 'w', encoding='utf-8') as f:
                f.write(f"pps_error: {e}\n")
        except Exception:
            pass
        pps = (None, None, None)
    return pps  # type: ignore[return-value]


def load_or_derive_metadata(src_path: str, out_path: str) -> dict:
    md_path = os.path.splitext(src_path)[0] + "_metadata.yaml"
    if os.path.exists(md_path):
        try:
            import yaml
            with open(md_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception as e:
            logging.warning(f"Failed reading existing metadata yaml: {e}")
    # Fallback: derive minimal metadata
    return {"Source": os.path.basename(src_path)}


def _project(image: BioImage, proj_method: Optional[str]) -> np.ndarray:
    # Load to numpy eagerly; BioImage.data may be dask-backed
    base = image.data
    # If this is a Dask array, it has a callable .compute(); otherwise fallback to np.asarray
    base_np: np.ndarray
    try:
        compute_fn = getattr(base, "compute", None)
        if callable(compute_fn):
            base_np = np.asarray(compute_fn())
        else:
            base_np = np.asarray(base)
    except Exception:
        base_np = np.asarray(base)

    # Expect axis order in BioImage to be TCZYX whenever present
    # No-op if no projection requested
    if not proj_method:
        return base_np

    m = proj_method.lower()
    # Project over Z if has Z>1; otherwise return as-is
    # Identify axes by shape length; simplest heuristic
    # BioImage usually orders TCZYX; pad missing leading dims
    arr = base_np
    # Normalize to 5D (T,C,Z,Y,X)
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    T, C, Z, Y, X = arr.shape
    if Z <= 1:
        return arr  # already 2D per T,C

    if m == "max":
        proj = arr.max(axis=2)
    elif m == "sum":
        proj = arr.sum(axis=2)
    elif m == "mean":
        proj = arr.mean(axis=2)
    elif m == "median":
        proj = np.median(arr, axis=2)
    elif m == "min":
        proj = arr.min(axis=2)
    elif m == "std":
        proj = arr.std(axis=2)
    else:
        logging.warning(f"Unknown projection '{proj_method}', using max")
        proj = arr.max(axis=2)
    # Insert a singleton Z back to keep TCZYX
    proj = proj[:, :, np.newaxis, ...]
    return proj


def _apply_drift_correction(
    img_tczyx: np.ndarray,
    drift_channel: int,
    method: str,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str], Optional[str]]:
    if logger is None:
        logger = logging
    if drift_channel <= -1:
        logger.info("Drift correction skipped: drift_correct_channel is set to -1.")
        return img_tczyx, None, None, None

    error_report = None
    try:
        from drift_correction import (
            drift_correct_xy_parallel as dc_cpu,
            drift_correct_xy_pygpureg as dc_gpu,
            drift_correct_xy_cupy as dc_cupy,
        )
    except Exception as e:
        msg = f"drift_correction module not available; skipping drift correction: {e}"
        logger.error(msg)
        error_report = msg
        return img_tczyx, None, None, error_report

    m = (method or "gpu").lower()
    try:
        if m == "cpu":
            out, shifts = dc_cpu(img_tczyx, drift_channel, logger=logger)
            return out, shifts, "cpu", None
        if m == "cupy":
            out, shifts = dc_cupy(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
            return out, shifts, "cupy", None
        if m == "gpu":
            out, shifts = dc_gpu(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
            return out, shifts, "gpu", None
        if m == "auto":
            # Try CuPy → GPU → CPU
            try:
                out, shifts = dc_cupy(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
                return out, shifts, "cupy", None
            except Exception as e:
                error_report = f"CuPy drift correction failed: {e}"
                logger.warning(error_report)
            try:
                out, shifts = dc_gpu(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
                return out, shifts, "gpu", None
            except Exception as e:
                error_report = f"GPU drift correction failed: {e}"
                logger.warning(error_report)
            try:
                out, shifts = dc_cpu(img_tczyx, drift_channel, logger=logger)
                return out, shifts, "cpu", None
            except Exception as e:
                error_report = f"CPU drift correction failed: {e}"
                logger.warning(error_report)
            return img_tczyx, None, None, error_report
        error_report = f"Unknown drift method '{method}', skipping drift correction."
        logger.error(error_report)
        return img_tczyx, None, None, error_report
    except Exception as e:
        error_report = f"Drift correction failed: {e}"
        logger.error(error_report)
        return img_tczyx, None, None, error_report


def _save_outputs(
    image_tczyx: np.ndarray,
    shifts: Optional[np.ndarray],
    pps,
    out_tif_path: str,
    out_shifts_path: str,
    metadata: dict,
    proj_method: Optional[str],
    drift_channel: int,
    out_md_path: str,
    drift_method: Optional[str],
    error_report: Optional[str] = None,
    logger: Optional[Any] = None,
) -> None:
    if logger is None:
        logger = logging
    rp.save_tczyx_image(image_tczyx, out_tif_path, dim_order="TCZYX", physical_pixel_sizes=pps)

    if shifts is not None:
        try:
            np.save(out_shifts_path, shifts)
        except Exception as e:
            logger.warning(f"Failed writing shifts .npy: {e}")

    # Save error report if present
    if error_report:
        try:
            with open(os.path.splitext(out_tif_path)[0] + "_drift_error.txt", "w", encoding="utf-8") as f:
                f.write(error_report + "\n")
        except Exception as e:
            logger.warning(f"Failed writing drift error report: {e}")

    # Update metadata sidecar
    convert_info: dict[str, Any] = {"Projection": {"Method": proj_method}}
    if shifts is not None:
        convert_info["Drift correction"] = {"Channel": drift_channel, "Method": drift_method}
    if error_report:
        convert_info["Drift correction error"] = error_report
    metadata["Convert to tif"] = convert_info
    try:
        import yaml
        with open(out_md_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
    except Exception as e:
        logger.warning(f"Failed writing metadata yaml: {e}")


def process_file(
    img: BioImage,
    input_file_path: str,
    output_tif_file_path: str,
    drift_correct_channel: int = -1,
    projection_method: Optional[str] = None,
    drift_method: str = "gpu",
) -> None:
    # SAFEGUARDS: avoid overwriting inputs or metadata
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    output_shifts_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

    if os.path.abspath(input_file_path) == os.path.abspath(output_tif_file_path):
        logging.error("Output equals input; aborting to prevent overwrite")
        return
    if os.path.abspath(input_metadata_file_path) == os.path.abspath(output_metadata_file_path):
        logging.error("Output metadata equals input metadata; aborting to prevent overwrite")
        return

    # Determine dtype early to force load
    try:
        _ = img.data.dtype
    except Exception as e:
        logging.error(f"Image lacks dtype: {e}; skipping {input_file_path}")
        return

    pps = get_physical_pixel_sizes_safe(img, input_file_path, output_tif_file_path)
    metadata = load_or_derive_metadata(input_file_path, output_tif_file_path)

    arr = _project(img, projection_method)
    out, shifts, used_method, error_report = _apply_drift_correction(arr, drift_correct_channel, drift_method, logger=logging)

    _save_outputs(out, shifts, pps, output_tif_file_path, output_shifts_file_path, metadata,
                  projection_method, drift_correct_channel, output_metadata_file_path, used_method, error_report, logger=logging)


def process_pattern(args: argparse.Namespace) -> None:
    # Determine search set
    is_glob = any(ch in args.input_search_pattern for ch in "*?[")
    if is_glob:
        files = rp.get_files_to_process2(args.input_search_pattern, getattr(args, 'search_subfolders', False) or False)
        base_folder = os.path.dirname(args.input_search_pattern) or "."
    else:
        recursive = getattr(args, 'search_subfolders', False) or False
        pattern = os.path.join(args.input_search_pattern, "**", "*") if recursive else os.path.join(args.input_search_pattern, "*")
        files = rp.get_files_to_process2(pattern, recursive)
        base_folder = args.input_search_pattern

    # Destination
    dest = args.output_folder if getattr(args, "output_folder", None) else base_folder + "_tif"
    os.makedirs(dest, exist_ok=True)

    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, args.collapse_delimiter)
        out_path = os.path.splitext(collapsed)[0] + args.output_file_name_extension + ".tif"
        out_path = os.path.join(dest, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)


        try:
            img = rp.load_tczyx_image(src)
        except Exception as e:
            logging.error(f"Failed to load {src}: {e}")
            continue

        process_file(
            img=img,
            input_file_path=src,
            output_tif_file_path=out_path,
            drift_correct_channel=args.drift_correct_channel,
            projection_method=args.projection_method,
            drift_method=args.drift_correct_method,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert images to OME-TIFF, optionally Z-project and drift-correct. "
            "Use --drift-correct-channel to enable correction and --drift-correct-method {cpu,gpu,cupy,auto}."
        )
    )
    parser.add_argument("--input-search-pattern", type=str, required=True)
    parser.add_argument("--search-subfolders", action="store_true")
    parser.add_argument("--collapse-delimiter", type=str, default="__")

    parser.add_argument("--projection-method", type=str, default=None,
                        choices=[None, "max", "sum", "mean", "median", "min", "std"],
                        help="Z projection method over Z axis if Z>1")
    parser.add_argument("--multipoint-files", action="store_true", help="(not implemented) multiple scenes to separate files")

    parser.add_argument("--drift-correct-channel", type=int, default=-1)
    parser.add_argument("--drift-correct-method", type=str, default="cupy",
                        choices=["cpu", "gpu", "cupy", "auto"])

    parser.add_argument("--no-parallel", action="store_true", help="unused; kept for CLI compatibility")

    parser.add_argument("--output-file-name-extension", type=str, default="")
    parser.add_argument("--output-folder", type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    process_pattern(args)


if __name__ == "__main__":
    main()
