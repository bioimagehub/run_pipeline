import os
import sys
import argparse
import logging
from typing import Optional, Any, Tuple
import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from bioio.writers import OmeTiffWriter  # type: ignore
from bioio import BioImage

# Local helpers
# Use standard import since bioimage_pipeline_utils is in same directory
import bioimage_pipeline_utils as rp  


def get_physical_pixel_sizes_safe(image: BioImage, src_path: str, out_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Safely retrieve physical pixel sizes from a BioImage object.
    If retrieval fails, logs a warning and writes an error sidecar file.

    Args:
        image: BioImage object to extract pixel sizes from.
        src_path: Path to the source image file.
        out_path: Path to the intended output file (for error sidecar).

    Returns:
        Tuple of (Z, Y, X) pixel sizes, or (None, None, None) if unavailable.
    """
    try:
        pps = image.physical_pixel_sizes if image.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        logger.warning(f"Error retrieving physical pixel sizes: {e} for file {src_path}. Using None.")
        # Persist an error sidecar for traceability
        try:
            with open(os.path.splitext(out_path)[0] + "_error.txt", 'w', encoding='utf-8') as f:
                f.write(f"pps_error: {e}\n")
        except Exception:
            pass
        pps = (None, None, None)
    return pps  # type: ignore[return-value]


def load_or_derive_metadata(src_path: str, out_path: str) -> dict:
    """
    Load metadata from a YAML sidecar if present, else derive minimal metadata.

    Args:
        src_path: Path to the source image file.
        out_path: Path to the intended output file.

    Returns:
        Dictionary of metadata.
    """
    md_path = os.path.splitext(src_path)[0] + "_metadata.yaml"
    if os.path.exists(md_path):
        try:
            import yaml
            with open(md_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception as e:
            logger.warning(f"Failed reading existing metadata yaml: {e}")
    # Fallback: derive minimal metadata
    return {"Source": os.path.basename(src_path)}


def _project(image: BioImage, proj_method: Optional[str]) -> np.ndarray:
    """
    Optionally Z-project a BioImage to reduce Z dimension using a specified method.
    Always returns a 5D array (TCZYX).

    Args:
        image: BioImage object to project.
        proj_method: Projection method ('max', 'sum', 'mean', etc.), or None for no projection.

    Returns:
        Projected numpy array in TCZYX order.
    """
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
        logger.warning(f"Unknown projection '{proj_method}', using max")
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
    """
    Apply drift correction to a TCZYX image array using the specified method.

    Args:
        img_tczyx: Input image as a numpy array (TCZYX).
        drift_channel: Channel index to use for drift correction (-1 to skip).
        method: Drift correction method ('cpu', 'gpu', 'cupy', 'auto').
        logger: Logger instance (optional).

    Returns:
        Tuple of (corrected array, shifts array, method used, error report).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
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
            out, shifts, _ = dc_cupy(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
            return out, shifts, "cupy", None
        if m == "gpu":
            out, shifts, _ = dc_gpu(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
            return out, shifts, "gpu", None
        if m == "auto":
            # Try CuPy → GPU → CPU
            try:
                out, shifts, _ = dc_cupy(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
                return out, shifts, "cupy", None
            except Exception as e:
                error_report = f"CuPy drift correction failed: {e}"
                logger.warning(error_report)
            try:
                out, shifts, _ = dc_gpu(img_tczyx, drift_correct_channel=drift_channel, logger=logger)
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
    """
    Save output image, drift correction shifts, error reports, and metadata sidecar.

    Args:
        image_tczyx: Output image array (TCZYX).
        shifts: Drift correction shifts array (optional).
        pps: Physical pixel sizes.
        out_tif_path: Output OME-TIFF path.
        out_shifts_path: Output shifts .npy path.
        metadata: Metadata dictionary to update and save.
        proj_method: Projection method used.
        drift_channel: Channel used for drift correction.
        out_md_path: Output metadata YAML path.
        drift_method: Drift correction method used.
        error_report: Error report string (optional).
        logger: Logger instance (optional).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
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
    """
    Process a single image: project, drift-correct, and save outputs and metadata.

    Args:
        img: Input BioImage object.
        input_file_path: Path to input image file.
        output_tif_file_path: Path to output OME-TIFF file.
        drift_correct_channel: Channel index for drift correction (-1 to skip).
        projection_method: Z-projection method (optional).
        drift_method: Drift correction method ('cpu', 'gpu', 'cupy', 'auto').
    """
    # SAFEGUARDS: avoid overwriting inputs or metadata
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    output_shifts_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

    if os.path.abspath(input_file_path) == os.path.abspath(output_tif_file_path):
        logger.error("Output equals input; aborting to prevent overwrite")
        return
    if os.path.abspath(input_metadata_file_path) == os.path.abspath(output_metadata_file_path):
        logger.error("Output metadata equals input metadata; aborting to prevent overwrite")
        return

    # Determine dtype early to force load
    try:
        _ = img.data.dtype
    except Exception as e:
        logger.error(f"Image lacks dtype: {e}; skipping {input_file_path}")
        return

    pps = get_physical_pixel_sizes_safe(img, input_file_path, output_tif_file_path)
    metadata = load_or_derive_metadata(input_file_path, output_tif_file_path)

    arr = _project(img, projection_method)
    out, shifts, used_method, error_report = _apply_drift_correction(arr, drift_correct_channel, drift_method, logger=logger)

    _save_outputs(out, shifts, pps, output_tif_file_path, output_shifts_file_path, metadata,
                  projection_method, drift_correct_channel, output_metadata_file_path, used_method, error_report, logger=logger)


def process_pattern(args: argparse.Namespace) -> None:
    # Determine if recursive search is requested
    search_subfolders = '**' in args.input_search_pattern
    
    # Expand glob pattern using standardized helper function
    files = rp.get_files_to_process2(args.input_search_pattern, search_subfolders=search_subfolders)
    if not files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        return
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Determine base_folder for path collapsing
    # If pattern contains '**', use the part before '**' as base
    # Otherwise, use the parent directory of the pattern
    if '**' in args.input_search_pattern:
        # Extract base path before '**'
        base_folder = args.input_search_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:  # If pattern starts with '**', use current directory
            base_folder = os.getcwd()
        # Normalize path separators and resolve to absolute path
        base_folder = os.path.abspath(base_folder)
        logger.info(f"Using base folder for path collapsing: {base_folder}")
        logger.info(f"Subfolders after '**' will be collapsed with delimiter '{args.collapse_delimiter}'")
    else:
        # For non-recursive patterns, use parent of first file
        from pathlib import Path
        base_folder = str(Path(files[0]).parent)

    # Destination
    dest = args.output_folder if getattr(args, "output_folder", None) else base_folder + "_tif"
    os.makedirs(dest, exist_ok=True)
    logger.info(f"Output folder: {dest}")

    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, args.collapse_delimiter)
        out_path = os.path.splitext(collapsed)[0] + args.output_file_name_extension + ".tif"
        out_path = os.path.join(dest, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            img = rp.load_tczyx_image(src)
        except Exception as e:
            logger.error(f"Failed to load {src}: {e}")
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
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, e.g., 'data/*.tif' or 'data/**/*.tif' for recursive search). Use '**' to search subfolders recursively."
    )
    parser.add_argument(
        "--search-subfolders",
        action="store_true",
        help="(Deprecated) Recursive search is now automatic when '**' is in the pattern"
    )
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder structure in output filenames when using '**' (default: __). Example: 'folder1/folder2/image.tif' becomes 'folder1__folder2__image.tif'"
    )

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

    parser.add_argument("--dry-run", action="store_true", help="Print planned actions but do not write any files.")
    parser.add_argument("--version", action="store_true", help="Print version and exit.")

    args = parser.parse_args()

    if args.version:
        # Print version from VERSION file if present, else fallback
        version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "VERSION")
        version = None
        try:
            with open(version_file, "r", encoding="utf-8") as vf:
                version = vf.read().strip()
        except Exception:
            version = "unknown"
        print(f"convert_to_tif.py version: {version}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.dry_run:
        # Print planned actions for each file, do not write outputs
        search_subfolders = '**' in args.input_search_pattern
        files = rp.get_files_to_process2(args.input_search_pattern, search_subfolders=search_subfolders)
        
        if not files:
            print(f"[DRY RUN] No files found matching pattern: {args.input_search_pattern}")
            return
        
        # Determine base_folder using same logic as main processing
        if '**' in args.input_search_pattern:
            base_folder = args.input_search_pattern.split('**')[0].rstrip('/\\')
            if not base_folder:
                base_folder = os.getcwd()
            base_folder = os.path.abspath(base_folder)
            print(f"[DRY RUN] Using base folder for path collapsing: {base_folder}")
            print(f"[DRY RUN] Subfolders after '**' will be collapsed with delimiter '{args.collapse_delimiter}'")
        else:
            from pathlib import Path
            base_folder = str(Path(files[0]).parent)

        dest = args.output_folder if getattr(args, "output_folder", None) else base_folder + "_tif"
        print(f"[DRY RUN] Would process {len(files)} files. Output folder: {dest}")
        for src in files:
            collapsed = rp.collapse_filename(src, base_folder, args.collapse_delimiter)
            out_path = os.path.splitext(collapsed)[0] + args.output_file_name_extension + ".tif"
            out_path = os.path.join(dest, out_path)
            print(f"[DRY RUN] Would convert: {src} -> {out_path}")
        return

    process_pattern(args)


if __name__ == "__main__":
    main()
