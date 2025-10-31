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
import local_bioimage_pipeline_utils as rp
import local_extract_metadata as extract_metadata  


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
    Load metadata from a YAML sidecar if present, else extract from image.

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
    
    # Extract metadata using the extract_metadata module
    try:
        metadata = extract_metadata.get_all_metadata(src_path, output_file=None)
        return metadata
    except Exception as e:
        logger.warning(f"Failed extracting metadata: {e}")
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


def _save_outputs(
    image_tczyx: np.ndarray,
    pps,
    out_tif_path: str,
    metadata: dict,
    proj_method: Optional[str],
    out_md_path: str,
    logger: Optional[Any] = None,
) -> None:
    """
    Save output image and metadata sidecar.

    Args:
        image_tczyx: Output image array (TCZYX).
        pps: Physical pixel sizes.
        out_tif_path: Output OME-TIFF path.
        metadata: Metadata dictionary to update and save.
        proj_method: Projection method used.
        out_md_path: Output metadata YAML path.
        logger: Logger instance (optional).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    rp.save_tczyx_image(image_tczyx, out_tif_path, dim_order="TCZYX", physical_pixel_sizes=pps)

    # Add "Image metadata" wrapper if not already present
    if "Image metadata" not in metadata and any(key in metadata for key in ["Image dimensions", "Physical dimensions", "Channels"]):
        # Wrap the extracted metadata under "Image metadata"
        image_meta_keys = ["Image dimensions", "Physical dimensions", "Channels", "ROIs"]
        image_metadata = {k: metadata.pop(k) for k in image_meta_keys if k in metadata}
        metadata["Image metadata"] = image_metadata
    
    # Update metadata sidecar with conversion info
    convert_info: dict[str, Any] = {"Projection": {"Method": proj_method}}
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
    projection_method: Optional[str] = None,
) -> None:
    """
    Process a single image: project and save outputs and metadata.

    Args:
        img: Input BioImage object.
        input_file_path: Path to input image file.
        output_tif_file_path: Path to output OME-TIFF file.
        projection_method: Z-projection method (optional).
    """
    # SAFEGUARDS: avoid overwriting inputs or metadata
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"

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

    _save_outputs(arr, pps, output_tif_file_path, metadata,
                  projection_method, output_metadata_file_path, logger=logger)


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
            projection_method=args.projection_method,
        )

    # After processing, emit standardized glob patterns for downstream steps
    # Normalize to forward slashes for cross-platform globbing
    dest_norm = dest.replace("\\", "/")
    tif_glob = f"{dest_norm}/**/*.tif"
    md_glob = f"{dest_norm}/**/*_metadata.yaml"

    # Print patterns to stdout for pipeline consumption
    print(f"OUTPUT_GLOB_TIFF: {tif_glob}")
    print(f"OUTPUT_GLOB_METADATA: {md_glob}")

    # Also persist patterns to a sidecar file for tooling to read if desired
    try:
        import json
        patterns = {
            "tiff": tif_glob,
            "metadata": md_glob,
            "next_input_search_pattern": tif_glob,
        }
        with open(os.path.join(dest, "_output_patterns.json"), "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write _output_patterns.json: {e}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Convert images to OME-TIFF with optional Z-projection."
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
        # Also show the glob patterns that downstream steps can use
        dest_norm = dest.replace("\\", "/")
        print(f"[DRY RUN] OUTPUT_GLOB_TIFF: {dest_norm}/**/*.tif")
        print(f"[DRY RUN] OUTPUT_GLOB_METADATA: {dest_norm}/**/*_metadata.yaml")
        return

    process_pattern(args)


if __name__ == "__main__":
    main()
