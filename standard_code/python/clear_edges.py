"""Clear edge-touching objects from binary or indexed masks.

This CLI processes mask images in batch mode using TCZYX-safe loading/saving.
By default, objects touching XY image borders are removed. Optionally include
Z borders as well.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Any

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

import bioimage_pipeline_utils as rp


logger = logging.getLogger(__name__)


def _default_dims_order_for_ndim(ndim: int) -> str:
    """Return fallback dims order for array ranks 1-5."""
    defaults = {
        1: "X",
        2: "YX",
        3: "ZYX",
        4: "CZYX",
        5: "TCZYX",
    }
    if ndim not in defaults:
        raise ValueError(f"Unsupported array ndim={ndim}; expected 1-5 dimensions")
    return defaults[ndim]


def _tczyx_to_original_layout(
    arr_tczyx: np.ndarray,
    original_shape: tuple[int, ...],
    input_dims_order: Optional[str],
) -> np.ndarray:
    """Map TCZYX array back to the original array rank/order."""
    if arr_tczyx.ndim != 5:
        raise ValueError(f"Expected TCZYX 5D array, got shape {arr_tczyx.shape}")

    original_ndim = len(original_shape)
    order = input_dims_order.strip().upper() if input_dims_order else _default_dims_order_for_ndim(original_ndim)

    if len(order) != original_ndim:
        raise ValueError(
            f"input_dims_order '{order}' length must match original ndim {original_ndim}"
        )
    if any(dim not in "TCZYX" for dim in order):
        raise ValueError(
            f"input_dims_order '{order}' contains invalid dims; only T,C,Z,Y,X are allowed"
        )
    if len(set(order)) != len(order):
        raise ValueError(f"input_dims_order '{order}' contains duplicate dimensions")

    index = tuple(slice(None) if dim in order else 0 for dim in "TCZYX")
    reduced = arr_tczyx[index]
    current_order = "".join(dim for dim in "TCZYX" if dim in order)
    perm = [current_order.index(dim) for dim in order]
    restored = np.transpose(reduced, axes=perm)

    if restored.shape != original_shape:
        raise ValueError(
            f"Restored array shape {restored.shape} does not match original shape {original_shape}"
        )
    return restored


def _is_binary_mask(volume_zyx: np.ndarray) -> bool:
    """Return True if a mask volume contains only 0/1 values."""
    unique_vals = np.unique(volume_zyx)
    if unique_vals.size == 0:
        return True
    if unique_vals.size > 2:
        return False
    return bool(np.all(np.isin(unique_vals, [0, 1])))


def _edge_labels_from_volume(labels_zyx: np.ndarray, clear_z_edges: bool) -> np.ndarray:
    """Collect non-zero labels touching requested border planes."""
    border_values = [
        labels_zyx[:, 0, :],
        labels_zyx[:, -1, :],
        labels_zyx[:, :, 0],
        labels_zyx[:, :, -1],
    ]
    if clear_z_edges:
        border_values.extend([labels_zyx[0, :, :], labels_zyx[-1, :, :]])

    touching = np.unique(np.concatenate([v.ravel() for v in border_values]))
    return touching[touching != 0]


def _clear_binary_edges(volume_zyx: np.ndarray, clear_z_edges: bool) -> np.ndarray:
    """Remove connected components touching requested edges from a binary mask."""
    foreground = volume_zyx > 0
    if not np.any(foreground):
        return volume_zyx

    structure = np.ones((3, 3, 3), dtype=bool)
    label_result: Any = ndi.label(foreground, structure=structure)
    cc = label_result[0]
    nlabels = int(label_result[1])
    if nlabels == 0:
        return volume_zyx

    touching = _edge_labels_from_volume(cc, clear_z_edges)
    if touching.size == 0:
        return volume_zyx

    output = np.array(volume_zyx, copy=True)
    output[np.isin(cc, touching)] = 0
    return output


def _clear_indexed_edges(volume_zyx: np.ndarray, clear_z_edges: bool) -> np.ndarray:
    """Remove indexed labels touching requested edges from a label mask."""
    touching = _edge_labels_from_volume(volume_zyx, clear_z_edges)
    if touching.size == 0:
        return volume_zyx

    output = np.array(volume_zyx, copy=True)
    output[np.isin(output, touching)] = 0
    return output


def clear_edges_mask(data_tczyx: np.ndarray, clear_z_edges: bool = False) -> np.ndarray:
    """Clear edge-touching objects per T,C volume from a TCZYX mask stack."""
    if data_tczyx.ndim != 5:
        raise ValueError(f"Expected TCZYX 5D array, got shape {data_tczyx.shape}")

    output = np.array(data_tczyx, copy=True)
    t_size, c_size = output.shape[0], output.shape[1]

    for t_idx in range(t_size):
        for c_idx in range(c_size):
            volume = output[t_idx, c_idx, :, :, :]
            if _is_binary_mask(volume):
                output[t_idx, c_idx, :, :, :] = _clear_binary_edges(volume, clear_z_edges)
            else:
                output[t_idx, c_idx, :, :, :] = _clear_indexed_edges(volume, clear_z_edges)

    return output


def clear_single_file(
    input_path: str,
    output_path: str,
    output_format: str,
    clear_z_edges: bool = False,
    input_dims_order: Optional[str] = None,
) -> bool:
    """Process one file and save edge-cleared mask output."""
    try:
        logger.info("Clearing edges: %s", os.path.basename(input_path))
        normalized_output_format = rp.normalize_output_format(output_format)

        cellpose_payload_dict: Optional[dict[str, Any]] = None
        cellpose_masks_key: Optional[str] = None
        cellpose_masks_shape: Optional[tuple[int, ...]] = None
        cellpose_masks_dtype: Optional[np.dtype] = None

        if normalized_output_format == "npy" and input_path.lower().endswith(".npy"):
            raw_payload = np.load(input_path, allow_pickle=True)
            if isinstance(raw_payload, np.ndarray) and raw_payload.dtype == object and raw_payload.shape == ():
                maybe_dict = raw_payload.item()
                if isinstance(maybe_dict, dict):
                    for preferred_key in ("masks", "labels", "mask", "segmentation"):
                        if preferred_key in maybe_dict:
                            cellpose_payload_dict = dict(maybe_dict)
                            cellpose_masks_key = preferred_key
                            original_masks = np.asarray(maybe_dict[preferred_key])
                            cellpose_masks_shape = tuple(original_masks.shape)
                            cellpose_masks_dtype = original_masks.dtype
                            logger.info(
                                "Detected dict-based npy payload with key '%s'; preserving metadata keys",
                                preferred_key,
                            )
                            break

        img = rp.load_tczyx_image(input_path, input_dims_order=input_dims_order)
        data = np.asarray(img.data)
        cleared = clear_edges_mask(data, clear_z_edges=clear_z_edges)

        save_kwargs = {}
        physical_pixel_sizes = getattr(img, "physical_pixel_sizes", None)

        if (
            normalized_output_format == "npy"
            and cellpose_payload_dict is not None
            and cellpose_masks_key is not None
            and cellpose_masks_shape is not None
            and cellpose_masks_dtype is not None
        ):
            restored_masks = _tczyx_to_original_layout(
                cleared,
                original_shape=cellpose_masks_shape,
                input_dims_order=input_dims_order,
            )
            cellpose_payload_dict[cellpose_masks_key] = restored_masks.astype(cellpose_masks_dtype, copy=False)
            np.save(output_path, np.array(cellpose_payload_dict, dtype=object), allow_pickle=True)
            logger.info("Saved Cellpose-compatible npy payload: %s", output_path)
            return True

        if normalized_output_format in {"tif", "ome.tif"} and physical_pixel_sizes is not None:
            save_kwargs["physical_pixel_sizes"] = physical_pixel_sizes

        rp.save_with_output_format(cleared, output_path, normalized_output_format, **save_kwargs)
        logger.info("Saved: %s", output_path)
        return True
    except Exception as exc:
        logger.error("Failed to clear edges for %s: %s", input_path, exc)
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str],
    output_suffix: str,
    output_format: str,
    collapse_delimiter: str,
    clear_z_edges: bool,
    input_dims_order: Optional[str],
    no_force: bool,
    no_parallel: bool,
    maxcores: Optional[int],
    dry_run: bool,
) -> None:
    """Batch process files matching pattern with optional parallel execution."""
    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error("No files found matching pattern: %s", input_pattern)
        return

    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)

    if output_folder is None:
        output_folder = f"{base_folder}_edge_cleared"

    os.makedirs(output_folder, exist_ok=True)
    output_extension = rp.output_extension_for_format(output_format, tiff_extension=".ome.tif")

    file_pairs: list[tuple[str, str]] = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        out_name = os.path.basename(
            rp.resolve_output_path(collapsed, extension=output_extension, suffix=output_suffix)
        )
        dst = os.path.join(output_folder, out_name)
        if no_force and os.path.exists(dst):
            logger.info("Skipping existing output (no-force): %s", dst)
            continue
        file_pairs.append((src, dst))

    if not file_pairs:
        logger.warning("No files to process after filtering existing outputs")
        return

    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Output format: {output_format}")
        print(f"[DRY RUN] Clear Z edges: {clear_z_edges}")
        print(f"[DRY RUN] Input dims order: {input_dims_order or 'auto/default'}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    if no_parallel or len(file_pairs) == 1:
        for src, dst in file_pairs:
            ok = clear_single_file(
                src,
                dst,
                output_format,
                clear_z_edges=clear_z_edges,
                input_dims_order=input_dims_order,
            )
            if not ok:
                logger.error("Failed: %s", src)
        return

    max_workers = rp.resolve_maxcores(maxcores, len(file_pairs))
    logger.info("Processing with %d workers", max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clear_single_file, src, dst, output_format, clear_z_edges, input_dims_order): (src, dst)
            for src, dst in file_pairs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            src, _dst = futures[future]
            try:
                ok = future.result()
                if not ok:
                    logger.error("Failed: %s", src)
            except Exception as exc:
                logger.error("Exception processing %s: %s", src, exc)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Clear edge-touching objects from binary/indexed masks. "
            "Default removes objects touching XY borders; add --clear-z-edges to also clear Z borders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Clear edges in XY only (default)
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/clear_edges.py'
    - --input-search-pattern: '%YAML%/input_masks/**/*.tif'
    - --output-folder: '%YAML%/output_masks'
    - --output-suffix: '_clearedges'
    - --output-format: ome.tif

- name: Clear edges in XYZ
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/clear_edges.py'
    - --input-search-pattern: '%YAML%/input_masks/**/*.tif'
    - --output-folder: '%YAML%/output_masks'
    - --clear-z-edges
    - --log-level: INFO

- name: Clear Cellpose seg npy with explicit dims override
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/clear_edges.py'
    - --input-search-pattern: '%YAML%/nucleus/*.ome_seg.npy'
    - --output-folder: '%YAML%/nucleus'
    - --output-suffix: '_nucmask'
    - --output-format: npy
    - --input-dims-order: ZYX

- name: Pause for manual inspection
    type: pause
    message: 'Inspect edge-cleared masks before continuing.'

- name: Stop intentionally
    type: stop
    message: 'Pipeline stopped intentionally.'

- name: Force reprocessing for later segments
    type: force
    message: 'Reprocessing all subsequent steps.'
""",
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Glob pattern for input mask files",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Destination folder for outputs (default: input base + '_edge_cleared')",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_clearedges",
        help="Suffix appended before extension (default: _clearedges)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="ome.tif",
        choices=["ome.tif", "tif", "npy", "ilastik-h5", "h5"],
        help="Output format (default: ome.tif)",
    )
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter used when collapsing relative subfolder paths (default: '__')",
    )
    parser.add_argument(
        "--clear-z-edges",
        action="store_true",
        help="Also remove objects touching Z borders (default only clears XY borders)",
    )
    parser.add_argument(
        "--input-dims-order",
        type=str,
        default=None,
        help=(
            "Optional input dims order for array-based inputs (.npy/.npz), using letters from TCZYX. "
            "Examples: YX, ZYX, CZYX, TCZYX. Default uses auto mapping by array rank."
        ),
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Skip files whose output already exists",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--maxcores",
        type=int,
        default=None,
        help="Maximum CPU cores to use for parallel processing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        output_suffix=args.output_suffix,
        output_format=args.output_format,
        collapse_delimiter=args.collapse_delimiter,
        clear_z_edges=args.clear_z_edges,
        input_dims_order=args.input_dims_order,
        no_force=args.no_force,
        no_parallel=args.no_parallel,
        maxcores=args.maxcores,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
