"""
Select mask objects that touch ROI points from metadata YAML.

For each 2D mask slice across all T, C, Z dimensions, this script keeps only
objects that directly touch a ROI point. All other objects are removed (set to 0).

If the input mask is binary, objects are identified using 4-way connected
components (up, down, left, right).
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

import bioimage_pipeline_utils as rp


def extract_roi_positions(yaml_path: str) -> Optional[List[Tuple[float, float]]]:
    """
    Extract ROI positions from metadata YAML file.

    Args:
        yaml_path: Path to metadata YAML file.

    Returns:
        List of (x_pixels, y_pixels) tuples, or None if no ROIs found.
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)

        if not metadata or "Image metadata" not in metadata:
            return None

        image_metadata = metadata["Image metadata"]
        if "ROIs" not in image_metadata or not image_metadata["ROIs"]:
            return None

        roi_positions: List[Tuple[float, float]] = []
        for roi_entry in image_metadata["ROIs"]:
            if "Roi" not in roi_entry:
                continue

            roi = roi_entry["Roi"]
            positions = roi.get("Positions", {})
            x_pixels = positions.get("x", None)
            y_pixels = positions.get("y", None)

            if x_pixels is not None and y_pixels is not None:
                roi_positions.append((float(x_pixels), float(y_pixels)))

        return roi_positions if roi_positions else None

    except Exception as exception:
        logging.warning(f"Failed to extract ROI from {yaml_path}: {exception}")
        return None


def is_binary_mask(mask_2d: np.ndarray) -> bool:
    """
    Determine whether a 2D mask should be treated as binary.

    Args:
        mask_2d: 2D indexed mask.

    Returns:
        True if mask has at most one non-zero value (plus zero).
    """
    if mask_2d.size == 0:
        return False

    unique_values = np.unique(mask_2d)
    non_zero_values = unique_values[unique_values > 0]
    return non_zero_values.size <= 1


def label_connected_components_4way(binary_mask_2d: np.ndarray) -> np.ndarray:
    """
    Label connected components in a 2D binary mask using 4-way connectivity.

    Args:
        binary_mask_2d: 2D binary mask where foreground is > 0.

    Returns:
        2D int32 labeled mask with background 0 and components 1..N.
    """
    foreground = binary_mask_2d > 0
    labels = np.zeros(binary_mask_2d.shape, dtype=np.int32)
    height, width = binary_mask_2d.shape
    current_label = 0

    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or labels[start_y, start_x] != 0:
                continue

            current_label += 1
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current_label

            while stack:
                y_idx, x_idx = stack.pop()

                for ny, nx in ((y_idx - 1, x_idx), (y_idx + 1, x_idx), (y_idx, x_idx - 1), (y_idx, x_idx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if not foreground[ny, nx] or labels[ny, nx] != 0:
                        continue

                    labels[ny, nx] = current_label
                    stack.append((ny, nx))

    return labels


def get_labels_for_touch_selection(mask_2d: np.ndarray) -> np.ndarray:
    """
    Return label image used for ROI-touch object selection.

    For indexed masks, this is the original mask. For binary masks, this is a
    4-way connected-component labeling.
    """
    if is_binary_mask(mask_2d):
        return label_connected_components_4way(mask_2d)
    return mask_2d.astype(np.int64, copy=False)


def select_touching_labels_for_all_slices(mask_data: np.ndarray, roi_positions: List[Tuple[float, float]]) -> np.ndarray:
    """
    Keep only objects touching ROI points for each T/C/Z slice.

    Args:
        mask_data: 5D TCZYX indexed mask data.
        roi_positions: List of ROI positions as (x, y) pixel coordinates.

    Returns:
        Filtered 5D mask data with non-selected labels set to 0.
    """
    selected_mask = np.zeros_like(mask_data)

    t_size, c_size, z_size, _, _ = mask_data.shape

    for t_idx in range(t_size):
        for c_idx in range(c_size):
            for z_idx in range(z_size):
                mask_2d = mask_data[t_idx, c_idx, z_idx, :, :]
                if not np.any(mask_2d > 0):
                    continue

                label_map = get_labels_for_touch_selection(mask_2d)

                labels_to_keep = set()
                for roi_x, roi_y in roi_positions:
                    y_round = int(round(roi_y))
                    x_round = int(round(roi_x))

                    if not (0 <= y_round < label_map.shape[0] and 0 <= x_round < label_map.shape[1]):
                        continue

                    label_id = int(label_map[y_round, x_round])
                    if label_id > 0:
                        labels_to_keep.add(label_id)

                if labels_to_keep:
                    keep_values = np.fromiter(labels_to_keep, dtype=label_map.dtype)
                    keep_mask = np.isin(label_map, keep_values)
                    selected_mask[t_idx, c_idx, z_idx, :, :] = np.where(keep_mask, mask_2d, 0)

    return selected_mask


def process_file(mask_path: str, yaml_path: str, output_folder: str, output_extension: str) -> None:
    """Process one mask/YAML pair and save ROI-selected mask."""
    os.makedirs(output_folder, exist_ok=True)

    logging.info(f"Processing mask: {mask_path}")
    mask_img = rp.load_tczyx_image(mask_path)
    mask_data = mask_img.data

    if mask_data is None:
        raise ValueError(f"Mask data is None for {mask_path}")

    roi_positions = extract_roi_positions(yaml_path)
    if not roi_positions:
        logging.warning(f"No ROI positions found for {yaml_path}; output mask will be empty.")
        selected_data = np.zeros_like(mask_data)
    else:
        selected_data = select_touching_labels_for_all_slices(mask_data, roi_positions)

    base_name = Path(mask_path).stem
    output_path = os.path.join(output_folder, f"{base_name}{output_extension}.tif")

    rp.save_mask(
        selected_data,
        output_path,
        as_binary=True,
    )

    logging.info(f"Saved selected mask: {output_path}")


def select_pairs_by_mode(
    file_pairs: List[Tuple[str, str]],
    mode: str,
) -> List[Tuple[str, str]]:
    """Select pairs using mode syntax compatible with qc_mask.py."""
    mode_value = str(mode).lower()
    import re

    mode_match = re.match(r"(all|examples|first|random)(\d+)?$", mode_value)
    if not mode_match:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'all', 'examples[N]', 'first[N]', or 'random[N]'"
        )

    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else 3

    n_files = len(file_pairs)
    if mode_type == "all":
        return file_pairs

    if mode_type == "first":
        return file_pairs[: min(mode_count, n_files)]

    if mode_type == "random":
        import random

        return random.sample(file_pairs, min(mode_count, n_files))

    if n_files <= mode_count:
        return file_pairs

    if mode_count == 1:
        indices = [n_files // 2]
    elif mode_count == 2:
        indices = [0, n_files - 1]
    else:
        indices = [0]
        for index in range(1, mode_count - 1):
            indices.append(int(index * n_files / (mode_count - 1)))
        indices.append(n_files - 1)

    return [file_pairs[index] for index in indices]


def build_file_pairs(
    mask_search_pattern: str,
    yaml_search_pattern: str,
    search_subfolders: bool,
) -> List[Tuple[str, str]]:
    """Build matched mask/YAML file pairs using grouped-file utility."""
    search_patterns = {
        "mask": mask_search_pattern,
        "yaml": yaml_search_pattern,
    }

    grouped_files = rp.get_grouped_files_to_process(search_patterns, search_subfolders)
    if not grouped_files:
        raise FileNotFoundError("No matching mask-yaml pairs found.")

    file_pairs: List[Tuple[str, str]] = []
    for basename, files in grouped_files.items():
        if "mask" not in files:
            logging.warning(f"Missing mask for basename '{basename}', skipping.")
            continue
        if "yaml" not in files:
            logging.warning(f"Missing yaml for basename '{basename}', skipping.")
            continue

        file_pairs.append((files["mask"], files["yaml"]))

    if not file_pairs:
        raise ValueError("No complete mask-yaml pairs found.")

    return file_pairs


def process_folder(args: argparse.Namespace) -> None:
    """Process matched mask/YAML pairs sequentially or in parallel."""
    all_pairs = build_file_pairs(
        mask_search_pattern=args.mask_search_pattern,
        yaml_search_pattern=args.yaml_search_pattern,
        search_subfolders=args.search_subfolders,
    )

    selected_pairs = select_pairs_by_mode(all_pairs, args.mode)

    logging.info(f"Matched {len(all_pairs)} total mask-yaml pairs")
    logging.info(f"Selected {len(selected_pairs)} pairs using mode '{args.mode}'")

    if args.no_parallel or len(selected_pairs) <= 1:
        for mask_path, yaml_path in tqdm(selected_pairs, desc="Processing files", unit="file"):
            process_file(mask_path, yaml_path, args.output_folder, args.output_extension)
        return

    cpu_count = os.cpu_count() or 1
    max_workers = max(cpu_count - 1, 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, mask_path, yaml_path, args.output_folder, args.output_extension)
            for mask_path, yaml_path in selected_pairs
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            future.result()


# Example run_pipeline.exe YAML:
# ---
# run:
# - name: Select ROI-touching mask objects
#   environment: uv@3.11:mask-select-from-point
#   commands:
#   - python
#   - '%REPO%/standard_code/python/mask_select_from_point.py'
#   - --mask-search-pattern: '%YAML%/output_masks/**/*_mask.tif'
#   - --yaml-search-pattern: '%YAML%/input_tif/**/*_metadata.yaml'
#   - --output-folder: '%YAML%/output_selected_masks'
#   - --mode: all
#   - --output-extension: _mask
#   - --no-parallel
def main() -> None:
    parser = argparse.ArgumentParser(
                description="Keep only mask objects touching ROI points from metadata YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Select ROI-touching mask objects
  environment: uv@3.11:mask-select-from-point
  commands:
  - python
  - '%REPO%/standard_code/python/mask_select_from_point.py'
  - --mask-search-pattern: '%YAML%/output_masks/**/*_mask.tif'
  - --yaml-search-pattern: '%YAML%/input_tif/**/*_metadata.yaml'
  - --output-folder: '%YAML%/output_selected_masks'
  - --mode: all
    - --output-extension: _mask
  - --no-parallel
""",
    )

    parser.add_argument(
        "--mask-search-pattern",
        type=str,
        required=True,
        help="Glob pattern for mask images.",
    )
    parser.add_argument(
        "--yaml-search-pattern",
        type=str,
        required=True,
        help="Glob pattern for metadata YAML files containing ROI positions.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Folder to save selected masks. If not provided, defaults to mask pattern folder.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help=(
            "Processing mode: 'all' = all files, 'examples[N]' = evenly spaced N files, "
            "'first[N]' = first N files, 'random[N]' = random N files."
        ),
    )
    parser.add_argument(
        "--search-subfolders",
        action="store_true",
        help="Enable recursive search for files.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (default: parallel enabled).",
    )
    parser.add_argument(
        "--output-extension",
        "--output-suffix",
        dest="output_extension",
        type=str,
        default="",
        help='Extension text appended before .tif (default: ""). Example: "_mask".',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.output_folder is None:
        args.output_folder = os.path.dirname(args.mask_search_pattern) or "."

    process_folder(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
