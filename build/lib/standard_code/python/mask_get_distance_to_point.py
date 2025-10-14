import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.graph import MCP_Geometric
from scipy.spatial.distance import cdist

from bioio.writers import OmeTiffWriter
import bioimage_pipeline_utils as rp


def get_coordinates_from_metadata(yaml_path: str) -> Optional[List[tuple[int, int]]]:
    """Extract (x, y) coordinates from ROI metadata."""
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    rois = metadata.get("Image metadata", {}).get("ROIs", [])
    if not rois:
        return None

    coords = []
    for roi in rois:
        pos = roi.get("Roi", {}).get("Positions", {})
        x = int(round(pos.get("x", -1)))
        y = int(round(pos.get("y", -1)))
        if x != -1 and y != -1:
            coords.append((x, y))

    return coords or None


def find_nearest_non_zero(mask: np.ndarray, point: tuple[int, int]) -> tuple[float, int, tuple[int, int]]:
    """Find the nearest non-zero pixel to a given point."""
    non_zero_indices = np.argwhere(mask != 0)
    if len(non_zero_indices) == 0:
        raise ValueError("No non-zero values found in the mask.")

    distances = np.linalg.norm(non_zero_indices - np.array(point), axis=1)
    min_index = int(np.argmin(distances))
    closest_y, closest_x = non_zero_indices[min_index]
    label_value = int(mask[closest_y, closest_x])

    return float(distances[min_index]), label_value, (int(closest_y), int(closest_x))


def compute_distance_map(mask: np.ndarray, start: tuple[int, int], method: str = "geodesic") -> np.ndarray:
    """
    Compute distance map using different methods.
    
    Parameters:
        mask: Binary mask (non-zero is walkable)
        start: Tuple (y, x) of starting point
        method: One of ['euclidean', 'cityblock', 'chessboard', 'geodesic']
    
    Returns:
        Distance map as ndarray
    """
    if mask[start] == 0:
        raise ValueError(f"Starting point {start} is not inside a labeled object.")

    shape = mask.shape
    walkable = mask > 0
    coords = np.indices(shape).transpose(1, 2, 0).reshape(-1, 2)  # (N, 2) yx
    start_arr = np.array(start).reshape(1, 2)

    if method == "euclidean":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="euclidean").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "cityblock":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="cityblock").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "chessboard":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="chebyshev").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "geodesic":
        mcp = MCP_Geometric(np.where(walkable, 1.0, np.inf), fully_connected=True)
        costs, _ = mcp.find_costs([start])
        costs[~walkable] = np.inf
        return costs

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'euclidean', 'cityblock', 'chessboard', 'geodesic'.")


def process_file(yaml_path: str, mask_path: str, output_dir: str, distance_method:str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(mask_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_{distance_method}.tif")

    mask = rp.load_tczyx_image(mask_path)
    if not mask or mask.data is None:
        print(f"[ERROR] Mask data is None: {mask_path}")
        return

    coords_list = get_coordinates_from_metadata(yaml_path)
    if coords_list is None:
        print(f"[WARNING] No valid ROI coordinates found: {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    try:
        t, c, z = mask.dims.T, mask.dims.C, mask.dims.Z
    except AttributeError:
        print(f"[ERROR] Missing dimension info in: {mask_path}")
        return

    try:
        physical_sizes = mask.physical_pixel_sizes or (None, None, None)
    except Exception as e:
        print(f"[ERROR] Could not retrieve physical pixel sizes: {e}")
        physical_sizes = (None, None, None)

    distance_mask = np.full_like(mask.data, np.inf, dtype=np.float32)

    for t_idx in range(t):
        for c_idx in range(c):
            for z_idx in range(z):
                mask_2d = mask.data[t_idx, c_idx, z_idx, :, :]
                if np.sum(mask_2d) == 0:
                    continue

                for coord in coords_list:
                    if not coord or len(coord) < 2:
                        continue
                    point = (int(round(coord[1])), int(round(coord[0])))

                    if not (0 <= point[0] < mask_2d.shape[0] and 0 <= point[1] < mask_2d.shape[1]):
                        print(f"[WARNING] Point {point} out of bounds in: {mask_path}")
                        continue

                    try:
                        _, _, nearest = find_nearest_non_zero(mask_2d, point)
                        distance_map = compute_distance_map(mask_2d, nearest, method=distance_method)
                        
                        # if t_idx == 0:
                        #     rp.plot_masks((mask_2d, "Mask"), (distance_map, "Distance Map"), metadata=metadata)
                        
                        
                        current = distance_mask[t_idx, c_idx, z_idx, :, :]
                        distance_mask[t_idx, c_idx, z_idx, :, :] = np.fmin(current, distance_map)
                    except ValueError as e:
                        print(f"[WARNING] Skipping point {point} in {mask_path}: {e}")
                        continue

    distance_mask = np.where(np.isinf(distance_mask), 0, distance_mask).astype(np.int32)

    # rp.plot_masks(
    #     (mask.data[0, 2, 0, :, :], "Original Mask"),
    #     (distance_mask[0, 2, 0, :, :], "Distance Map"),
    #     metadata=metadata
    # )

    rp.save_tczyx_image(distance_mask, output_path, dim_order="TCZYX", physical_pixel_sizes=physical_sizes)
    # print(f"[INFO] Saved: {output_path}")


def process_folder(args: argparse.Namespace, parallel:bool) -> None:
    # Build pairs from two glob patterns by substituting '*' with base names
    mask_files = rp.get_files_to_process2(args.mask_search_pattern, args.search_subfolders) if args.mask_search_pattern else []
    yaml_files = rp.get_files_to_process2(args.yaml_search_pattern, args.search_subfolders) if args.yaml_search_pattern else []
    os.makedirs(args.output_folder, exist_ok=True)

    tasks = []

    def has_star(p: str) -> bool:
        return p is not None and ('*' in p or '?' in p or '[' in p)

    if mask_files:
        if not has_star(args.yaml_search_pattern):
            print("[WARNING] --yaml-search-pattern has no wildcard. Expecting single file per mask base name.")
        for mask_path in mask_files:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            yaml_path = args.yaml_search_pattern.replace('*', base_name)
            if os.path.exists(yaml_path):
                tasks.append((yaml_path, mask_path))
            else:
                print(f"[WARNING] Missing YAML for mask base '{base_name}': {yaml_path}")
    elif yaml_files:
        if not has_star(args.mask_search_pattern):
            print("[WARNING] --mask-search-pattern has no wildcard. Expecting single file per YAML base name.")
        for yaml_path in yaml_files:
            base_name = os.path.splitext(os.path.basename(yaml_path))[0]
            mask_path = args.mask_search_pattern.replace('*', base_name)
            if os.path.exists(mask_path):
                tasks.append((yaml_path, mask_path))
            else:
                print(f"[WARNING] Missing mask for YAML base '{base_name}': {mask_path}")
    else:
        print("[ERROR] No files matched either pattern.")
        return

    if not tasks:
        print("[WARNING] No valid YAML/mask pairs were found.")
        return

    if not parallel:
        for yaml_path, mask_path in tqdm(tasks, desc="Processing files", unit="file"):
            try:
                process_file(yaml_path, mask_path, args.output_folder, args.distance_method)
            except Exception as e:
                print(f"[ERROR] Failed to process {yaml_path}: {e}")
        return
    
    else:
        cpu_count = os.cpu_count()
        if not isinstance(cpu_count, int):
            print("[ERROR] Unable to determine CPU count, defaulting to 1 worker.")
            cpu_count = 1
        cpu_count = max(cpu_count - 1, 1)

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_file, y, m, args.output_folder, args.distance_method) for y, m in tasks]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] A file failed to process: {e}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Compute shortest distances from input coordinates to nearest mask objects.")
    parser.add_argument("--mask-search-pattern", type=str, required=False, help="Glob for masks, e.g. './output_masks/*_segmentation.tif'")
    parser.add_argument("--yaml-search-pattern", type=str, required=False, help="Glob for YAMLs, e.g. './output_masks/*_segmentation_metadata.yaml'")
    parser.add_argument("--search-subfolders", action="store_true", help="Enable recursive search (only if pattern doesn't already include '**')")
    parser.add_argument("--distance-method", type=str, default="geodesic", choices=["euclidean", "cityblock", "chessboard", "geodesic"], help="Distance computation method.")
    parser.add_argument("--output-folder", type=str, help="Destination folder for output distance maps.")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")

    args = parser.parse_args()

    if not args.mask_search_pattern and not args.yaml_search_pattern:
        parser.error("Provide at least one of --mask-search-pattern or --yaml-search-pattern.")

    # Default output to the directory of whichever pattern was provided
    if not args.output_folder:
        base_pattern = args.mask_search_pattern or args.yaml_search_pattern
        args.output_folder = os.path.dirname(base_pattern) or "."

    parallel = args.no_parallel == False # inverse

    process_folder(args, parallel=parallel)