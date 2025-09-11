import argparse
import numpy as np
import multiprocessing
from typing import List, Tuple
import run_pipeline_helper_functions as rp
from bioio.writers import OmeTiffWriter
import re
import os


def process_file(mask1_path: str, mask2_path: str, output_path: str, output_suffix: str = "_subtracted.tif", output_suffix2: str = "_filtered.tif", enforce_one_to_one_overlap: bool = False) -> None:
    """
    Subtracts mask2 from mask1 (set mask1 pixels to 0 where mask2 is nonzero). Optionally, remove all mask regions in mask1 and mask2 that do not overlap any region in the other mask.
    Always saves the subtracted mask1. If ensure_overlap is set, also saves the filtered mask2.
    """
    mask1_img = rp.load_bioio(mask1_path)
    mask2_img = rp.load_bioio(mask2_path)
    physical_pixel_sizes = mask1_img.physical_pixel_sizes if mask1_img.physical_pixel_sizes is not None else (None, None, None)
    mask1 = mask1_img.data[0, 0, :, :, :] if mask1_img.data.ndim == 5 else mask1_img.data
    mask2 = mask2_img.data[0, 0, :, :, :] if mask2_img.data.ndim == 5 else mask2_img.data
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape for subtraction.")


    if enforce_one_to_one_overlap:
        filtered_mask1, filtered_mask2 = ensure_one_to_one_pairing(mask1, mask2)
    else:
        filtered_mask1 = mask1.copy()
        filtered_mask2 = mask2.copy()


    # Subtract mask2 from mask1 (after filtering if needed)
    result_mask1 = np.where(filtered_mask2 > 0, 0, filtered_mask1)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(mask1_path))[0]
    output_tif_file_path = os.path.join(output_path, f"{base_name}{output_suffix}")
    OmeTiffWriter.save(result_mask1, output_tif_file_path, dim_order="ZYX", physical_pixel_sizes=physical_pixel_sizes)
    if enforce_one_to_one_overlap:
        base_name2 = os.path.splitext(os.path.basename(mask2_path))[0]
        output_tif_file_path2 = os.path.join(output_path, f"{base_name2}{output_suffix2}")
        OmeTiffWriter.save(filtered_mask2, output_tif_file_path2, dim_order="ZYX", physical_pixel_sizes=physical_pixel_sizes)





# --- Strict One-to-One Pairing Function ---
def ensure_one_to_one_pairing(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforces a strict one-to-one pairing between mask1 and mask2 objects based on maximum overlap.
    Each mask1 and mask2 object is paired at most once. The label of mask2 is set to the paired mask1 label.
    Unpaired objects are removed. If mask2 is not fully inside mask1, mask1 is grown to encompass mask2.
    Returns filtered mask1 and mask2.
    """
    mask1_labels = np.unique(mask1)
    mask1_labels = mask1_labels[mask1_labels != 0]
    mask2_labels = np.unique(mask2)
    mask2_labels = mask2_labels[mask2_labels != 0]

    # Build overlap matrix: rows=mask2, cols=mask1
    overlap_matrix = np.zeros((len(mask2_labels), len(mask1_labels)), dtype=np.int32)
    for i, lbl2 in enumerate(mask2_labels):
        mask2_bin = (mask2 == lbl2)
        for j, lbl1 in enumerate(mask1_labels):
            mask1_bin = (mask1 == lbl1)
            overlap_matrix[i, j] = np.sum(mask2_bin & mask1_bin)

    # Build all possible (mask2, mask1, overlap) pairs
    pairs = []
    for i, lbl2 in enumerate(mask2_labels):
        for j, lbl1 in enumerate(mask1_labels):
            ov = overlap_matrix[i, j]
            if ov > 0:
                pairs.append((ov, i, j, lbl2, lbl1))
    # Sort by overlap descending
    pairs.sort(reverse=True)

    used_mask1 = set()
    used_mask2 = set()
    assignments = []  # (lbl2, lbl1)
    for ov, i, j, lbl2, lbl1 in pairs:
        if lbl2 not in used_mask2 and lbl1 not in used_mask1:
            assignments.append((lbl2, lbl1))
            used_mask2.add(lbl2)
            used_mask1.add(lbl1)

    # Build filtered masks
    filtered_mask1 = np.zeros_like(mask1)
    filtered_mask2 = np.zeros_like(mask2)
    for lbl2, lbl1 in assignments:
        mask2_bin = (mask2 == lbl2)
        mask1_bin = (mask1 == lbl1)
        filtered_mask1[mask1_bin] = lbl1
        # Set mask2 to the paired mask1 label
        filtered_mask2[mask2_bin] = lbl1
        # If mask2 is not fully inside mask1, grow mask1 to encompass mask2
        if not np.all(mask1_bin[mask2_bin]):
            filtered_mask1[mask2_bin] = lbl1

    return filtered_mask1, filtered_mask2

def process_pair(args):
    mask1_path, mask2_path, output_path, output_suffix, output_suffix2, enforce_one_to_one_overlap = args
    process_file(mask1_path, mask2_path, output_path, output_suffix, output_suffix2, enforce_one_to_one_overlap)


def process_masks(mask1_pattern: str, mask2_pattern: str, output_path: str, output_suffix: str = "_subtracted.tif", parallel: bool = True, search_subfolders: bool = False, output_suffix2: str = "_filtered.tif", enforce_one_to_one_overlap: bool = False):
    mask1_files = rp.get_files_to_process2(mask1_pattern, search_subfolders)
    mask2_files = rp.get_files_to_process2(mask2_pattern, search_subfolders)
    # Match by base name (without extension)
    mask1_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mask1_files}
    mask2_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mask2_files}
    common_basenames = set(mask1_dict.keys()) & set(mask2_dict.keys())
    jobs = [(mask1_dict[bn], mask2_dict[bn], output_path, output_suffix, output_suffix2, enforce_one_to_one_overlap) for bn in common_basenames]
    if parallel:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(process_pair, jobs)
    else:
        for job in jobs:
            process_pair(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract mask2 from mask1 (set mask1 pixels to 0 where mask2 is nonzero). Optionally, remove all mask regions in mask1 and mask2 that do not overlap any region in the other mask.")
    parser.add_argument('--mask1-search-pattern', required=True, help='Glob pattern for mask1 images, e.g. "folder/*_mask1.tif"')
    parser.add_argument('--mask2-search-pattern', required=True, help='Glob pattern for mask2 images, e.g. "folder/*_mask2.tif"')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder where processed images will be saved')
    parser.add_argument('--output-suffix', type=str, default='_subtracted.tif', help='Suffix for the mask1 output file (default: _subtracted.tif)')
    parser.add_argument('--output-suffix2', type=str, default='_filtered.tif', help='Suffix for the mask2 output file if strict one-to-one overlap is set (default: _filtered.tif)')
    parser.add_argument('--enforce-one-to-one-overlap', action='store_true', help='Enforce strict one-to-one pairing between mask1 and mask2 objects (labels in mask2 will match paired mask1).')
    parser.add_argument('--search-subfolders', action='store_true', help='Enable recursive search (only relevant if pattern does not already include "**")')
    parser.add_argument('--no-parallel', action='store_true', help='Do not use parallel processing')
    args = parser.parse_args()

    parallel = not args.no_parallel
    os.makedirs(args.output_folder, exist_ok=True)
    process_masks(
        mask1_pattern=args.mask1_search_pattern,
        mask2_pattern=args.mask2_search_pattern,
        output_path=args.output_folder,
        output_suffix=args.output_suffix,
        parallel=parallel,
        search_subfolders=args.search_subfolders,
        output_suffix2=args.output_suffix2,
        enforce_one_to_one_overlap=args.enforce_one_to_one_overlap
    )
