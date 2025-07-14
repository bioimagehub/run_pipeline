import argparse
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import run_pipeline_helper_functions as rp


def flatten_worker(args) -> Tuple[str, pd.DataFrame]:
    image_path, suffix, mask_folder, mask_suffix, mask_channel = args
    return flatten(image_path, suffix, mask_folder, mask_suffix, mask_channel)


def _save_csv(
    basename: str,
    df: pd.DataFrame,
    suffix: str,
    output_suffix: str,
    output_path: str
) -> None:
    if df.empty:
        print(f"[INFO] No data to save for: {basename}")
        return

    out_csv = os.path.join(output_path, f"{basename}{output_suffix}.csv")

    if os.path.exists(out_csv):
        try:
            old_df = pd.read_csv(out_csv)
            shared_cols = list(set(df.columns) & set(old_df.columns) & {"basename", "T", "C", "Z", "Y", "X"})
            if len(shared_cols) < 3:
                print(f"[WARN] Too few shared columns to merge: {shared_cols}. Overwriting {out_csv}")
            else:
                df = pd.merge(old_df, df, on=shared_cols, how="outer")
        except Exception as e:
            print(f"[ERROR] Failed to merge with {out_csv}: {e}")

    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved or updated: {out_csv}")

def flatten(
    image_path: str,
    suffix: str,
    mask_folder: str,
    mask_suffix: str,
    mask_channel: int
) -> Tuple[str, pd.DataFrame]:
    img = rp.load_bioio(image_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return "", pd.DataFrame()

    basename = os.path.basename(image_path).replace(suffix, "")
    img_data = img.data

    if img_data.ndim != 5:
        print(f"[ERROR] Image does not have 5 dimensions (T, C, Z, Y, X): {image_path}, got {img_data.shape}")
        return basename, pd.DataFrame()

    mask_path = os.path.join(mask_folder, basename + mask_suffix)
    if not os.path.exists(mask_path):
        print(f"[WARN] Mask not found for {basename}: {mask_path}")
        return basename, pd.DataFrame()

    mask_img = rp.load_bioio(mask_path)
    if mask_img is None:
        print(f"[WARN] Failed to load mask for {basename}")
        return basename, pd.DataFrame()

    mask_data = mask_img.data
    try:
        mask_data = mask_data[:, mask_channel, ...]  # shape: (T, Z, Y, X)
    except IndexError:
        print(f"[ERROR] mask_channel {mask_channel} is out of bounds for mask shape {mask_img.data.shape}")
        return basename, pd.DataFrame()

    # Build valid mask across all C (same coords)
    T, C, Z, Y, X = img_data.shape
    valid_mask = (img_data != 0) & (mask_data[:, np.newaxis, ...] != 0)  # shape (T, C, Z, Y, X)
    valid_voxels = np.argwhere(np.any(valid_mask, axis=1))  # where any channel has a valid pixel

    if valid_voxels.size == 0:
        print(f"[INFO] No valid voxels found in {basename}")
        return basename, pd.DataFrame()

    df = pd.DataFrame(valid_voxels, columns=["T", "Z", "Y", "X"])

    for ch in range(C):
        ch_mask = valid_mask[:, ch, ...]
        ch_values = img_data[:, ch, ...][ch_mask]
        ch_coords = np.argwhere(ch_mask)

        # Create a temp DataFrame and merge on voxel coords
        ch_df = pd.DataFrame(ch_coords, columns=["T", "Z", "Y", "X"])
        ch_df[f"value_{suffix}_ch{ch}"] = ch_values
        df = pd.merge(df, ch_df, on=["T", "Z", "Y", "X"], how="left")

    df["basename"] = basename
    return basename, df


def process_folder(
    image_folders: List[str],
    image_suffixes: List[List[str]],
    output_path: str,
    output_suffix: str,
    parallel: bool,
    mask_folder: Optional[str] = None,
    mask_suffix: Optional[str] = None,
    mask_channel: Optional[int] = None
) -> None:
    os.makedirs(output_path, exist_ok=True)

    for folder, suffixes in zip(image_folders, image_suffixes):
        for suffix in suffixes:
            print(f"[INFO] Processing folder: {folder} with suffix: {suffix}")
            image_paths = rp.get_files_to_process(folder, suffix, search_subfolders=False)
            if not image_paths:
                print(f"[WARN] No images found in folder {folder} with suffix {suffix}.")
                continue

            tasks = [(p, suffix, mask_folder, mask_suffix, mask_channel) for p in image_paths]

            if parallel:
                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(flatten_worker, task) for task in tasks]
                    for future in as_completed(futures):
                        basename, df = future.result()
                        _save_csv(basename, df, suffix, output_suffix, output_path)
            else:
                for task in tasks:
                    basename, df = flatten_worker(task)
                    _save_csv(basename, df, suffix, output_suffix, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask-folder", type=str, required=True,
                        help="Path to folder with binary mask images")
    parser.add_argument("--mask-suffix", type=str, required=True,
                        help="Suffix of mask files (e.g. _mask.tif)")
    parser.add_argument("--mask-channel", type=int, required=True,
                        help="Which channel to use as mask (zero-based index)")

    parser.add_argument("--input-folders", nargs='+', type=str, required=True,
                        help="List of input folders containing images")
    parser.add_argument("--input-suffix", nargs='+', type=rp.split_comma_separated_strstring, required=True,
                        help="Suffixes of input images; CSV-style list per folder")

    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to the output folder for CSVs")
    parser.add_argument("--output-suffix", type=str, default="_results",
                        help="Suffix to add to output CSV files")

    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    process_folder(
        image_folders=args.input_folders,
        image_suffixes=args.input_suffix,
        output_path=args.output_folder,
        output_suffix=args.output_suffix,
        parallel=not args.no_parallel,
        mask_folder=args.mask_folder,
        mask_suffix=args.mask_suffix,
        mask_channel=args.mask_channel
    )
