import argparse
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import run_pipeline_helper_functions as rp

import yaml

import numpy as np
import pandas as pd
import os



def flatten(
    image_path: str,
    suffix: str,
    mask: Optional[np.ndarray] = None
) -> Tuple[str, pd.DataFrame]:

    image = rp.load_bioio(image_path).data

    if image.ndim != 5:
        raise ValueError(f"Expected 5D image (TCZYX), got shape {image.shape} in {image_path}")

    T, C, Z, Y, X = image.shape
    basename = os.path.basename(image_path).split(".")[0].replace(suffix, "")

    coords = np.indices((T, Z, Y, X)).reshape(4, -1).T
    df = pd.DataFrame(coords, columns=["Frame", "Z", "Y", "X"])

    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_idx = np.where(mask_flat > 0)[0]
        df = df.iloc[valid_idx]

    df["basename"] = basename

    for c in range(C):
        data_flat = image[:, c].reshape(-1)
        if mask is not None:
            data_flat = data_flat[valid_idx]
        colname = os.path.splitext(suffix.lstrip("_"))[0]  # Remove leading underscore and extension

        if colname == "":
            df[f"C{c}"] = data_flat
        elif C == 1:
            df[colname] = data_flat
        else:
            df[f"{colname}_C{c}"] = data_flat

    return basename, df

def process_file(
    basename: str,
    image_files: List[str],
    mask_suffix: str,
    mask_channel: int,
    output_path: str,
    output_suffix: str,
) -> None:
    dfs = []

    # Identify and load the mask, if any
    mask_file = next((f for f in image_files if f.endswith(mask_suffix)), None)
    mask_data = None
    if mask_file:
        # print(f"[INFO] Using mask: {mask_file}")
        mask_img = rp.load_bioio(mask_file).data
        if mask_img.ndim != 5:
            raise ValueError(f"Expected 5D mask image (TCZYX), got shape {mask_img.shape}")
        mask_data = mask_img[:, mask_channel] > 0  # Shape: T x Z x Y x X

    for image_file in image_files:
        suffix = os.path.splitext(os.path.basename(image_file))[0].replace(basename, "")
        _, df = flatten(
            image_path=image_file,
            suffix=suffix,
            mask=mask_data
        )
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print(f"[INFO] No valid data for {basename}")
        return

    # Merge all DataFrames on voxel coordinates
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on=["Frame", "Z", "Y", "X", "basename"],
            how="outer",
            suffixes=('', '_other')
        )

    # Save to CSV
    output_file = os.path.join(output_path, f"{basename}{output_suffix}.csv")
    merged_df.to_csv(output_file, index=False)
    # print(f"[INFO] Saved merged results to {output_file}")



def process_folder(args: argparse.Namespace) -> None:
    os.makedirs(args.output_folder, exist_ok=True)

    image_folders: List[str] = args.image_folders
    image_suffixes: List[List[str]] = args.image_suffixes
    mask_suffix: Optional[str] = args.mask_suffix
    mask_channel: Optional[int] = args.mask_channel

    processing_tasks = {}

    # Make a dictionary where the keuys are the image basenames (without the suffix at the end) and the values are the full paths to the images
    # For example: {"image1": ["/path/to/image1.tif", "/path/to/image1_mask.tif"], "image2": ["/path/to/image2.tif", "/path/to/image2_mask.tif"]}
    for folder, suffixes in zip(image_folders, image_suffixes):
        for suffix in suffixes:
            image_files = rp.get_files_to_process(folder, suffix, search_subfolders=False)
            for image_file in image_files:
                basename = os.path.basename(image_file)[:-len(suffix)]
                if basename not in processing_tasks:
                    processing_tasks[basename] = []
                processing_tasks[basename].append(image_file)
    
    

    tasks = list(processing_tasks.items())
    if args.no_parallel or len(tasks) == 1:
        # Sequential processing
        for basename, image_files in tasks:
            # print()
            # print(f"Processing {basename} with files: {image_files}")
            process_file(basename, image_files, mask_suffix, mask_channel, args.output_folder, args.output_suffix)
    else:
        # Parallel processing
        print(f"[INFO] Running {len(tasks)} tasks in parallel...")
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_file,
                    basename,
                    image_files,
                    mask_suffix,
                    mask_channel,
                    args.output_folder,
                    args.output_suffix
                )
                for basename, image_files in tasks
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Exception in parallel task: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folders", type=str, nargs= "+", required=True,
                        help="Path to folders with images to flatten")
    parser.add_argument("--image-suffixes", type=rp.split_comma_separated_strstring, nargs= "+", required=True,
                        help="Suffixes of input images; CSV-style list per folder")
    
    parser.add_argument("--mask-suffix", type=str, help="Only consider pixels inside a given mask. e.g. _mask.tif. If not given, all pixels are considered valid. Must match the image suffix.")
    parser.add_argument("--mask-channel", type=int, default=0, help="Which channel to use as mask (zero-based index)")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to the output folder for CSVs")
    parser.add_argument("--output-suffix", type=str, default="_results",
                        help="Suffix to add to output CSV files")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    args = parser.parse_args()

    process_folder(args = args)

