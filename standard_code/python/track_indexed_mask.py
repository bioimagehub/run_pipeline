import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import center_of_mass
import trackpy as tp
from bioio.writers import OmeTiffWriter
import logging

# Local imports
import bioimage_pipeline_utils as rp


def track_labels_with_trackpy(indexed_masks, channel_zero_base=0, output_mask_path=None):
    T, C, Z, Y, X = indexed_masks.shape
    assert 0 <= channel_zero_base < C, "channel_zero_base out of bounds"

    mask_channel = indexed_masks[:, channel_zero_base]
    records = []

    for t in range(T):
        for z in range(Z):
            slice_mask = mask_channel[t, z]
            labels = np.unique(slice_mask)
            labels = labels[labels != 0]
            for label in labels:
                coords = center_of_mass(slice_mask == label)
                records.append({
                    'frame': t,
                    'z': z,
                    'y': coords[0],
                    'x': coords[1],
                    'label': label,
                    't_local': t * Z + z
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No labels found in the selected channel for tracking.")

    # Create a string buffer to capture the print output
    # Set the logging level for the trackpy logger to ERROR to suppress lower-level messages
    logging.getLogger('trackpy').setLevel(logging.ERROR)


    df_tracked = tp.link_df(df, search_range=10, pos_columns=['x', 'y'], t_column='t_local')

    new_indexed_masks = np.copy(indexed_masks)
    new_indexed_masks[:, channel_zero_base] = 0

    for row in df_tracked.itertuples():
        t, z = row.frame, row.z
        original_label = row.label
        slice_mask = mask_channel[t, z]
        new_indexed_masks[t, channel_zero_base, z][slice_mask == original_label] = row.particle + 1

    if output_mask_path:
        rp.save_tczyx_image(new_indexed_masks, output_mask_path, dim_order="TCZYX")

    return df_tracked, new_indexed_masks


def process_file(input_file_path: str, output_file_path: str, tracking_channel: int):
    masks = rp.load_tczyx_image(input_file_path).data
    df, new_mask = track_labels_with_trackpy(masks, channel_zero_base=tracking_channel, output_mask_path=output_file_path)
    # print(f"Tracked {len(df['particle'].unique())} objects in file: {os.path.basename(input_file_path)}")
    return df


def process_folder_or_file(args: argparse.Namespace):
    recursive = getattr(args, 'search_subfolders', False) or False
    if os.path.isdir(args.input_search_pattern):
        ext = getattr(args, 'extension', '') or '.tif'
        pattern = os.path.join(args.input_search_pattern, '**', f'*{ext}') if recursive else os.path.join(args.input_search_pattern, f'*{ext}')
        base_folder = args.input_search_pattern
    else:
        pattern = args.input_search_pattern
        base_folder = os.path.dirname(pattern) or '.'
    input_files = rp.get_files_to_process2(pattern, recursive)

    if not input_files:
        print("No files found to process.")
        return

    destination_folder = (args.input_search_pattern.rstrip(os.sep) if os.path.isdir(args.input_search_pattern) else base_folder.rstrip(os.sep)) + "_tracked"
    os.makedirs(destination_folder, exist_ok=True)

    def process_single_file(input_file_path):
        output_file_name = rp.collapse_filename(input_file_path, base_folder, args.collapse_delimiter)
        output_file_name = os.path.splitext(output_file_name)[0] + args.output_file_name_extension + ".tif"
        output_file_path = os.path.join(destination_folder, output_file_name)
        process_file(input_file_path, output_file_path, tracking_channel=args.tracking_channel)

    if not args.no_parallel:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)(delayed(process_single_file)(file) for file in tqdm(input_files, desc="Tracking objects", unit="file"))
    else:
        for input_file_path in tqdm(input_files, desc="Tracking objects", unit="file"):
            process_single_file(input_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track labeled objects across time using TrackPy.")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern to search for input files (e.g. './data/*.tif')")
    parser.add_argument("--extension", type=str, default=".tif", help="File extension to search for.")
    parser.add_argument("--search-subfolders", action="store_true", help="Search recursively in subfolders.")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Delimiter for flattening output filenames.")
    parser.add_argument("--tracking-channel", type=int, default=0, help="Channel to use for tracking (default: 0).")
    parser.add_argument("--output-file-name-extension", type=str, default="_tracked", help="Extension to append to output file name (default: '_tracked').")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing.")

    args = parser.parse_args()
    process_folder_or_file(args)
