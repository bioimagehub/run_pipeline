
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from tqdm import tqdm

import bioimage_pipeline_utils as rp

def find_maxima_in_image(image_path: str, output_csv: str, min_distance, threshold_abs, mask_patterns=None, search_subfolders=False):
    """
    Find local maxima in a TCZYX image and save their coordinates to a CSV file.
    min_distance and threshold_abs can be lists (per channel) or single values or expressions (see below).
    threshold_abs supports: absolute value, mean, median, 2*mean, 2*median, or -1 (skip channel).
    If mask_patterns is provided, for each maxima, extract the value at XY in each mask and store in <masklabel>_label columns.
    """
    img = rp.load_tczyx_image(image_path)
    T, C, Z, Y, X = img.shape
    all_results = []
    # Expand min_distance to per-channel list
    if isinstance(min_distance, int) or isinstance(min_distance, float):
        min_distance = [min_distance] * C
    if len(min_distance) == 1:
        min_distance = min_distance * C

    # threshold_abs: allow string expressions per channel
    def parse_thresh_expr(expr, arr):
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, str):
            expr = expr.strip().lower()
            if expr == "mean":
                return float(np.mean(arr))
            elif expr == "median":
                return float(np.median(arr))
            elif expr == "-1":
                return -1
            elif expr.endswith("*mean"):
                try:
                    factor = float(expr.split("*mean")[0])
                    return factor * float(np.mean(arr))
                except Exception:
                    raise ValueError(f"Invalid threshold-abs expression: {expr}")
            elif expr.endswith("*median"):
                try:
                    factor = float(expr.split("*median")[0])
                    return factor * float(np.median(arr))
                except Exception:
                    raise ValueError(f"Invalid threshold-abs expression: {expr}")
            else:
                try:
                    return float(expr)
                except Exception:
                    raise ValueError(f"Invalid threshold-abs value: {expr}")
        raise ValueError(f"Invalid threshold-abs value: {expr}")

    # If threshold_abs is a single value, expand to all channels
    if isinstance(threshold_abs, (int, float, str)):
        threshold_abs = [threshold_abs] * C
    if len(threshold_abs) == 1:
        threshold_abs = threshold_abs * C

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_data = []
    mask_labels = []
    if mask_patterns:
        import re
        for pattern in mask_patterns:
            # Extract mask label from pattern (e.g. *_nuc.tif -> nuc)
            m = re.search(r'\*_(.*?)\.', pattern)
            if m:
                mask_label = m.group(1)
            else:
                star_idx = pattern.find('*')
                if star_idx != -1:
                    after_star = pattern[star_idx+1:]
                    mask_label = os.path.splitext(after_star)[0].lstrip('_').lstrip('.')
                else:
                    mask_label = os.path.splitext(os.path.basename(pattern))[0]
            mask_labels.append(mask_label)
            mask_pattern = pattern.replace('*', base_name)
            mask_files = rp.get_files_to_process2(mask_pattern, search_subfolders)
            if len(mask_files) == 0:
                mask_data.append(None)
                continue
            mask = rp.load_tczyx_image(mask_files[0])
            mask_data.append(mask)

    for t in range(T):
        for c in range(C):
            for z in range(Z):
                image_xy = img.data[t, c, z, :, :]
                # Evaluate threshold for this channel/plane
                thresh = parse_thresh_expr(threshold_abs[c], image_xy)
                if thresh == -1:
                    continue  # skip this channel
                coordinates = peak_local_max(
                    image_xy,
                    min_distance=int(min_distance[c]),
                    threshold_abs=thresh,
                    exclude_border=False
                )
                if coordinates.size > 0:
                    df = pd.DataFrame(coordinates, columns=['Y', 'X'])
                    df['T'] = t
                    df['C'] = c
                    df['Z'] = z
                    df['basename'] = base_name
                    # For each mask, extract value at (Y, X) in the corresponding mask plane
                    if mask_data:
                        for i, mask in enumerate(mask_data):
                            label_col = f"{mask_labels[i]}_label"
                            vals = []
                            if mask is not None:
                                mt = t if mask.shape[0] > t else 0
                                mz = z if mask.shape[2] > z else 0
                                mask_xy = mask.data[mt, 0, mz, :, :]
                                for idx, row in df.iterrows():
                                    y, x = int(row['Y']), int(row['X'])
                                    vals.append(mask_xy[y, x])
                            else:
                                vals = [np.nan] * len(df)
                            df[label_col] = vals
                    all_results.append(df)
    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        results.to_csv(output_csv, index=False)
    else:
        print("No maxima found.")

def main():
    parser = argparse.ArgumentParser(description="Find local maxima in a TCZYX image and save coordinates to CSV.")
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern for input images, e.g. "folder/*.tif" or "folder/somefile*.tif". Use a single file path for one image.')
    parser.add_argument('--output', required=False, default=None, help='Path to output CSV file or output folder (if using batch mode). Defaults to the folder of the input.')
    parser.add_argument('--min-distance', type=rp.split_comma_separated_intstring, default=[10], help='Minimum number of pixels separating peaks. Comma-separated list, one per channel, or single value for all (default: 10)')
    parser.add_argument('--threshold-abs', type=rp.split_comma_separated_strstring, default=[0], help='Minimum intensity of peaks. Comma-separated list, one per channel, or expressions: mean, median, 2*mean, 2*median, -1 (skip channel).')
    parser.add_argument('--mask-search-patterns', nargs='*', default=None, help='List of glob patterns for mask images, e.g. "folder/*_nuc.tif folder/*_cyt.tif". For each maxima, the value at XY in each mask will be stored in a column <masklabel>_label.')
    parser.add_argument('--search-subfolders', action='store_true', help='Enable recursive search (only relevant if pattern does not already include "**")')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing (default: parallel enabled)')
    args = parser.parse_args()

    image_files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    is_batch = len(image_files) > 1
    if args.output is None:
        input_folder = os.path.dirname(args.input_search_pattern)
        output_folder = input_folder
    else:
        output_folder = args.output
    if is_batch and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    jobs = []
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if is_batch:
            out_csv = os.path.join(output_folder, f'{base_name}_maxima.csv')
        else:
            out_csv = output_folder
        jobs.append((image_path, out_csv))

    def process_job(job):
        image_path, out_csv = job
        try:
            find_maxima_in_image(
                image_path, out_csv,
                min_distance=args.min_distance,
                threshold_abs=args.threshold_abs,
                mask_patterns=args.mask_search_patterns,
                search_subfolders=args.search_subfolders
            )
            return (image_path, out_csv, None)
        except Exception as e:
            return (image_path, out_csv, str(e))

    if not args.no_parallel and is_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job) for job in jobs]
            for f in tqdm(as_completed(futures), total=len(jobs), desc='Finding maxima'):
                result = f.result()
                if result[2] is not None:
                    print(f"Error processing {result[0]}: {result[2]}")
    else:
        for job in tqdm(jobs, desc='Finding maxima'):
            result = process_job(job)
            if result[2] is not None:
                print(f"Error processing {result[0]}: {result[2]}")

if __name__ == "__main__":
    main()
