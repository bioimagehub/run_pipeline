
import numpy as np
import pandas as pd
from skimage import measure
import run_pipeline_helper_functions as rp
import os


def measure_masked_image(image_path: str, mask_path: str, output_csv: str):
    """
    Measures properties of regions in a mask for each T, C, Z in a TCZYX image.
    Saves results to a CSV file.
    """
    # Load images using rp.load_bioio
    img = rp.load_bioio(image_path)
    mask = rp.load_bioio(mask_path)

    # Ensure both images are TCZYX
    if img.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape. Got {img.shape} and {mask.shape}")

    T, C, Z, Y, X = img.shape
    all_results = []

    for t in range(T):
        for c in range(C):
            for z in range(Z):
                image_xy = img[t, c, z, :, :]
                mask_xy = mask[t, c, z, :, :]
                props = measure.regionprops_table(
                    label_image=mask_xy,
                    intensity_image=image_xy,
                    properties=(
                        'label', 'area', 'mean_intensity', 'centroid',
                        'bbox', 'min_intensity', 'max_intensity', 'perimeter',
                    )
                )
                df = pd.DataFrame(props)
                df['T'] = t
                df['C'] = c
                df['Z'] = z
                all_results.append(df)

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        results.to_csv(output_csv, index=False)
    else:
        print("No regions found in mask.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure regions in a mask for a TCZYX image or batch of images.")
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern for input images, e.g. "folder/*.tif" or "folder/somefile*.tif". Use a single file path for one image.')
    parser.add_argument('--mask-search-pattern', required=True, help='Glob pattern for mask images, e.g. "folder/*_nuc.tif". The * will be replaced with the prefix of the input image before the * in its pattern.')
    parser.add_argument('--output', required=False, default=None, help='Path to output CSV file or output folder (if using batch mode). Defaults to the folder of the mask.')
    parser.add_argument('--search_subfolders', action='store_true', help='Enable recursive search (only relevant if pattern does not already include "**")')
    args = parser.parse_args()

    image_files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    is_batch = len(image_files) > 1
    # Set default output folder to mask folder if not provided
    if args.output is None:
        # Use the folder of the mask-search-pattern
        mask_folder = os.path.dirname(args.mask_search_pattern)
        output_folder = mask_folder
    else:
        output_folder = args.output
    if is_batch and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract prefix before * in input pattern and mask pattern
    import re
    def get_prefix(pattern):
        m = re.search(r'(.*)\*', pattern)
        return m.group(1) if m else ''
    input_prefix = get_prefix(args.input_search_pattern)
    mask_prefix = get_prefix(args.mask_search_pattern)

    for image_path in image_files:
        # Find corresponding mask file by matching prefix
        rel_img = image_path[len(input_prefix):] if image_path.startswith(input_prefix) else os.path.basename(image_path)
        rel_img_noext = os.path.splitext(rel_img)[0]
        # Replace * in mask pattern with rel_img_noext (or rel_img)
        mask_pattern = args.mask_search_pattern.replace('*', rel_img_noext)
        mask_files = rp.get_files_to_process2(mask_pattern, args.search_subfolders)
        if len(mask_files) == 0:
            print(f"Warning: Mask not found for {image_path} using pattern {mask_pattern}, skipping.")
            continue
        if len(mask_files) > 1:
            print(f"Warning: Multiple masks found for {image_path} using pattern {mask_pattern}, skipping.")
            continue
        mask_path = mask_files[0]
        if is_batch:
            out_csv = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_measurements.csv')
        else:
            out_csv = output_folder
        print(f"Measuring {image_path} with mask {mask_path} -> {out_csv}")
        try:
            measure_masked_image(image_path, mask_path, out_csv)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    main()
