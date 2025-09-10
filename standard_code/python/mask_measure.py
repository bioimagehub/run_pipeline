
import numpy as np
import pandas as pd
from skimage import measure
import run_pipeline_helper_functions as rp
import os
from tqdm import tqdm
import argparse


def measure_masked_image(image_path: str, mask_path: str, output_csv: str):
    """
    Measures properties of regions in a mask for each T, C, Z in a TCZYX image.
    Saves results to a CSV file.
    """
    # Load images using rp.load_bioio
    img = rp.load_bioio(image_path)
    mask = rp.load_bioio(mask_path)

    # Handle mask shape logic
    T, C, Z, Y, X = img.shape
    mT, mC, mZ, mY, mX = mask.shape
    if mC != 1:
        raise NotImplementedError(f"Mask must have only one channel. Got {mC}.")
    if (Y, X) != (mY, mX):
        raise ValueError(f"Image and mask XY dimensions must match. Got {Y, X} and {mY, mX}")

    # Time logic
    if T == mT:
        t_map = list(range(T))
        mask_t_map = list(range(mT))
    elif T > 1 and mT == 1:
        t_map = list(range(T))
        mask_t_map = [0] * T
    else:
        raise ValueError(f"Incompatible number of timepoints: image {T}, mask {mT}")

    # Z logic
    if Z == mZ:
        z_map = list(range(Z))
        mask_z_map = list(range(mZ))
    elif Z > 1 and mZ == 1:
        z_map = list(range(Z))
        mask_z_map = [0] * Z
    else:
        raise ValueError(f"Incompatible number of Z planes: image {Z}, mask {mZ}")

    all_results = []
    for t_idx, t in enumerate(t_map):
        mt = mask_t_map[t_idx]
        for c in range(C):
            for z_idx, z in enumerate(z_map):
                mz = mask_z_map[z_idx]
                image_xy = img.data[t, c, z, :, :]
                mask_xy = mask.data[mt, 0, mz, :, :]
                props = measure.regionprops_table(
                    label_image=mask_xy,
                    intensity_image=image_xy,
                    properties=(
                        'label', 'area', 'mean_intensity', 'centroid',
                        'bbox', 'min_intensity', 'max_intensity', 'perimeter',
                    )
                )
                df = pd.DataFrame(props)
                # Calculate sum_intensity and median_intensity manually
                sum_ints = []
                median_ints = []
                for label in df['label']:
                    region_mask = (mask_xy == label)
                    region_pixels = image_xy[region_mask]
                    sum_ints.append(region_pixels.sum())
                    if region_pixels.size > 0:
                        median_ints.append(np.median(region_pixels))
                    else:
                        median_ints.append(np.nan)
                df['sum_intensity'] = sum_ints
                df['median_intensity'] = median_ints
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
    parser = argparse.ArgumentParser(description="Measure regions in a mask for a TCZYX image or batch of images.")
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern for input images, e.g. "folder/*.tif" or "folder/somefile*.tif". Use a single file path for one image.')
    parser.add_argument('--mask-search-pattern', required=True, help='Glob pattern for mask images, e.g. "folder/*_nuc.tif". The * will be replaced with the prefix of the input image before the * in its pattern.')
    parser.add_argument('--output', required=False, default=None, help='Path to output CSV file or output folder (if using batch mode). Defaults to the folder of the mask.')
    parser.add_argument('--search_subfolders', action='store_true', help='Enable recursive search (only relevant if pattern does not already include "**")')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing (default: parallel enabled)')
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

    # Prepare jobs: (image_path, mask_path, out_csv, base_name, mask_name)
    jobs = []
    # Get the part after the * in the mask search pattern (e.g. '_cyt.tif' or '_nuc.tif'), and remove extension
    mask_pattern_after_star = None
    mask_star_idx = args.mask_search_pattern.find('*')
    if mask_star_idx != -1:
        mask_pattern_after_star = args.mask_search_pattern[mask_star_idx+1:]
        mask_pattern_after_star_noext = os.path.splitext(mask_pattern_after_star)[0]
    else:
        mask_pattern_after_star_noext = None
    for image_path in image_files:
        rel_img = image_path[len(input_prefix):] if image_path.startswith(input_prefix) else os.path.basename(image_path)
        rel_img_noext = os.path.splitext(rel_img)[0]
        mask_pattern = args.mask_search_pattern.replace('*', rel_img_noext)
        mask_files = rp.get_files_to_process2(mask_pattern, args.search_subfolders)
        if len(mask_files) == 0:
            print(f"Warning: Mask not found for {image_path} using pattern {mask_pattern}, skipping.")
            continue
        if len(mask_files) > 1:
            print(f"Warning: Multiple masks found for {image_path} using pattern {mask_pattern}, skipping.")
            continue
        mask_path = mask_files[0]
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        # mask_name: part after * in mask pattern, no extension, strip leading _ or .
        if mask_pattern_after_star_noext:
            mask_name = mask_pattern_after_star_noext.lstrip('_').lstrip('.')
        else:
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        if is_batch:
            out_csv = os.path.join(output_folder, f'{base_name}.csv')
        else:
            out_csv = output_folder
        jobs.append((image_path, mask_path, out_csv, base_name, mask_name))


    def measure_masked_image_with_extra_cols(image_path, mask_path, output_csv, base_name, mask_name):
        import pandas as pd
        import tempfile
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        measure_masked_image(image_path, mask_path, temp_csv)
        df = pd.read_csv(temp_csv)
        df['basename'] = base_name
        df['mask_name'] = mask_name
        df.to_csv(output_csv, index=False)

    def process_job(job):
        image_path, mask_path, out_csv, base_name, mask_name = job
        try:
            measure_masked_image_with_extra_cols(image_path, mask_path, out_csv, base_name, mask_name)
            return (image_path, mask_path, out_csv, None)
        except Exception as e:
            return (image_path, mask_path, out_csv, str(e))

    if not args.no_parallel and is_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job) for job in jobs]
            for f in tqdm(as_completed(futures), total=len(jobs), desc='Measuring'):
                result = f.result()
                if result[3] is not None:
                    print(f"Error processing {result[0]}: {result[3]}")
    else:
        for job in tqdm(jobs, desc='Measuring'):
            result = process_job(job)
            if result[3] is not None:
                print(f"Error processing {result[0]}: {result[3]}")


if __name__ == "__main__":
    main()
