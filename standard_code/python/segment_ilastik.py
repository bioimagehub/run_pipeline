import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from bioio.writers import OmeTiffWriter
from skimage.measure import label


from segment_threshold import LabelInfo, remove_small_or_large_labels, remove_on_edges, fill_holes_indexed
from track_indexed_mask import track_labels_with_trackpy
import run_pipeline_helper_functions as rp





def process_file(args, input_file):
    # Define output file paths
    out_np_file = os.path.join(args.input_folder, os.path.splitext(os.path.basename(input_file))[0] + "_segmentation.np")
    output_tif_file_path = os.path.join(args.output_folder, os.path.basename(os.path.splitext(out_np_file)[0]) + ".tif")

    # subprocess.run([
    #     args.ilastik_path,
    #     '--headless',
    #     f'--project="{args.project_path}"',
    #     '--export_source=simple segmentation',
    #     f'--raw_data="{input_file}"',
    #     f'--output_filename_format="{out_file}"'
    # ])


    f = h5py.File(os.path.splitext(out_np_file)[0] + ".h5", 'r')
    group_key = list(f.keys())[0]
    data = np.array(f[group_key]) # TZYXC

    # Rearrange to TCZYX
    if data.ndim == 5:
        data_tczyx = np.transpose(data, (0, 4, 1, 2, 3))
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}, expected 5D (T, Z, Y, X, C)")

    # C should always be 1 for segmentation if not contunue to next step
    if data_tczyx.shape[1] != 1:
        print(f"Unexpected number of channels: {data_tczyx.shape[1]}, expected 1 for segmentation")
        return
    

    unique_values = np.unique(data_tczyx)
    unique_values = unique_values[unique_values > 1]  # Exclude background (0)

    # return if no unique values found
    if len(unique_values) == 0:
        print(f"No unique values found in {input_file}.")
        OmeTiffWriter.save(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        return

    # make a dataset like data_tczyx byt with the len(unique_values) channels
    output_data = np.zeros((data_tczyx.shape[0], len(unique_values), data_tczyx.shape[2], data_tczyx.shape[3], data_tczyx.shape[4]), dtype=np.uint8)

    # loop over unique values and set the corresponding channel to 255 where the value is value
    for i, value in enumerate(unique_values):
        output_data[:, i, :, :, :] = (data_tczyx[:,0,:,:,:] == value).astype(np.uint8) * 255  # Convert boolean to uint8
        
        # ADD a unique index to each connected component
        for t in range(output_data.shape[0]):
            output_data[t, i, :, :, :] = label(output_data[t, i, :, :, :], connectivity=1)  # Label connected components

    fill_holes_indexed(output_data)
    
    # --- Min/Max size filtering ---
    min_sizes = args.min_sizes if hasattr(args, 'min_sizes') else [0]
    max_sizes = args.max_sizes if hasattr(args, 'max_sizes') else [float('inf')]
    if isinstance(min_sizes, int):
        min_sizes = [min_sizes]
    if isinstance(max_sizes, int):
        max_sizes = [max_sizes]
    channels = list(range(output_data.shape[1]))
    label_info_list = []
    for t in range(output_data.shape[0]):
        for c in channels:
            for z in range(output_data.shape[2]):
                lbls = np.unique(output_data[t, c, z])
                for lbl in lbls:
                    if lbl == 0:
                        continue
                    label_info_list.append(LabelInfo(label_id=lbl, x_center=0, y_center=0, npixels=np.sum(output_data[t, c, z] == lbl), frame=t, channel=c, z_plane=z))

    output_data, label_info_list = remove_small_or_large_labels(output_data, label_info_list, channels=channels, min_sizes=min_sizes, max_sizes=max_sizes)

    # return if no labels left after filtering min max sizes
    if not label_info_list:
        print(f"No labels left after min-max filtering in {input_file}.")
        OmeTiffWriter.save(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        return

    # --- Edge removal ---
    if getattr(args, 'remove_xy_edges', False) or getattr(args, 'remove_z_edges', False):
        output_data, label_info_list = remove_on_edges(output_data, label_info_list, remove_xy_edges=getattr(args, 'remove_xy_edges', False), remove_z_edges=getattr(args, 'remove_z_edges', False))

    # return if no labels left after edge removal
    if not label_info_list:
        print(f"No labels left after edge removal in {input_file}.")
        OmeTiffWriter.save(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        return

    # Tracking
    _, output_data = track_labels_with_trackpy(output_data)
 
    # Save as OME-TIFF
    OmeTiffWriter.save(output_data, output_tif_file_path, dim_order="TCZYX")
    # print(f"Saved segmentation to {output_tif_file_path}")

def process_folder(args: argparse.Namespace) -> None:
    # Find files to process
    files_to_process = rp.get_files_to_process(args.input_folder, args.input_suffix, search_subfolders=False)
    
    if args.no_parallel:
        for input_file in files_to_process:
            process_file(args, input_file)
    else:
        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, args, input_file): input_file for input_file in files_to_process}
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {input_file} generated an exception: {exc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BioImage files.")
    parser.add_argument("--ilastik-path", type=str, required=True, help="Path to the ilastik executable")
    parser.add_argument("--input-folder", type=str, required=True, help="Path to the input folder containing BioImage files")
    parser.add_argument("--input-suffix", type=str, required=True, help="File ending e.g. .tif")
    parser.add_argument("--project-path", type=str, required=True, help="Path to already trained ilp project")
    parser.add_argument("--min-sizes", type=rp.split_comma_separated_intstring, default=[0], help="Minimum size for processing. Comma-separated list, one per channel, or single value for all.")
    parser.add_argument("--max-sizes", type=rp.split_comma_separated_intstring, default=[99999999], help="Maximum size for processing. Comma-separated list, one per channel, or single value for all.")
    parser.add_argument("--remove-xy-edges", action="store_true", help="Remove edges in XY")
    parser.add_argument("--remove-z-edges", action="store_true", help="Remove edges in Z")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--output-folder", type=str, help="Output folder for processed files")

    args = parser.parse_args()

    # Process the folder
    process_folder(args)
