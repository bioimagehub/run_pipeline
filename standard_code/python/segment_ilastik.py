import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
import logging
import tempfile
import time
import uuid

import bioimage_pipeline_utils as rp

import tifffile

# Add tqdm for progress bars (fallback to no-op if not installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# Configure logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

def process_file(args, input_file):
    # Removed per-file print; progress is handled by tqdm in process_folder
    # Define output file paths
    
    in_dir = os.path.dirname(input_file)
    h5_output_path = os.path.join(in_dir, os.path.splitext(os.path.basename(input_file))[0] + "_segmentation.h5")
    tif_output_path = os.path.join(args.output_folder, os.path.basename(os.path.splitext(h5_output_path)[0]) + ".tif")

    # Prepare a unique Ilastik log directory per process to avoid Windows file-lock races
    unique_log_dir = os.path.join(tempfile.gettempdir(), f"ilastik_logs_{uuid.uuid4().hex}")
    os.makedirs(unique_log_dir, exist_ok=True)
    env = os.environ.copy()
    env["ILASTIK_LOG_DIR"] = unique_log_dir

    # Run Ilastik headless, show all output for debugging
    try:
        proc = subprocess.run([
            args.ilastik_path,
            '--headless',
            f'--project={args.project_path}',
            #'--export_source=Probabilities',
            input_file
            # f'--raw_data={input_file}',
            # f'--output_filename_format={h5_output_path}'
        ], capture_output=True, text=True, env=env)
    except FileNotFoundError as e:
        logging.error(f"Failed to run Ilastik executable: {args.ilastik_path}")
        logging.error(f"Error: {e}")
        logging.error(f"Please check that the ilastik-path is correct and the file exists")
        raise

    # Print all stdout and stderr output from Ilastik
    if proc.stdout:
        print(f"[ILASTIK STDOUT] {os.path.basename(input_file)}:")
        print(proc.stdout)
    if proc.stderr:
        print(f"[ILASTIK STDERR] {os.path.basename(input_file)}:")
        print(proc.stderr)

    # Give the filesystem a brief moment for the output to appear
    if not os.path.exists(h5_output_path):
        for _ in range(10):  # wait up to ~10s
            time.sleep(1)
            if os.path.exists(h5_output_path):
                break
    
    # Check if Ilastik produced output
    if not os.path.exists(h5_output_path):
        logging.error(f"Ilastik failed to produce output for: {input_file}")
        logging.error(f"Expected H5 output at: {h5_output_path}")
        return
    
    # Read Ilastik probability output
    try:
        with h5py.File(h5_output_path, 'r') as f:
            group_key = list(f.keys())[0]
            data = np.array(f[group_key])  # Expected: TZYXC (probabilities per class)
    except Exception as e:
        logging.error(f"Failed to read H5 file: {h5_output_path}")
        logging.error(f"Error: {e}")
        raise

    # Rearrange to TCZYX (be tolerant to 4D TZYX without channel)
    if data.ndim == 5:
        # TZYXC -> TCZYX
        data_tczyx = np.transpose(data, (0, 4, 1, 2, 3))
    elif data.ndim == 4:
        # Assume T, Z, Y, X -> add singleton channel dimension
        data_tczyx = data[:, np.newaxis, :, :, :]
    else:
        logging.error(f"Unexpected data shape: {data.shape}, expected 5D (TZYXC) or 4D (TZYX)")
        return
    
    # Convert probability maps to binary mask
    # If multi-channel (e.g., [background, foreground]), use the foreground channel (last one)
    # If single channel, threshold directly
    if data_tczyx.shape[1] > 1:
        # Use last channel as foreground probability (common Ilastik convention)
        probabilities = data_tczyx[:, -1:, :, :, :]  # Keep as TCZYX with C=1
    else:
        probabilities = data_tczyx
    
    # Threshold probabilities at 0.5 to create binary mask
    # Convert to uint8 with values 0 and 255 for ImageJ compatibility
    binary_mask = (probabilities > 0.5).astype(np.uint8) * 255
    
    # For ImageJ compatibility, squeeze out the channel dimension
    output_array = binary_mask[:, 0, :, :, :]  # TZYX
    
    # Save as ImageJ TIF with proper metadata
    try:
        tifffile.imwrite(
            tif_output_path,
            output_array,
            imagej=True,
            metadata={'axes': 'TZYX'},
            compression='deflate'
        )
        logging.info(f"Saved binary mask: {tif_output_path}")
    except Exception as e:
        logging.error(f"Failed to save TIF file: {tif_output_path}")
        logging.error(f"Error: {e}")
        raise
    return


def process_folder(args: argparse.Namespace) -> None:
    # Find files to process using glob pattern
    pattern = args.input_search_pattern
    files_to_process = rp.get_files_to_process2(pattern, False)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if args.no_parallel:
        for input_file in tqdm(files_to_process, desc="Processing files", unit="file"):
            process_file(args, input_file)
    else:
        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, args, input_file): input_file for input_file in files_to_process}
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files", unit="file"):
                input_file = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {input_file} generated an exception: {exc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ilastik segmentation in parallel and save masks as ImageJ TIFFs.")
    parser.add_argument("--ilastik-path", type=str, required=True, help="Path to the ilastik executable")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input images, e.g. './input_Ilastik/*.h5'")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to save the output mask TIF files")
    parser.add_argument("--project-path", type=str, required=True, help="Path to trained Ilastik project file (.ilp)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--qc-key", type=str, default="segment_ilastik", help="Key to use in QC YAML files (default: segment_ilastik)")




    args = parser.parse_args()
    if not os.path.exists(args.project_path):
        print(f"Please start Ilastik and save the project file to: {args.project_path}")
    while not os.path.exists(args.project_path):
        time.sleep(10)
        continue

        
    # Process the folder
    process_folder(args)
