import argparse
from tqdm import tqdm
import json
import numpy as np
from bioio.writers import OmeTiffWriter
import os
from joblib import Parallel, delayed
import yaml

from skimage import io
from skimage.measure import block_reduce
import numpy as np


# local imports
import bioimage_pipeline_utils as rp  
from extract_metadata import get_all_metadata

from skimage.transform import resize



def process_file(input_file_path: str, output_tif_file_path: str, merge_channels: str, output_format: str = "tif", output_dim_order: str = "TCZYX", scale:float = 1) -> None:
    try:
        # Define output file names
        input_metadata_file_path:str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
        output_metadata_file_path:str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"  
        
        # Check if metadata file exists and open/make
        if os.path.exists(input_metadata_file_path):
            with open(input_metadata_file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = get_all_metadata(img)
            #TODO Prompt the user to fill in the channel names etc
            # This is the core metadata without drift correction info
            with open(input_metadata_file_path, 'w') as f:
                yaml.dump(metadata, f)
        
        # Load image and metadata
        img = rp.load_tczyx_image(input_file_path)
        physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

        img_np = img.data  # TCZYX
        
        # Parse merge_channels string to list of channel groups
        channel_groups = json.loads(merge_channels)  # e.g. [[0,1], [2], [3,4]]
        channel_groups = [g if isinstance(g, list) else [g] for g in channel_groups] # Ensure all groups are lists
        #print(channel_groups)

        output_channels = []

        for group in channel_groups:
            group = list(group)  # ensure it's a list in case it's a tuple
            if len(group) == 1:
                # Keep as-is
                output = img_np[:, group[0]:group[0]+1, :, :, :]
            else:
                # Sum-project along channel axis
                output = img_np[:, group, :, :, :].sum(axis=1, keepdims=True, dtype=img_np.dtype)
            output_channels.append(output)

        if os.path.exists(output_tif_file_path):
            os.remove(output_tif_file_path)

        # Concatenate all processed channel groups
        out_img = np.concatenate(output_channels, axis=1)  # TCZYX



        # Apply output scaling if specified
        if scale < 1:
            print(f"[INFO] Downsampling image by a factor of {1/scale}")
            print(f"[INFO] Original shape: {out_img.shape}")
            out_img = block_reduce(out_img, block_size=(1, 1, 1, int(1/scale), int(1/scale)), func=np.max)
            print(f"[INFO] New shape after downsampling: {out_img.shape}")

            # Scale physical_pixel_sizes
            physical_pixel_sizes = tuple(p * scale for p in physical_pixel_sizes)

        elif scale > 1:
            print(f"[INFO] Upsampling image by a factor of {scale}")
            print(f"[INFO] Original shape: {out_img.shape}")
            t, c, z, y, x = out_img.shape
            new_y = int(y * scale)
            new_x = int(x * scale)
            up_img = np.zeros((t, c, z, new_y, new_x), dtype=out_img.dtype)
            for ti in range(t):
                for ci in range(c):
                    for zi in range(z):
                        up_img[ti, ci, zi] = resize(
                            out_img[ti, ci, zi],
                            (new_y, new_x),
                            order=0,  # nearest-neighbor
                            preserve_range=True,
                            anti_aliasing=False
                        ).astype(out_img.dtype)
            out_img = up_img
            print(f"[INFO] New shape after upsampling: {out_img.shape}")

            # Scale physical_pixel_sizes
            physical_pixel_sizes = tuple(p * scale for p in physical_pixel_sizes)

        

        # Save output
        if output_format == "tif":
            rp.save_tczyx_image(out_img, output_tif_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)
        elif output_format == "npy":
            # Rearrange dimensions if needed
            if output_dim_order == "TZYXC":
                # out_img is TCZYX, convert to TZYXC
                out_img = np.transpose(out_img, (0,2,3,4,1))
            np.save(os.path.splitext(output_tif_file_path)[0] + ".npy", out_img)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Save metadata to YAML file
        metadata["Merge channels"] = {
            "Method": "Sum projection",
            "Merge_channels": merge_channels,
            "Output_format": output_format,
            "Output_dim_order": output_dim_order,
        }
        with open(output_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
        return

def process_folder(args: argparse.Namespace, use_parallel = True) -> None:
        
    # Find files to process using glob pattern (prefer --input-search-pattern, fallback to --input-folder)
    if hasattr(args, 'input_search_pattern') and args.input_search_pattern:
        pattern = args.input_search_pattern
    else:
        pattern = os.path.join(args.input_folder, '*.tif')
    files_to_process = rp.get_files_to_process2(pattern, False)
    # files_to_process = [files_to_process[0]] # For debugging
    
    # Make output folder
    os.makedirs(args.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist


    if use_parallel: # Process each file in parallel
        Parallel(n_jobs=-1)(
            delayed(process_file)(input_file_path,
                                  os.path.join(args.output_folder, os.path.basename(input_file_path)),
                                  args.merge_channels,
                                  args.output_format,
                                  args.output_dim_order,
                                  args.output_scale)
            for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file")
        )
    else: # Process each file sequentially        
        for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            # Define output file name
            output_tif_file_path:str = os.path.join(args.output_folder, os.path.basename(input_file_path))
            # process file
            
            process_file(input_file_path, output_tif_file_path, args.merge_channels, args.output_format, args.output_dim_order, args.output_scale)  # Process each file
            


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Process BioImage files.")
    parser.add_argument("--input-search-pattern", type=str, required=False, help="Glob pattern for input images, e.g. 'folder/*.tif'")
    parser.add_argument("--input-folder", type=str, required=False, help="Deprecated: input folder (use --input-search-pattern)")
    parser.add_argument("--output-folder", type=str, required=False, help="Path to save the processed files")
    parser.add_argument("--merge-channels", type=str, required=True, help="E.g. '[[0,1,2,3], 4]' to merge channels 0,1,2,3 and keep channel 4 and remove  >4")
    parser.add_argument("--output-format", type=str, choices=["tif", "npy"], default="tif", help="Output format: 'tif' (OME-TIFF) or 'npy' (NumPy array)")
    parser.add_argument("--output-dim-order", type=str, choices=["TCZYX", "TZYXC"], default="TCZYX", help="Output dimension order for npy: 'TCZYX' (default) or 'TZYXC'")
    parser.add_argument("--output-scale", type=float, default=1, help="Over or under-sampling factor for the output image. Default is 1 (no scaling).")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")

    args = parser.parse_args()

    # Check if output folder path is provided, if not set default
    if args.output_folder is None:
        base = os.path.dirname(args.input_search_pattern) if args.input_search_pattern else args.input_folder
        args.output_folder = os.path.join(base, "_merged")

    # Process the folder
    process_folder(args, use_parallel = not args.no_parallel)

    # Emit standardized output glob patterns for downstream consumption
    dest_norm = args.output_folder.replace("\\", "/")
    if args.output_format == "tif":
        print(f"OUTPUT_GLOB_TIFF: {dest_norm}/**/*.tif")
        print(f"OUTPUT_GLOB_METADATA: {dest_norm}/**/*_metadata.yaml")
        patterns = {"tiff": f"{dest_norm}/**/*.tif", "metadata": f"{dest_norm}/**/*_metadata.yaml", "next_input_search_pattern": f"{dest_norm}/**/*.tif"}
    else:
        print(f"OUTPUT_GLOB_NPY: {dest_norm}/**/*.npy")
        print(f"OUTPUT_GLOB_METADATA: {dest_norm}/**/*_metadata.yaml")
        patterns = {"npy": f"{dest_norm}/**/*.npy", "metadata": f"{dest_norm}/**/*_metadata.yaml", "next_input_search_pattern": f"{dest_norm}/**/*.npy"}
    try:
        import json
        with open(os.path.join(args.output_folder, "_output_patterns.json"), "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2)
    except Exception:
        pass

