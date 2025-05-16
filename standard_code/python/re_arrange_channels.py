import os
import argparse
import numpy as np
from bioio.writers import OmeTiffWriter

# Local imports
import run_pipeline_helper_functions as rp


def rearrange_channels(input_files, new_order, output_folder):
    # Convert new order to a list of indexes
    channels_order = [int(i) for i in str(new_order)]
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for input_file in input_files:
        if os.path.isfile(input_file):
            print(f"Processing {input_file}...")
            img = rp.load_bioio(input_file)
            data = img.data  # Assuming it's a numpy array in TCZYX format

            # Ensure the data has the correct dimensions
            if len(data.shape) < 5:
                print(f"Error: {input_file} does not have the expected TCZYX shape.")
                continue
            
            # Rearranging the channels
            rearranged = data[:, channels_order, ...]  # Rearrange channels according to new order
            
            # Preparing to save the output TIFF
            output_tif_file_path = os.path.join(output_folder, os.path.basename(input_file))
            physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

            # Save rearranged image as a TIFF file
            OmeTiffWriter.save(rearranged, output_tif_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)
            print(f"Saved rearranged image as {output_tif_file_path}")
        else:
            print(f"Error: File {input_file} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rearrange channels of TIFF images.')
    parser.add_argument('--new_order', required=True, type=str, help='New order of channels as a number e.g. 3012')
    parser.add_argument('--input-files', required=True, nargs='+', help='List of input TIFF files')
    parser.add_argument('--output-folder', required=True, type=str, help='Output folder for rearranged TIFF files')

    args = parser.parse_args()

    rearrange_channels(args.input_files, args.new_order, args.output_folder)
