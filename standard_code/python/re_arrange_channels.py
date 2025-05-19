import os
import argparse
import numpy as np
import yaml
from bioio.writers import OmeTiffWriter

# Local imports
import run_pipeline_helper_functions as rp

def rearrange_channels(input_files, new_order, output_folder, yaml_extension=None):
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

            # Process YAML file to also rearrange the channels there
            yaml_file_path = os.path.splitext(input_file)[0] + yaml_extension
            if os.path.isfile(yaml_file_path):
                with open(yaml_file_path, 'r') as f:
                    metadata = yaml.safe_load(f)

                # Rearranging channel names
                if 'Image metadata' in metadata and 'Channels' in metadata['Image metadata']:
                    channels_metadata = metadata['Image metadata']['Channels']
                    # Assuming channels_metadata is a list of dicts like [{'Name': 'Noise'}, {'Name': 'Input Laser Channel'}, ...]
                    rearranged_channels_metadata = [channels_metadata[i] for i in channels_order]
                    metadata['Image metadata']['Channels'] = rearranged_channels_metadata

                # Save the updated metadata back to the YAML file
                with open(yaml_file_path, 'w') as f:
                    yaml.dump(metadata, f)
                print(f"Updated metadata saved to: {yaml_file_path}")
            else:
                print(f"Warning: Metadata file does not exist: {yaml_file_path}")

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
    parser.add_argument('--yaml-extension', required=False, default="_metadata.yaml", type=str, help='Extension of yaml metadata file relative to basename of input file')

    args = parser.parse_args()

    rearrange_channels(args.input_files, args.new_order, args.output_folder, args.yaml_extension)
