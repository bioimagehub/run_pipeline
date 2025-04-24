from bioio import BioImage
import bioio_bioformats
from bioio.writers import OmeTiffWriter

import argparse
import os
import numpy as np
import extract_metadata as meta 
import yaml
import json

def process_image(args: argparse.Namespace):

    input_file_path: str = args.input_file
    # Set output file path
    if args.output_file is None:
        output_folder_path = os.path.dirname(input_file_path) + "_tif"
        os.makedirs(output_folder_path, exist_ok=True)  # Create the output folder if it doesn't exist
        output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + ".tif"    
        output_file_path = os.path.join(output_folder_path, output_file_name)
    else:
        output_file_path: str = args.output_file

    # Load image data
    img = BioImage(input_file_path, reader=bioio_bioformats.Reader)

    metadata = meta.get_metadata(img)  # Extract metadata from the image
    metadata_file_path = os.path.splitext(output_file_path)[0] + ".yaml"
    with open(metadata_file_path, 'w') as f:
        yaml.dump(metadata, f)

    # Check for drift correction
    drift_correction_args = args.drift_correction_args

    if drift_correction_args is None:
        img.save(output_file_path)  # Save the image to the specified output file path
    else:  # Drift correction is requested
        from drift_correct_file_in_RAM import register_image  # Import only if needed
        
        if drift_correction_args == "":  # Use default values if empty string
            drift_correction_args = '{"drift_correct_channel":1,"upsample_factor":20,"space":"real","disambiguate":true,"overlap_ratio":0.9}'

        try:
            drift_correction_params = json.loads(drift_correction_args)

            # Retrieve drift correction parameters with defaults
            drift_correct_channel = drift_correction_params.get('drift_correct_channel', 1)
            upsample_factor = drift_correction_params.get('upsample_factor', 20)
            space = drift_correction_params.get('space', 'real') 
            disambiguate = drift_correction_params.get('disambiguate', True)
            overlap_ratio = drift_correction_params.get('overlap_ratio', 0.9)

            img_np, shifts = register_image(img.dask_data, drift_correct_channel, upsample_factor, space, disambiguate, overlap_ratio)
            
            output_file_path = os.path.splitext(output_file_path)[0] + "_drift.tif"  # Update output file name
            print(f"Drift correction applied. Saving to {output_file_path}")
            
            OmeTiffWriter.save(img_np, output_file_path, dim_order="TCZYX")  # Save the image to the specified output file path
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for drift correction arguments.")
            return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to be processed")
    parser.add_argument("-o", "--output_file", type=str, help="Path for the output TIFF file")
    
    # The drift correction argument can be optional
    parser.add_argument("-d", "--drift_correction_args", nargs='?', const="", default=None, help="Apply drift correction as a JSON string")
    # Example usage: python your_script.py -i input_file.tif -o output_file.tif -d '{"drift_correct_channel":1,"upsample_factor":20,"space":"real","disambiguate":true,"overlap_ratio":0.9}'

    args = parser.parse_args()

    process_image(args)

if __name__ == "__main__":
    main()
