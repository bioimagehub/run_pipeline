from nd2reader import ND2Reader
import os
import yaml  # Assuming you want to store metadata in YAML format

def process_file(file_path):
    # Read the ND2 file and access metadata
    with ND2Reader(file_path) as images:
        metadata = images.metadata
        
        # Get the size of a pixel in micrometers
        pixel_microns = metadata.get('pixel_microns', 1)  # Default to 1 if not found
        
        # Access ROIs from metadata
        rois = metadata.get('rois', [])
        
        roi_info = []
        
        # Loop through each ROI to extract position and size information
        for roi in rois:
            positions = roi.get('positions', [])
            sizes = roi.get('sizes', [])
            shape = roi.get('shape', 'unknown')  # Extract shape if available
            roi_type = roi.get('type', 'unknown')  # Extract type if available
            
            # Convert position and size from micrometers to pixels
            for pos, size in zip(positions, sizes):
                pos_pixels = [p / pixel_microns for p in pos]
                size_pixels = [s / pixel_microns for s in size]

                roi_pixels = {
                    "Roi": {
                        "Positions": {
                            "x": pos_pixels[0],
                            "y": pos_pixels[1]
                        },
                        "Size": {
                            "x": size_pixels[0],
                            "y": size_pixels[1]
                        },
                        "Shape": shape,
                        "Type": roi_type
                    }
                }

                roi_info.append(roi_pixels)
                
                # Print position and size in pixels
                print(f"Position in pixels: {pos_pixels}")
                print(f"Size in pixels: {size_pixels}")
    return roi_info

def process_folder(folder_path, extension, metadata_extension):
    """
    Process all files in folder with the specified extension, and save the processed data.
    """
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                roi_info = process_file(file_path)
                
                yaml_file_path = os.path.splitext(file_path)[0] + metadata_extension
                
                if not os.path.exists(yaml_file_path):
                    with open(yaml_file_path, 'w') as yaml_file:
                        yaml.dump(roi_info, yaml_file)
                else:
                    with open(yaml_file_path, 'r') as yaml_file:
                        existing_data = yaml.safe_load(yaml_file)
                        
                    existing_data['ROI'] = roi_info  # Assuming you want to update ROI part
                    
                    with open(yaml_file_path, 'w') as yaml_file:
                        yaml.dump(existing_data, yaml_file)
          
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed.")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder.")
    parser.add_argument("-m", "--metadata_extension", type=str, default="_metadata.yaml", help="File extension of image metadata file")
    args = parser.parse_args()
          
    if args.folder_path is None:
        args.folder_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input"
    
    if args.extension is None:
        args.extension = ".nd2"
    
    process_folder(args.folder_path, args.extension, args.metadata_extension)
