import os

import extract_metadata as meta





# open image 

# Get metadata
meta.get_metadata(file_path)



def process_folder(folder_path, extension, metadata_extension):
    """
    Process all files in folder with the specified extension, and save the processed data.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                metadata = get_metadata(file_path)
                
                yaml_file_path = os.path.splitext(file_path)[0] + metadata_extension
                with open(yaml_file_path, 'w') as yaml_file:
                        yaml.dump(metadata, yaml_file, sort_keys=False)