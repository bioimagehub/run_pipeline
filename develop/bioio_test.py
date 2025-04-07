from bioio import BioImage
import bioio_bioformats
import os
import yaml
import tqdm
import argparse

def get_metadata(img):
    # Image dimensions
    t, c, z, y, x = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X

    # Physical dimensions
    z_um, y_um, x_um = img.physical_pixel_sizes.Z, img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X

    # Channel info
    channel_info = [str(n) for n in img.channel_names]

    image_metadata = {
        'Image metadata': {
            'Channels': [{'Name': f'Please fill in e.g. {name}'} for name in channel_info],
            'Image dimensions': {'C': c, 'T': t, 'X': x, 'Y': y, 'Z': z},
            'Physical dimensions': {'T_ms': None, 'X_um': x_um, 'Y_um': y_um, 'Z_um': None},
        }
    }

    return image_metadata

def bfconvert(input_file_path, output_file_path, save_tif=True):
    """
    Convert a file using Bio-Formats.
    """
    img = BioImage(input_file_path, reader=bioio_bioformats.Reader)
    
    if save_tif:
        img.save(output_file_path)

    # Extract some simple metadata and write to YAML file
    metadata_path = output_file_path.replace(".tif", "_metadata.yaml")
    metadata = get_metadata(img)
    with open(metadata_path, 'w') as yaml_file:
        yaml.dump(metadata, yaml_file, sort_keys=False)

def process_file(file_path, convert_and_save_tif):
    # Specify output path based on the input file name
    output_file_path = os.path.splitext(file_path)[0] + ".tif"
    print(f"Processing {file_path}...")
    bfconvert(file_path, output_file_path, save_tif=convert_and_save_tif)

def process_folder(args):
    """
    Loop over files in the folder and process each file.
    """
    folder_path = args.folder_path
    extension = args.extension
    convert_and_save_tif = args.convert_and_save_tif

    # Gather files (search subfolders if requested)
    files_to_process = []
    if args.search_subfolders:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(extension):
                    files_to_process.append(os.path.join(dirpath, filename))
    else:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(extension):
                    files_to_process.append(entry.path)

    for file_path in tqdm.tqdm(files_to_process, desc="Processing files", unit="file"):
        process_file(file_path, convert_and_save_tif)  # Directly call process_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder")
    parser.add_argument("-m", "--metadata_extension", type=str, default="_metadata.yaml", help="File extension for output metadata file")
    parser.add_argument("-R", "--search_subfolders", action="store_true", help="Search for files in subfolders")
    parser.add_argument("-C", "--convert_and_save_tif", action="store_true", help="Convert files to TIF format")
    parser.add_argument("--collapse_delimiter", type=str, default="__", help="Delimiter used to collapse file paths")

    args = parser.parse_args()

    # Validate folder path
    while args.folder_path is None or not os.path.isdir(args.folder_path):
        if args.folder_path is not None and not os.path.isdir(args.folder_path):
            print("The provided folder path does not exist.")
        args.folder_path = input("Please provide a valid path to a folder with images: ")

    bf_extensions = ['.tif', '.nd2', '.ims']
    
    # Validate file extension
    while args.extension is None or args.extension not in bf_extensions:
        if args.extension is not None and args.extension not in bf_extensions:
            print(f"The extension '{args.extension}' is not supported.")
        args.extension = input("Please define the file extension to search for (e.g., .tif): ")
    
    process_folder(args)

if __name__ == "__main__":
    main()
