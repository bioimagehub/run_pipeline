#!/usr/bin/env python
import os
import sys
import yaml
import nd2
import argparse
import subprocess
from tqdm import tqdm

# Import bioio modules (these trigger Java/BioFormats logging)
from bioio import BioImage  # conda install conda-forge::bioio-base
import bioio_bioformats  # pip install bioio-bioformats


def get_metadata(file_path, return_image=False):
    """
    Loads image from file, processes it, and returns metadata (and image if requested).
    """
    img = BioImage(file_path, reader=bioio_bioformats.Reader)

    # Image dimensions
    t, c, z, y, x = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X

    # Physical dimensions
    z_um, y_um, x_um = img.physical_pixel_sizes.Z, img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X

    # Channel info
    channel_info = [str(n) for n in img.channel_names]

    # Extract metadata
    image_size_kb = os.path.getsize(file_path) / 1024  # Convert bytes to kilobytes

    image_metadata = {
        'Image metadata': {
            'Channels': [{'Name': f'Please fill in e.g. {name}'} for name in channel_info],
            'Image dimensions': {'C': c, 'T': t, 'X': x, 'Y': y, 'Z': z},
            'Physical dimensions': {'T_ms': None, 'X_um': x_um, 'Y_um': y_um, 'Z_um': None},
            'Size (kb)': image_size_kb,
        }
    }

    if file_path.lower().endswith(".nd2"):
        image_metadata = get_roi_info(file_path, image_metadata)
    
    if return_image:
        return img, image_metadata
    else:
        del img
        return None, image_metadata


def get_roi_info(file_path, image_metadata):
    # Ensure ROIs field only exists for ND2 files
    image_metadata['Image metadata']['ROIs'] = []
    with nd2.ND2File(file_path) as nd2_file:
        for roi_id, roi in nd2_file.rois.items():
            if roi.info.shapeType != nd2.structures.RoiShapeType.Circle:
                print(roi.info.shapeType)
                image_metadata['Image metadata']['ROIs'].append({
                    'Shape': str(roi.info.shapeType),
                    'Comment': "This shape has not been incorporated yet"
                })
            else:
                anim_params = roi.animParams[0] if roi.animParams else None
                image_width = image_metadata["Image metadata"]["Image dimensions"]["X"]
                image_height = image_metadata["Image metadata"]["Image dimensions"]["Y"]
                if anim_params:
                    center_x = float(0.5 * image_width * (1 + getattr(anim_params, 'centerX', 0)))
                    center_y = float(0.5 * image_height * (1 + getattr(anim_params, 'centerY', 0)))
                    box_shape = getattr(anim_params, 'boxShape', None)
                    if box_shape:
                        diameter_x = box_shape.sizeX * image_width
                        diameter_y = box_shape.sizeY * image_height
                        average_diameter = (diameter_x + diameter_y) / 2
                        radius = average_diameter / 2
                        image_metadata['Image metadata']['ROIs'].append({
                            'Center X': center_x,
                            'Center Y': center_y,
                            'Shape': 'Circle',
                            'Radius': f"Not validated: {radius}",
                            'Comment': "ROI extraction under development"
                        })
    return image_metadata


def collapse_filename(file_path, base_folder, delimiter):
    """
    Collapse the file path into a single filename, preserving directory info.
    """
    rel_path = os.path.relpath(file_path, start=base_folder)
    collapsed = delimiter.join(rel_path.split(os.sep))
    return collapsed


def process_file(file_path, folder_path, convert_and_save_tif, metadata_extension, collapse_delimiter):
    """
    Process a single file: save TIFF (if requested) and metadata.
    This function is intended to run in a subprocess.
    """
    tif_file_path = None
    if convert_and_save_tif:
        img, metadata = get_metadata(file_path, True)
        collapsed_filename = collapse_filename(file_path, folder_path, collapse_delimiter)
        collapsed_filename = os.path.splitext(collapsed_filename)[0] + ".tif"
        destination_folder = folder_path + "_tif"
        os.makedirs(destination_folder, exist_ok=True)
        tif_file_path = os.path.join(destination_folder, collapsed_filename)
        
        
        img.save(tif_file_path)
        
        yaml_file_path = os.path.splitext(tif_file_path)[0] + metadata_extension
    else:
        _, metadata = get_metadata(file_path)
        yaml_file_path = os.path.splitext(file_path)[0] + metadata_extension

    # Write metadata to YAML file.
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(metadata, yaml_file, sort_keys=False)
    sys.exit(0)


def process_folder_main(args):
    """
    Loop over files in the folder and, for each file, spawn a subprocess that calls
    the BioImage code with output suppressed.
    """
    folder_path = args.folder_path
    extension = args.extension
    metadata_extension = args.metadata_extension
    convert_and_save_tif = args.convert_and_save_tif
    collapse_delimiter = args.collapse_delimiter

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

    for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        # Build the command to run the same script in process-file mode.
        cmd = [
            sys.executable, __file__,
            "--process-file",
            "--file_path", file_path,
            "--folder_path", folder_path,
            "--metadata_extension", metadata_extension,
            "--collapse_delimiter", collapse_delimiter
        ]
        if convert_and_save_tif:
            cmd.append("--convert_and_save_tif")
        # Call the subprocess and suppress its stdout/stderr.
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"Processed {file_path}")


def main():
    parser = argparse.ArgumentParser()
    # Arguments for folder processing (default mode)
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder")
    parser.add_argument("-m", "--metadata_extension", type=str, default="_metadata.yaml", help="File extension for output metadata file")
    parser.add_argument("-R", "--search_subfolders", action="store_true", help="Search for files in subfolders")
    parser.add_argument("-C", "--convert_and_save_tif", action="store_true", help="Convert files to TIF format")
    parser.add_argument("--collapse_delimiter", type=str, default="__", help="Delimiter used to collapse file paths")
    # A flag that tells the script to process a single file (used by subprocess calls)
    parser.add_argument("--process-file", action="store_true", help=argparse.SUPPRESS)
    # Argument used only in process-file mode:
    parser.add_argument("--file_path", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.process_file:
        if not args.file_path:
            sys.exit("Error: --file_path is required in process-file mode.")
        convert_flag = getattr(args, "convert_and_save_tif", False)
        process_file(
            file_path=args.file_path,
            folder_path=args.folder_path,
            convert_and_save_tif=convert_flag,
            metadata_extension=args.metadata_extension,
            collapse_delimiter=args.collapse_delimiter
        )
    else:
        while args.folder_path is None or not os.path.isdir(args.folder_path):
            if args.folder_path is not None and not os.path.isdir(args.folder_path):
                print("The provided folder path does not exist.")
            args.folder_path = input("Please provide a valid path to a folder with images: ")
        bf_extensions = ['.tif', '.nd2', '.ims']
        while args.extension is None or args.extension not in bf_extensions:
            if args.extension is not None and args.extension not in bf_extensions:
                print(f"The extension '{args.extension}' is not supported.")
            args.extension = input("Please define the file extension to search for (e.g., .tif): ")
        process_folder_main(args)


if __name__ == "__main__":
    main()
