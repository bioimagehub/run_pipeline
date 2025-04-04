from bioio import BioImage # conda install conda-forge::bioio-base
import bioio_bioformats # pip install bioio-bioformats
import os
import yaml  #conda install anaconda::pyyaml
import nd2  #conda install -c conda-forge nd2
import argparse
from tqdm import tqdm
import sys

# tested for python 3.10
# used un conda environment called bioio

def get_metadata(file_path, return_image = False):
    """
    Loads image from file, processes it, and writes metadata to a YAML file.
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
            'Image dimensions': {
                'C': c,
                'T': t,
                'X': x,
                'Y': y,
                'Z': z,
            },
            'Physical dimensions': {
                'T_ms': None,
                'X_um': x_um,
                'Y_um': y_um,
                'Z_um': None,
            },
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

    # Open ND2File for ROI processing
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
                    # Calculate position, converted to list for YAML readability
                    cx = getattr(anim_params, 'centerX', 0)
                    cy = getattr(anim_params, 'centerY', 0)
                    
                    center_x = float(0.5 * image_width * (1 + getattr(anim_params, 'centerX', 0)))
                    center_y = float(0.5 * image_height * (1 + getattr(anim_params, 'centerY', 0)))
                    center_z =  None # TODO if need be                


                    box_shape = getattr(anim_params, 'boxShape', None)
                    if box_shape:
                        diameter_x = box_shape.sizeX * image_width
                        diameter_y = box_shape.sizeY * image_height
                        # Assuming symmetric circles, use average diameter
                        average_diameter = (diameter_x + diameter_y) / 2

                                
                        radius = average_diameter/2

                        # Append ROI info
                        image_metadata['Image metadata']['ROIs'].append({
                            'Center X': center_x,
                            'Center Y': center_y,
                            'Shape': 'Circle',
                            'Radius': f"Not validated: {radius}",
                            'Comment': "ROI extraction are under development!!!, Center X/Y looks OK but radius is not validated"
                        })
    return image_metadata


def collapse_filename(file_path, base_folder, delimiter):
    """
    Collapse the path into a single filename while preserving directory information.
    """
    # Remove the base path prefix and replace os path separators with the delimiter
    rel_path = os.path.relpath(file_path, start=base_folder)
    collapsed = delimiter.join(rel_path.split(os.sep))
    return collapsed


def process_folder(folder_path, extension, metadata_extension, search_subfolders, convert_and_save_tif, collapse_delimiter):
    """
    Process all files in folder with the specified extension and save the processed data.
    
    If any file contains the collapse_delimiter in its name, print a warning and stop execution.
    """
    destination_folder = folder_path + "_tif"
    if convert_and_save_tif:
        os.makedirs(destination_folder, exist_ok=True)

    # List of files to be processed within the root folder
    files_to_process = []

    if search_subfolders:
        # Use os.walk to traverse subfolders
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(extension):
                    files_to_process.append(os.path.join(dirpath, filename))
    else:
        # Use os.scandir to list files in the specified directory
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(extension):
                    files_to_process.append(entry.path)
    
    

    # Loop through each file with a progress bar
    for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        print(f"Processing file: {file_path}")
        # Determine the output file path
        if convert_and_save_tif:
            img , metadata = get_metadata(file_path, True)
            collapsed_filename = collapse_filename(file_path, folder_path, collapse_delimiter)
            collapsed_filename = os.path.splitext(collapsed_filename)[0] + ".tif"
            
            tif_file_path = os.path.join(destination_folder, collapsed_filename)

            # Save image as TIFF using PIL
            # Save image using the BioImage class
            img.save(tif_file_path)
            print(f"Saved TIF file: {tif_file_path}")
            
            # Define the yaml file path
            yaml_file_path = os.path.splitext(tif_file_path)[0] + metadata_extension
        else:
            _ , metadata = get_metadata(file_path)                    
            yaml_file_path = os.path.splitext(file_path)[0] + metadata_extension

        # Save metadata
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(metadata, yaml_file, sort_keys=False)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder.")
    parser.add_argument("-m", "--metadata_extension", type=str, default="_metadata.yaml", help="File extension for output metadata file")

    # Optional arguments related to TODO
    parser.add_argument("-R", "--search-subfolders",action="store_true", help="Search for files in subfolders")
    parser.add_argument("-C", "--convert-and-save-tif", action="store_true", help="Convert files to TIF format and save them without subfolders")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Character used to separate directories in collapsed path")

    args = parser.parse_args()

    # Extensions lists
    tested_extensions = [".tif", ".nd2", ".ims"]
    bf_extensions = [
        '.1sc', '.2', '.2fl', '.3', '.4', '.acff', '.afi', '.afm', '.aim', '.al3d', '.ali', '.am', 
        '.amiramesh', '.apl', '.arf', '.avi', '.bif', '.bin', '.bip', '.bmp', '.btf', '.c01', '.cfg', 
        '.ch5', '.cif', '.cr2', '.crw', '.csv', '.cxd', '.dat', '.db', '.dcimg', '.dcm', '.dib', '.dicom',
        '.dm2', '.dm3', '.dm4', '.dti', '.dv', '.eps', '.epsi', '.exp', '.fdf', '.fff', '.ffr', '.fits',
        '.flex', '.fli', '.frm', '.gel', '.gif', '.grey', '.h5', '.hdf', '.hdr', '.hed', '.his', '.htd',
        '.html', '.hx', '.i2i', '.ics', '.ids', '.im3', '.img', '.ims', '.inr', '.ipl', '.ipm', '.ipw',
        '.j2k', '.jp2', '.jpf', '.jpg', '.jpk', '.jpx', '.klb', '.l2d', '.labels', '.lei', '.lif', '.liff',
        '.lim', '.lms', '.lof', '.lsm', '.map', '.mdb', '.mea', '.mnc', '.mng', '.mod', '.mov', '.mrc',
        '.mrcs', '.mrw', '.msr', '.mtb', '.mvd2', '.naf', '.nd', '.nd2', '.ndpi', '.ndpis', '.nef', 
        '.nhdr', '.nii', '.nii.gz', '.nrrd', '.obf', '.obsep', '.oib', '.oif', '.oir', '.omp2info', '.par',
        '.pbm', '.pcoraw', '.pcx', '.pds', '.pgm', '.pic', '.pict', '.png', '.pnl', '.ppm', '.pr3', '.ps',
        '.psd', '.qptiff', '.r3d', '.raw', '.rcpnl', '.rec', '.res', '.scn', '.sdt', '.seq', '.sif',
        '.sld', '.sldy', '.sm2', '.sm3', '.spc', '.spe', '.spi', '.st', '.stk', '.stp', '.svs', '.sxm',
        '.tf2', '.tf8', '.tfr', '.tga', '.tif', '.tiff', '.tnb', '.top', '.txt', '.v', '.vff', '.vms',
        '.vsi', '.vws', '.wat', '.wpi', '.xdce', '.xlef', '.xml', '.xqd', '.xqf', '.xv', '.xys', '.zfp',
        '.zfr', '.zvi', 'etc.'
    ]

    # Ask for folder path if not provided
    while args.folder_path is None or not os.path.isdir(args.folder_path):
        if args.folder_path is not None and not os.path.isdir(args.folder_path):
            print("The provided folder path does not exist.")
        args.folder_path = input("Please provide a valid path to a folder with images: ")

    # Ask for extension if not valid or not provided
    while args.extension is None or args.extension not in bf_extensions:
        if args.extension is not None and args.extension not in bf_extensions:
            print(f"The extension '{args.extension}' is not supported.")
        args.extension = input("Please define the file extension to search for (e.g., .tif): ")

    # Issue a warning if the extension is not in tested_extensions
    if args.extension not in tested_extensions:
        print("Warning: this image format has not been validated. If you have validated that it works, please add the extension to tested_extensions.")

    # Process the folder
    process_folder(args.folder_path, args.extension, args.metadata_extension, args.search_subfolders, args.convert_and_save_tif, args.collapse_delimiter)


if __name__ == "__main__":
    main()



