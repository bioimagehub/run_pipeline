from bioio import BioImage
import bioio_bioformats
import os
import yaml
import sys


# def load_file(file_path, use_dask=True):
#     """
#     Load image from file.
#     If use_dask is True, returns a dask array with dimensions (T, C, Z, Y, X).
#     """
#     img = BioImage(file_path, reader=bioio_bioformats.Reader)
#     data = img.dask_data if use_dask else img.data
#     del img
#     print(f"Image dimensions: {data.shape}")
#     return data

def process_file(file_path):
    """
    Loads image from file, processes it, and writes metadata to a YAML file.
    """
      
    img = BioImage(file_path, reader=bioio_bioformats.Reader)
       
    # Image dimensions (from shape instead)
    t, c, z, y, x = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X
    # print(t, c, z, y, x)

    # Physical dimensions
    z_um, y_um, x_um = img.physical_pixel_sizes.Z, img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X
    #print(z_um, y_um, x_um)    

    # Channel info 
    channel_info = [str(n) for n in img.channel_names]
    #print(channel_info)
    
    # # Extract metadata - adjust these according to available attributes/methods from BioImage
    image_size_kb = os.path.getsize(file_path) / 1024  # Convert bytes to kilobytes


    if file_path.lower().endswith(".nd2"):
        # check if there are ROIs in the file
        import nd2
        f = nd2.ND2File(file_path)

        # Find ROIs
        
        center_X = [int(x/2) + int(roi.animParams[0].centerX) for roi in f.rois.values() if roi.animParams]
        center_Y = [int(y/2) + int(roi.animParams[0].centerY) for roi in f.rois.values() if roi.animParams]

        if len(center_X) > 0:
            print("centerX: ", center_X, " centerY: ", center_Y, "TODO: Find out if this is pixel units or microns")

    image_metadata = {
        'Image metadata': {
            'Size (kb)': image_size_kb,
            'Image dimensions': {
                'T': t,
                'C': c,
                'Z': z,
                'Y': y,
                'X': x,
            },
            'Physical dimensions': {
                'T_ms': None, # 
                'Z_um': z_um,
                'Y_um': y_um,
                'X_um': x_um,
            },
            'Channels': [{'Name': f'Please fill in e.g. {name}'} for name in channel_info],
        }

    
    }


    # Look for ROI definitions in ND2 files
    if file_path.lower().endswith(".nd2"):
            f = nd2.ND2File(file_path)

            # Find ROIs
            centers = [(int(x/2) + int(roi.animParams[0].centerX), 
                        int(y/2) + int(roi.animParams[0].centerY)) 
                    for roi in f.rois.values() if roi.animParams]

            # Add ROI information to metadata
            if centers:
                image_metadata['Image metadata']['ROIs'] = [{'Center X': cx, 'Center Y': cy} for cx, cy in centers]
                image_metadata['Image metadata']["Comment"] = "ROI extraction are under development!!!, check units (pixels or microns)."

    return image_metadata


def get_rois(file_path):
    
    # Look for ROI definitions in ND2 files
    if file_path.lower().endswith(".nd2"):
        # check if there are ROIs in the file
        import nd2
        f = nd2.ND2File(file_path)

        f.rois
    





def process_folder(folder_path, extension):
    """
    Process all files in folder with the specified extension, and save the processed data.
    """
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                print_roi(file_path)
                # processed_data = process_file(file_path)
                # # print(processed_data)
                # out_file_path = os.path.splitext(file_path)[0] + "_metadata.yaml"
                # with open(out_file_path, 'w') as file:
                #     yaml.dump(processed_data, file)
                



                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed.")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder.")
    #parser.add_argument("-o", "--output_path", type=str, default=None, help="Path to save processed data.")
    args = parser.parse_args()
      

    
    if args.folder_path is None:
        args.folder_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input"
        # TODO after testing raise ValueError("Please provide a folder path.")
    
    if args.extension is None:
        args.extension = ".nd2"
        # TODO after testing raise ValueError("Please provide a file extension.")
    
    # if args.output_path is None:
    #     args.output_path = args.folder_path

    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)

    process_folder(args.folder_path, args.extension)
