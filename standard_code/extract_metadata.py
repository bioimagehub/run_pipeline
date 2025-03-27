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
       
    crash 
    # print(img.dims)

    # # Extract metadata - adjust these according to available attributes/methods from BioImage
    # image_size_kb = os.path.getsize(file_path) / 1024  # Convert bytes to kilobytes

    # image_metadata = {
    #     'Image metadata': {
    #         'Size (kb)': image_size_kb,
    #         'Image dimensions': {
    #             'T': data.shape[0] if data.ndim > 0 else None,
    #             'C': data.shape[1] if data.ndim > 1 else None,
    #             'Z': data.shape[2] if data.ndim > 2 else None,
    #             'Y': data.shape[3] if data.ndim > 3 else None,
    #             'X': data.shape[4] if data.ndim > 4 else None,
    #         },
    #         'Physical dimensions': {
    #             'T_ms': getattr(img, 'physical_size_t', None),
    #             'Z_um': getattr(img, 'physical_size_z', None),
    #             'Y_um': getattr(img, 'physical_size_y', None),
    #             'X_um': getattr(img, 'physical_size_x', None),
    #         },
    #         'Channels': [{'Name': f'Please fill in e.g. {name}'} for name in getattr(img, 'channel_names', ['DAPI', 'Aktin-GFP', 'Tubulin-RFP'])],
    #     }
    # }

    # # Writing metadata to YAML
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    # yaml_file_path = os.path.join(output_path, f"{file_name}_metadata.yaml")

    # os.makedirs(output_path, exist_ok=True)
    # with open(yaml_file_path, 'w') as yaml_file:
    #     yaml.dump(image_metadata, yaml_file, default_flow_style=False)

    # print(f"Metadata saved to: {yaml_file_path}")
    # # Here should be code to properly clean up if necessary, e.g., closing images, etc.
    # del img

    return -1




def process_folder(folder_path, extension, output_path):
    """
    Process all files in folder with the specified extension, and save the processed data.
    """
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                processed_data = process_file(file_path)
                print(processed_data)
                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed.")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder.")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Path to save processed data.")
    args = parser.parse_args()
      

    
    if args.folder_path is None:
        args.folder_path = r"C:\Users\oodegard\Desktop\test_img_annotation"
        # TODO after testing raise ValueError("Please provide a folder path.")
    
    if args.extension is None:
        args.extension = ".nd2"
        # TODO after testing raise ValueError("Please provide a file extension.")
    
    if args.output_path is None:
        args.output_path = args.folder_path

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_folder(args.folder_path, args.extension, args.output_path)
