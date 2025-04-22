from bioio import BioImage
import bioio_bioformats
import argparse
import os
import numpy as np

def process_image(input_file_path: str, output_file_path:str = None, fps: int =24):
    # Set output file path
    if output_file_path is None:
        output_file_path = os.path.splitext(input_file_path)[0] + ".mp4"
    else:
        if not output_file_path.endswith(".mp4"):
            raise ValueError("Output file must have .mp4 extension")

    # Load image data
    img = BioImage(input_file_path, reader=bioio_bioformats.Reader)
    
    # # Get the image dimensions
    # dims = img.dims  # Dimensions object
    # print(f"Image dimensions: {dims}")  # Debugging output
    # dim_order = dims.order  # e.g. "TCZYX"

    # Convert to numpy array
    img_array: np.ndarray = img.data  # This is expected to be a numpy array
    img_array = np.transpose(img_array, (0, 2, 3, 1))  # Default to (T, Z, Y, X, C)

    # Perform maximum projection along the Z dimension (axis=1)
    if 'Z' in dim_order:
        img_array = np.max(img_array, axis=1)  # Maximum projection
        img_array = np.squeeze(img_array)  # Remove singleton dimensions if necessary

    # Handle single-channel (greyscale) and multi-channel (RGB)
    channels = img_array.shape[-1]  # Number of channels (C)
    
    if channels == 1:
        # Greyscale image
        print("Processing as greyscale image.")
        img_array = np.squeeze(img_array)  # Remove channel dimension
        dim_order = "TYX"  # Set dimension order for greyscale
    elif channels == 3:
        # RGB image
        print("Processing as RGB image.")
        dim_order = "TYX"  # Set dimension order for RGB
    elif channels > 3:
        # More than 3 channels
        print("Warning: More than 3 channels detected. Processing only the first three channels.")
        img_array = img_array[..., :3]  # Only take the first three channels
        dim_order = "TYX"  # Set dimension order for RGB
    
    # Save the data using TimeseriesWriter
    TimeseriesWriter.save(img_array, output_file_path, dim_order=dim_order, fps=fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to be processed")
    parser.add_argument("-o", "--output_file", type=str, help="Path for the output MP4 file")

    args = parser.parse_args()

    process_image(args.input_file, args.output_file, args.fps)

if __name__ == "__main__":
    main()
