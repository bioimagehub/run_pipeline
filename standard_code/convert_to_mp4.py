from bioio import BioImage
import bioio_bioformats
import argparse
import os
import numpy as np
from bioio.writers.timeseries_writer import TimeseriesWriter

def process_image(input_file_path, output_file_path=None, fps=24):
    # Set output file path
    if output_file_path is None:
        output_file_path = os.path.splitext(input_file_path)[0] + ".mp4"
    else:
        if not output_file_path.endswith(".mp4"):
            raise ValueError("Output file must have .mp4 extension")

    # Load image data
    print(f"Loading image data from {input_file_path}")

    img = BioImage(input_file_path, reader=bioio_bioformats.Reader)
    
    # Get the image dimensions
    dims = img.dims  # Dimensions object
    
    # Convert to numpy array
    img_array = img.data  

    # Mass project or remove Z dimension if necessary
    print("Maximum projection of Z dimension")
    img_array = np.max(img_array, axis=2)  # Maximum projection of Z

    # loop over channels and

    for i in range(dims.C):
        ch_img = img_array[:, i, :]

        # convert to uint8 from min and max
        ch_img = (ch_img - np.min(ch_img)) / (np.max(ch_img) - np.min(ch_img)) * 255
        ch_img = ch_img.astype(np.uint8)

        ch_name_split = os.path.splitext(output_file_path) 
        ch_name = ch_name_split[0] + f"_ch{i+1}" + ch_name_split[1]

        # Save the data using TimeseriesWriter
        TimeseriesWriter.save(ch_img, ch_name, dim_order="TYX", fps=fps)
    print("Done saving channels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to be processed")
    parser.add_argument("-o", "--output_file", type=str, help="Path for the output MP4 file")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for output video")

    args = parser.parse_args()

    process_image(args.input_file, args.output_file, args.fps)

if __name__ == "__main__":
    main()
