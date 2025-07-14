import argparse
import os
from typing import List, Optional
import numpy as np
import pandas as pd
import napari
from bioio_ome_tiff.writers import OmeTiffWriter

#from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import run_pipeline_helper_functions as rp


def process_file(file: str, mask_folder: str, mask_suffix: str) -> None:
    print(f"Processing file: {file}")
    img = rp.load_bioio(file) #  T, C, Z, Y, X 
    
    if img is None:
        print(f"Failed to load image: {file}")
        return
    physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

    img_data: np.ndarray = img.data 


    # Construct mask path
    base_name = os.path.splitext(os.path.basename(file))[0]
    mask_path = os.path.join(mask_folder, base_name + mask_suffix)

    if os.path.exists(mask_path):
        mask = rp.load_bioio(mask_path)
        print(f"Loaded mask from: {mask_path}")
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            mask_data = np.zeros_like(img_data, dtype=np.uint8)
        else:
            mask_data = mask.data
    else:
        mask_data = np.zeros_like(img_data, dtype=np.uint8)  # Create an empty mask if it doesn't exist

    # Ensure mask has same number of channels as image
    if img_data.shape[1] > 1 and (mask_data.ndim == 4 or mask_data.shape[1] == 1):
        # Expand the mask along channel axis
        mask_data = np.repeat(mask_data, img_data.shape[1], axis=1)


        # Start Napari viewer
        viewer = napari.Viewer()
        viewer.add_image(img_data, name="Image", channel_axis=1 if img._dims == 5 else None)
        mask_layer = viewer.add_labels(mask_data, name="Mask")
    else:
        print("Mask does not match image channels, using single channel mask.")
        viewer = napari.Viewer()
        viewer.add_image(img_data, name="Image", channel_axis=1 if img._dims == 5 else None)
        mask_layer = viewer.add_labels(mask_data, name="Mask", blending='additive')

    # Add a button to save edited mask
    def save_mask(*args):
        updated_mask = np.array(mask_layer.data).astype(np.uint8)

        # Collapse updated mask if original was single-channel
        if mask_data.ndim == 4:
            # Original mask had shape: (T, Z, Y, X)
            # Collapse across channels by OR-ing over channel dimension (axis=1)
            collapsed = np.any(updated_mask, axis=1).astype(np.uint8)
            OmeTiffWriter.save(collapsed, mask_path, dim_order="TZYX", physical_pixel_sizes=physical_pixel_sizes)

        elif mask_data.shape[1] == 1:
            # Original mask had shape: (T, 1, Z, Y, X)
            collapsed = np.any(updated_mask, axis=1, keepdims=True).astype(np.uint8)
            OmeTiffWriter.save(collapsed, mask_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)

        else:
            # Save updated multi-channel mask as-is
            OmeTiffWriter.save(updated_mask, mask_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)

        print(f"Saved edited mask to: {mask_path}")
    
    # viewer.window.add_plugin_dock_widget('napari', 'Plugin')  # Just to help UI layout
    viewer.bind_key('s', save_mask)  # Press 's' to save the mask
    napari.run()


def process_folder(
    image_folder: str,
    image_suffix: str,
    mask_folder: str,
    mask_suffix: str,
) -> None:
    
    files_to_process = rp.get_files_to_process(image_folder, image_suffix, False)
    for file in files_to_process:
        process_file(
            file,
            mask_folder=mask_folder,
            mask_suffix=mask_suffix
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder",type=str, required=True,
                        help="List of input folders containing images")
    parser.add_argument("--input-suffix", type=str, required=True,
                        help="Suffixes of input images; CSV-style list per folder")

    parser.add_argument("--mask-folder", type=str, required=True,
                        help="Path to folder with binary mask images")
    parser.add_argument("--mask-suffix", type=str, required=True,
                        help="Suffix of mask files (e.g. _mask.tif)")

    args = parser.parse_args()

    process_folder(
        image_folder=args.input_folder,
        image_suffix=args.input_suffix,
        mask_folder=args.mask_folder,
        mask_suffix=args.mask_suffix
    )
