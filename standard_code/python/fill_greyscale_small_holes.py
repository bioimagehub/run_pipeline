from __future__ import annotations

import logging

from scipy import ndimage as ndi

# Local imports
import bioimage_pipeline_utils as rp

from time import time


def process_file(input_path: str, output_path: str, size: int = 5, mode: str = 'reflect'):
    image = rp.load_tczyx_image(input_path)
    data = image.data

    closed = ndi.grey_closing(data, size=(1, 1, 1, size, size), mode=mode)
    delta = closed - data
    rp.save_tczyx_image(delta, output_path)   




def main():
    # path = r"E:\Oyvind\BIP-hub-test-data\fill_holes\input\f1.tif"
    path = r"c:\Users\oodegard\Documents\test_data\fill_holes\250225_RPE-mNG-Phafin2_BSD_10ul_001_1.ome.tif"

    output_path = path.replace(".ome.tif", "_filled.tif")
    
    start_time = time() 
    process_file(path, output_path, size=10, mode='reflect')
    print(f"Processing time: {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()


 