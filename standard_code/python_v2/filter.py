

import argparse

import run_pipeline_utils as rpu
import numpy as np
from scipy import ndimage

# Only the filter logic is specific here
def local_filter(plane, method="gaussian", size=(3, 3, 3), sigma_xy=1.0):
    if method == "gaussian":
        return ndimage.gaussian_filter(plane, sigma=sigma_xy)
    elif method == "mean":
        return ndimage.uniform_filter(plane, size=size[:2], mode="reflect")
    elif method == "median":
        return ndimage.median_filter(plane, size=size[:2], mode="reflect")
    elif method == "min":
        return ndimage.minimum_filter(plane, size=size[:2], mode="reflect")
    elif method == "max":
        return ndimage.maximum_filter(plane, size=size[:2], mode="reflect")
    else:
        raise NotImplementedError(f"Filter method {method} not implemented.")

def main():
    file_list = [r"E:\\Oyvind\\BIP-hub-scratch\\train_macropinosome_model\\input_drift_corrected\\Input_crest__KiaWee__input__250225_RPE-mNG-Phafin2_BSD_10ul_001_drift_1.ome.tif"]
    output_dir = r"E:\\Oyvind\\BIP-hub-scratch\\train_macropinosome_model\\filtered_output"
    size = (3, 3, 3)  # Example size for the filter
    sigma_xy = 2  # Example sigma for the xy plane
    method = "gaussian"  # Change to 'mean', 'median', 'min', 'max' to test other filters

    rpu.process_files(
        file_list,
        lambda plane: local_filter(plane, method=method, size=size, sigma_xy=sigma_xy),
        chunk_type="YX",
        output_dir=output_dir
    )
if __name__ == "__main__":
    main()