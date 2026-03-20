from __future__ import annotations
import os
import argparse
from matplotlib import image
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from contextlib import contextmanager
import skimage
from tqdm import tqdm

# Local imports
import bioimage_pipeline_utils as rp


from time import time


import numpy as np


from scipy import ndimage as ndi





def main():
    path = r"E:\Oyvind\BIP-hub-test-data\fill_holes\input\f1.tif"
    img = rp.load_tczyx_image(path).data[0,0,0,:,:]

    print(img.shape)

    # start = time()
    # out = skimage.morphology.remove_small_holes(img, area_threshold=100, connectivity=1) 
    # print(f"Time taken: {time() - start:.2f} seconds")
    # print(out)


    start = time()
    out = ndi.grey_closing(img, size=100)
    print(f"Time taken: {time() - start:.2f} seconds")
    print(out)


if __name__ == "__main__":
    main()


 