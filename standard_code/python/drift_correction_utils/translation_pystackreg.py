"""
PyStackReg Drift Correction - Translation Only

Wrapper for pystackreg (ImageJ TurboReg) using TRANSLATION mode only.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Tuple, Literal, Optional
import numpy as np
from bioio import BioImage
from pystackreg import StackReg
from tqdm import tqdm

# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    # Go up to standard_code/python directory to find bioimage_pipeline_utils
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)

def register_image_xy(
        img: BioImage,
        reference: Literal['first', 'previous', 'median'] = 'previous',
        channel: int = 0,
        show_progress: bool = True,
        no_gpu: bool = False
        ) -> Tuple[BioImage, np.ndarray]: 
    '''Register a TCZYX image using translation in XY dimensions only.
    
    Performs drift correction by computing transformations from a max-projected
    reference channel, then applying those transformations to all channels in
    the full 3D stack. Uses PyStackReg's TRANSLATION mode for rigid XY shifts.
    
    The Z dimension is max-projected for registration calculation only, but the
    full 3D volume is transformed and returned as a new BioImage object with
    original metadata preserved.
    
    Args:
        img: BioImage object containing TCZYX image data. Must have at least
            2 timepoints for registration to be meaningful.
        reference: Registration reference strategy:
            - 'first': Register all frames to the first timepoint
            - 'previous': Register each frame to its previous frame (default)
            - 'median': Register all frames to the temporal median image
        channel: Zero-indexed channel to use for computing transformations.
            Typically choose the brightest or most stable channel.
        show_progress: Whether to display progress bars (default: True).
    
    Returns:
        Tuple containing:
            - registered (np.ndarray): Drift-corrected 5D TCZYX numpy array
              with all channels transformed using the reference channel's
              transformation matrices
            - tmats (np.ndarray): Transformation matrices from pystackreg,
              shape (T, 3, 3) for TRANSLATION mode. Can be saved and reused.
    
    Raises:
        ValueError: If image has insufficient dimensions or invalid channel index
        
    Example:
        >>> img = rp.load_tczyx_image("timelapse.tif")
        >>> registered, tmats = register_image_xy(img, reference='first', channel=0)
        >>> rp.save_tczyx_image(registered, "corrected.tif")
    Note:
        For images with only 1 timepoint, returns original data unchanged
        with identity transformation matrix.
        In the future, a GPU-accelerated version of apply shifts may be implemented.
    '''
    logger.info(f"Starting XY drift correction with reference='{reference}', channel={channel}")
    
    # Extract reference channel as 4D TZYX array
    ref_channel_data = img.get_image_data("TZYX", C=channel)

    # Max projection over Z (axis=1) to reduce to 2D for registration
    # Shape: (T, Y, X) -> (T, Y, X)
    ref_channel_data = np.max(ref_channel_data, axis=1, keepdims=False)

    # Verify we have multiple timepoints
    if ref_channel_data.shape[0] <= 1:
        logger.warning("Image has only 1 timepoint, returning original BioImage")
        return img, np.array([np.eye(3)])
    
    # Initialize StackReg object for translation only
    sr = StackReg(StackReg.TRANSLATION)

    # Register the stack to compute transformation matrices
    logger.info(f"Computing transformations from max-projected stack with shape {ref_channel_data.shape} using reference '{reference}'")
    
    if show_progress:
        # Create progress bar for registration
        pbar = tqdm(total=ref_channel_data.shape[0], desc="Finding shifts", unit="frame")
        
        def progress_callback(current_iteration, end_iteration):
            pbar.n = current_iteration + 1  # +1 because pystackreg starts from 0
            pbar.refresh()
        
        tmats = sr.register_stack(ref_channel_data, reference=reference, progress_callback=progress_callback)
        pbar.close()
    else:
        tmats = sr.register_stack(ref_channel_data, reference=reference)

    # Release reference channel memory
    del ref_channel_data

    # Load entire image dataset (5D TCZYX)
    img_data = img.data

    # Apply the transformations to all channels and Z-slices
    logger.info(f"Applying transformations to full stack with shape {img_data.shape}")
    registered_data = np.zeros_like(img_data)
    
    # Setup progress tracking
    if show_progress:
        pbar = tqdm(total=img_data.shape[1] * img_data.shape[0], desc="Applying shifts", unit="frame")
    
    for c in range(img_data.shape[1]):  # Loop over channels (axis=1 in TCZYX)
        if show_progress:
            pbar.set_description(f"Applying shifts C={c}")
        else:
            logger.info(f"Transforming channel {c}/{img_data.shape[1]-1}")
        
        channel_data = img.get_image_data("TZYX", C=c)
        
        # Apply transformation to each Z-slice separately
        for z in range(channel_data.shape[1]):  # Loop over Z (axis=1 in TZYX)
            z_slice = channel_data[:, z, :, :]  # Extract TYX slice
            registered_data[:, c, z, :, :] = sr.transform_stack(z_slice, tmats=tmats)
            if show_progress and z == 0:  # Update once per frame (only on first Z-slice)
                pbar.update(z_slice.shape[0])
    
    if show_progress:
        pbar.close()

    logger.info("XY drift correction completed successfully")

    img_registered = BioImage(registered_data, physical_pixel_sizes=img.physical_pixel_sizes,
                              channel_names=img.channel_names, metadata=img.metadata)

    return img_registered, tmats



def test2_code():
    # Configure logging to show INFO messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15.tif"
    img = rp.load_tczyx_image(path)
    registered, tmats = register_image_xy(img, reference='previous', channel=0)

    np.save(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15_sStackReg_tmats.npy", tmats)
    
    #rp.save_tczyx_image(registered, r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15_sStackReg_corrected.tif")
    registered.save(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15_sStackReg_corrected.tif")


def test3_code():
    # Configure logging to show INFO messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
    img = rp.load_tczyx_image(path)
    registered, tmats = register_image_xy(img, reference='previous', channel=0)

    registered.save(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop_StackReg_corrected.tif")


if __name__ == "__main__":

    test3_code()