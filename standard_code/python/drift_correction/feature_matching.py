from typing import Tuple, Optional
import numpy as np
from skimage.feature import ORB, match_descriptors, corner_harris, corner_peaks
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.filters import gaussian, threshold_triangle
from skimage.exposure import equalize_hist, rescale_intensity

import os
import time

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    HAS_CUPY = True
except ImportError:
    cp = None
    cnd = None
    HAS_CUPY = False



# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import bioimage_pipeline_utils as rp

from drift_correct_utils import drift_correction_score, apply_shifts_to_tczyx_stack #, gaussian_blur_2d_gpu # if needed?

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show WARNING and above (ERROR, CRITICAL)


def feature_matching_drift_correction(image_stack: np.ndarray,
                                       reference_frame: str = 'first',
                                       channel: int = 0,
                                       max_shift_per_frame: float = 50.0,
                                       do_3d_correction: bool = False # Not implemented yet
                                       ) -> np.ndarray:
    """
    Detect drift in time-lapse images and compute correction shifts using feature matching.
    
    Args:
        image_stack (np.ndarray): Input 5D image stack with shape (T, C, Z, Y, X)
            - Must be properly formatted TCZYX array
            - Supports both 2D (Z=1) and 3D (Z>1) image series
        reference_frame (str): Reference selection strategy (default: 'first')
            - 'first': Align all frames to first timepoint
            - 'previous': Frame-to-frame alignment
        channel (int): Channel index for drift detection (default: 0)
        max_shift_per_frame (float): Max expected shift in pixels (default: 50.0)
        do_3d_correction (bool): Whether to perform 3D correction (default: False)
            - Currently only 2D correction is implemented and 3D is a placeholder for future extension
    
    Returns:
        np.ndarray: Correction shifts with shape (T, 3)
            - Format: [dz, dy, dx] per timepoint in ZYX order
    """
    
    if do_3d_correction:
        raise NotImplementedError("3D correction is not implemented yet.")
    
    if image_stack.ndim != 5:
        raise ValueError("Input image stack must be 5D (TCZYX)")
    
    # max project in Z dimension
    image_stack = np.max(image_stack, axis=2)

    T, _, _, _ = image_stack.shape
    shifts = np.zeros((T, 3), dtype=np.float32)

    if reference_frame == 'first':
        reference_volume = image_stack[0, channel]  # Use first frame for feature matching
    elif reference_frame == 'previous':
        reference_volume = image_stack[0, channel]  # Initialize for first frame
    else:
        raise ValueError("Invalid reference_frame. Use 'first' or 'previous'.")

    for t in range(T):
        if reference_frame == 'previous' and t != 0:
            current_volume = image_stack[t, channel]  # Current frame for matching
        else:
            current_volume = image_stack[t, channel]  # Current frame for reference





        # Initialize ORB for feature detection
        orb = ORB(n_keypoints=200)
        orb.detect_and_extract(reference_volume)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(current_volume)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        # Match features
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
        
        if matches.size > 0:
            src = keypoints1[matches[:, 0]]
            dst = keypoints2[matches[:, 1]]

            # Estimate transformation using RANSAC
            model, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                    residual_threshold=2, max_trials=100)

            # Calculate shifts from the transformation
            dz, dy = model.translation
            dx = 0.0  # Assuming no depth shift for 2D images

            # Validate shift against max shift limit
            shift_vector = np.array([0, dy, dx])
            if np.linalg.norm(shift_vector) > max_shift_per_frame:
                shift_vector = shift_vector / np.linalg.norm(shift_vector) * max_shift_per_frame

            if reference_frame == 'previous' and t != 0:
                shifts[t] = shifts[t-1] + shift_vector
            else:
                shifts[t] = shift_vector
        else:
            # No matches found, assign zero shift
            shifts[t] = [0.0, 0.0, 0.0]

        # Update reference volume for the next iteration if using reference 'previous'
        if reference_frame == 'previous':
            reference_volume = current_volume

    return shifts


def test_function():

    # Test 1: First frame as reference
    if(True):
        input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif"
        output_base = "E:/Oyvind/BIP-hub-test-data/drift/output_feature_matching/"
        # make dir
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        print("\n=== Example: Drift Correction with Feature Matching using first frame as reference ===")
        reference_frame = "first"  # or "first", "mean", etc.
        output_file = os.path.join(output_base, os.path.splitext(os.path.basename(input_file))[0] + f"_corrected_{reference_frame}.tif")

        # Load image and apply drift correction
        image_stack = rp.load_tczyx_image(input_file).data


        detected_shifts = feature_matching_drift_correction(
            image_stack, 
            reference_frame=reference_frame,  # Use previous frame as reference for better results
            channel=0, 
            max_shift_per_frame=50.0
        )

        # Apply correction shifts and save result
        corrected_image = apply_shifts_to_tczyx_stack(image_stack, detected_shifts, mode='constant')
        rp.save_tczyx_image(corrected_image, output_file)
        
        print(f"Drift correction completed. Saved to: {output_file}")
        
        # Validate the drift correction quality
        score_input = drift_correction_score(input_file, channel=0, reference='first', central_crop=0.8, z_project='max')
        print(f"Input image drift correction score: {score_input:.4f}")
        
        score = drift_correction_score(output_file, channel=0, reference='first', central_crop=0.8, z_project='max')
        print(f"Output image drift correction score: {score:.4f}")
        

    ##########################################
    # Test 3 Run with live cell data without known shifts
    if(False):
        print("\n=== Example: Testing with Real Data ===")
        input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng.nd2"
        output_base = "E:/Oyvind/BIP-hub-test-data/drift/output_feature_matching/"

        reference_frame = "previous"  # or "first", "mean", etc.
        output_file = os.path.join(output_base, os.path.splitext(os.path.basename(input_file))[0] + f"_corrected_{reference_frame}.tif")

        # Load image and apply drift correction
        image_stack = rp.load_tczyx_image(input_file).data


        detected_shifts = feature_matching_drift_correction(
            image_stack, 
            reference_frame=reference_frame,  # Use previous frame as reference for better results
            channel=0, 
            max_shift_per_frame=50.0
        )
        # Apply correction shifts and save result
        corrected_image = apply_shifts_to_tczyx_stack(image_stack, detected_shifts, mode='constant')
        rp.save_tczyx_image(corrected_image, output_file)

        print(f"Drift correction completed. Saved to: {output_file}")

        # Validate the drift correction quality
        score_input = drift_correction_score(input_file, channel=0, reference='previous', central_crop=0.8, z_project='max')
        print(f"Input image drift correction score: {score_input:.4f}")

        score = drift_correction_score(output_file, channel=0, reference='previous', central_crop=0.8, z_project='max')
        print(f"Output image drift correction score: {score:.4f}")

if __name__ == "__main__":
    test_function()