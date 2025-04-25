import numpy as np

def get_mask_by_threshold(array, threshold=None):
    """
    Generate an indexed binary mask from a numpy array based on a threshold.

    Parameters:
        array (numpy.ndarray): Input numpy array.
        threshold (float, optional): Threshold value. If None, the mean of the array is used.

    Returns:
        numpy.ndarray: Indexed binary mask.
    """
    if threshold is None:
        threshold = np.mean(array)
    
    # Create binary mask
    binary_mask = array > threshold
    
    # Generate indexed mask
    indexed_mask = np.zeros_like(array, dtype=int)
    indexed_mask[binary_mask] = 1
    
    return indexed_mask