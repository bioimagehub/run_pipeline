import numpy as np
import pandas as pd
import trackpy as tp
from scipy.ndimage import center_of_mass
from bioio.writers import OmeTiffWriter
import run_pipeline_helper_functions as rp


def track_labels_with_trackpy(indexed_masks, channel_zero_base=3, output_mask_path=None):
    """
    Track labels across frames using trackpy and return a new indexed binary mask
    with consistent IDs across time for one channel.

    Args:
        indexed_masks: 5D numpy array of indexed masks (T, C, Z, Y, X).
        channel_zero_base: The channel to track (zero-based indexing).
        output_mask_path: Optional path to save the new indexed mask.

    Returns:
        tuple: (DataFrame with tracked label information, new indexed mask with linked IDs)
    """
    T, C, Z, Y, X = indexed_masks.shape
    assert 0 <= channel_zero_base < C, "channel_zero_base out of bounds"

    mask_channel = indexed_masks[:, channel_zero_base]
    records = []

    # Step 1: Extract centroids from the tracking channel
    for t in range(T):
        for z in range(Z):
            slice_mask = mask_channel[t, z]
            labels = np.unique(slice_mask)
            labels = labels[labels != 0]  # skip background
            for label in labels:
                coords = center_of_mass(slice_mask == label)
                records.append({
                    'frame': t,
                    'z': z,
                    'y': coords[0],
                    'x': coords[1],
                    'label': label,
                    't_local': t * Z + z  # flattened time dimension
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No labels found in the selected channel for tracking.")

    # Step 2: Track with trackpy
    df_tracked = tp.link_df(df, search_range=10, pos_columns=['x', 'y'], t_column='t_local')

    # Step 3: Create a new full-size mask and fill only the tracked channel
    new_indexed_masks = np.copy(indexed_masks)  # preserve other channels
    new_indexed_masks[:, channel_zero_base] = 0  # clear tracked channel

    for row in df_tracked.itertuples():
        t, z = row.frame, row.z
        original_label = row.label
        slice_mask = mask_channel[t, z]
        new_indexed_masks[t, channel_zero_base, z][slice_mask == original_label] = row.particle + 1  # offset to avoid 0

    if output_mask_path:
        OmeTiffWriter.save(new_indexed_masks, output_mask_path, dim_order="TCZYX") 


    return df_tracked, new_indexed_masks



# Example usage:
input_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\output_nuc_mask\LDC20250314_1321N1_BANF1-V5-mEos4b_3SA_KD002.tif"
output_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\output_nuc_mask\LDC20250314_1321N1_BANF1-V5-mEos4b_3SA_KD002_linked.tif"

img = rp.load_bioio(input_path)

binary_masks = img.data

linked_results, indexed_mask = track_labels_with_trackpy(binary_masks, channel_zero_base=3,output_mask_path=output_path)
print(linked_results)
