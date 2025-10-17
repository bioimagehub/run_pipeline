"""
Debug script to test time-direction hole filling on a mask file.
Fills gaps in time where t-1 and t+1 have the same label.
"""
import numpy as np
from standard_code.python import bioimage_pipeline_utils as rp
import os

# Input file path
input_file = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\output_masks\SP20250625_PC_R3_LKO8_004_segmentation_filled.tif"

# Load the image using the standard pipeline helper
print(f"Loading image from: {input_file}")
img = rp.load_tczyx_image(input_file)
output_data = img.data.copy()  # Make a copy to preserve original

print(f"Image shape: {output_data.shape} (TCZYX)")
print(f"Data type: {output_data.dtype}")
print(f"Unique labels before filling: {len(np.unique(output_data))} ({np.unique(output_data)[:10]}...)")

# Count how many background pixels exist between matching labels
fill_count = 0

# --- Fill "holes" in time direction ---
# Defined as if the pixel before and after in time are the same label, fill in between
for c in range(output_data.shape[1]):
    for z in range(output_data.shape[2]):
        if output_data.shape[0] < 3:  # Need at least 3 time points
            print("Not enough time points (need at least 3)")
            continue
        
        print(f"Processing C={c}, Z={z}, Time-points={output_data.shape[0]}")
        
        # Get slices: before, current, after
        t_before = output_data[:-2, c, z, :, :]   # t=0 to t=T-3
        t_current = output_data[1:-1, c, z, :, :]  # t=1 to t=T-2
        t_after = output_data[2:, c, z, :, :]    # t=2 to t=T-1
        
        # Find where before and after match (and are non-zero)
        match_mask = (t_before == t_after) & (t_before > 0)
        
        # Fill current with the matching label where current is background (0)
        fill_mask = match_mask & (t_current == 0)
        
        # Count how many pixels will be filled
        n_filled = np.sum(fill_mask)
        fill_count += n_filled
        print(f"  Filling {n_filled} pixels in this C/Z combination")
        
        t_current[fill_mask] = t_before[fill_mask]
        
        # Write back to output_data
        output_data[1:-1, c, z, :, :] = t_current

print(f"\nTotal pixels filled: {fill_count}")
print(f"Unique labels after filling: {len(np.unique(output_data))}")

# Save the output
output_dir = os.path.dirname(input_file)
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(output_dir, base_name + "_filled_time.tif")

print(f"\nSaving filled image to: {output_file}")
rp.save_tczyx_image(output_data, output_file, dim_order="TCZYX")
print("Done!")

# Print some statistics
print("\n=== Statistics ===")
print(f"Original file: {input_file}")
print(f"Output file: {output_file}")
print(f"Shape: {output_data.shape} (TCZYX)")
print(f"Pixels filled: {fill_count}")
print(f"Unique labels: {len(np.unique(output_data))}")
