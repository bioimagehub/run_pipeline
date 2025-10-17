"""
Debug script to test time-direction hole filling on a mask file.
Step 1: Convert to binary (all labels → 1)
Step 2: Fill temporal gaps based on spatial overlap
Step 3: Save binary mask ready for tracking
"""
import numpy as np
from standard_code.python import bioimage_pipeline_utils as rp
import os

# Input file path
input_file = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\output_masks\SP20250625_PC_R3_LKO8_004_segmentation.tif"
max_time_gap = 2  # Maximum time gap to fill (e.g., 2 means fill if t-1 and t+1 have label)

# Load the image using the standard pipeline helper
print(f"Loading image from: {input_file}")
img = rp.load_tczyx_image(input_file)
output_data = img.data.copy()  # Make a copy to preserve original

print(f"Image shape: {output_data.shape} (TCZYX)")
print(f"Data type: {output_data.dtype}")
print(f"Unique labels before filling: {len(np.unique(output_data))} ({np.unique(output_data)})")

# Convert to binary: all non-zero labels become 1
print("\n=== Converting to binary ===")
output_data = (output_data > 0).astype(np.uint8)
print(f"Unique values after binarization: {np.unique(output_data)}")

# Analyze temporal label distribution
print("\n=== Object presence over time (binary) ===")
for t in range(output_data.shape[0]):
    n_pixels = np.sum(output_data[t] > 0)
    print(f"T={t}: {n_pixels} pixels")

# Count how many background pixels exist between matching labels
fill_count = 0

# --- Fill "holes" in time direction using spatial overlap ---
# Fill gaps up to max_time_gap: if a pixel has a label at t-gap and t+gap, fill all gaps in between
print(f"\n=== Filling temporal holes (max gap = {max_time_gap}) ===")

for c in range(output_data.shape[1]):
    for z in range(output_data.shape[2]):
        if output_data.shape[0] < (2 * max_time_gap + 1):  # Need enough time points
            print(f"Not enough time points (need at least {2 * max_time_gap + 1})")
            continue
        
        print(f"\nProcessing C={c}, Z={z}, Time-points={output_data.shape[0]}")
        
        # Iterate through each time point
        for t in range(output_data.shape[0]):
            t_current = output_data[t, c, z, :, :]
            
            # Only process background pixels
            is_background = (t_current == 0)
            if not np.any(is_background):
                continue
            
            # Check for labels within max_time_gap frames before and after
            for gap in range(1, max_time_gap + 1):
                t_before_idx = t - gap
                t_after_idx = t + gap
                
                # Check if indices are valid
                if t_before_idx < 0 or t_after_idx >= output_data.shape[0]:
                    continue
                
                t_before = output_data[t_before_idx, c, z, :, :]
                t_after = output_data[t_after_idx, c, z, :, :]
                
                # Find pixels that:
                # 1. Are background at current time
                # 2. Have non-zero values at both t-gap and t+gap (spatial overlap)
                has_before = (t_before > 0)
                has_after = (t_after > 0)
                
                fill_mask = is_background & has_before & has_after
                
                # Count and fill
                n_filled = np.sum(fill_mask)
                if n_filled > 0:
                    # Fill with the label from t-gap
                    t_current[fill_mask] = t_before[fill_mask]
                    output_data[t, c, z, :, :] = t_current
                    fill_count += n_filled
                    print(f"  T={t} (gap={gap}): Filled {n_filled} pixels")
                    
                    # Update is_background for next gap iteration
                    is_background = (t_current == 0)
                    if not np.any(is_background):
                        break  # No more background pixels to fill

print(f"\n=== Results ===")
print(f"Total pixels filled: {fill_count}")
print(f"Unique values after filling: {np.unique(output_data)} (should be [0, 1])")

# Analyze temporal distribution after filling
print("\n=== Object presence over time (after filling) ===")
for t in range(output_data.shape[0]):
    n_pixels = np.sum(output_data[t] > 0)
    print(f"T={t}: {n_pixels} pixels")

# Save the output
output_dir = os.path.dirname(input_file)
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(output_dir, base_name + "_binary_filled_time.tif")

print(f"\nSaving filled image to: {output_file}")
rp.save_tczyx_image(output_data, output_file, dim_order="TCZYX")
print("Done!")

# Print final statistics
print("\n=== Final Statistics ===")
print(f"Original file: {input_file}")
print(f"Output file: {output_file}")
print(f"Shape: {output_data.shape} (TCZYX)")
print(f"Pixels filled: {fill_count}")
print(f"Data type: {output_data.dtype}")
print(f"Values: {np.unique(output_data)} (binary: 0=background, 1=object)")
print("\nNOTE: This binary mask is ready for tracking with track_indexed_mask.py")
