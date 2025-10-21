"""
Quick test of drift_correction.py with phase_cross_correlation_v2
"""
import sys
import os
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import numpy as np
import bioimage_pipeline_utils as rp
from pathlib import Path

# Create a test image and save it
print("Creating test image...")
T, C, Z, Y, X = 10, 2, 1, 256, 256
img_data = np.zeros((T, C, Z, Y, X), dtype=np.uint16)

# Add a pattern that drifts
for t in range(T):
    for c in range(C):
        shift_x = t * 2
        shift_y = t * 3
        y_start = 80 + shift_y
        y_end = 160 + shift_y
        x_start = 80 + shift_x
        x_end = 160 + shift_x
        img_data[t, c, 0, y_start:y_end, x_start:x_end] = 1000

# Save test image
test_dir = Path(r"e:\Oyvind\OF_git\run_pipeline\test_data")
test_dir.mkdir(exist_ok=True)

input_file = test_dir / "test_drift.tif"
output_dir = test_dir / "output"
output_dir.mkdir(exist_ok=True)

print(f"Saving test image to: {input_file}")
img = rp.BioImage(img_data)
img.save(str(input_file))

print(f"\nTest file created: {input_file}")
print(f"Output directory: {output_dir}")
print("\nYou can now run:")
print(f'python standard_code/python/drift_correction.py --input-search-pattern "{input_file}" --output-folder "{output_dir}" --method phase_cross_correlation_v2 --no-parallel')
