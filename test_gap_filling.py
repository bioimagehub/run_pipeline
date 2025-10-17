"""Quick test to verify gap filling logic"""
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\cc_lab\Documents\git\run_pipeline\standard_code\python")
from fill_mask_gaps import interpolate_mask_morphology

# Create test masks
mask_before = np.zeros((10, 10), dtype=np.uint16)
mask_after = np.zeros((10, 10), dtype=np.uint16)

# Object 1 at position 3,3 (3x3 square)
mask_before[2:5, 2:5] = 1

# Object 2 at position 6,6 (3x3 square) - DIFFERENT ID!
mask_after[5:8, 5:8] = 2

print("Before mask (object ID=1):")
print(mask_before)
print("\nAfter mask (object ID=2):")
print(mask_after)

# Test interpolation with TWO different IDs
interpolated = interpolate_mask_morphology(mask_before, mask_after, n_steps=2, object_id_before=1, object_id_after=2)

print(f"\nGenerated {len(interpolated)} interpolated frames")
for i, frame in enumerate(interpolated):
    print(f"\nInterpolated frame {i+1} (should have object ID=1):")
    print(frame)
    print(f"  Pixels filled: {np.sum(frame == 1)}")
    print(f"  Max ID: {np.max(frame)}")
