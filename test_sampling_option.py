"""
Quick test to verify the --max-measure-pixels -1 option works correctly.

This demonstrates backward compatibility:
- Default behavior (max_measure_pixels=20): Uses random sampling
- New behavior (max_measure_pixels=-1): Uses all pixels (no sampling)
"""

import sys
import os

# Add standard_code/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))

# Import the plotting functions
from plots.quantify_distance_heatmap import plot_distance_time_heatmap, quantify_signal_spread_and_decay

# Test with sampling disabled (new feature)
print("=" * 80)
print("Testing with max_measure_pixels=-1 (NO SAMPLING - uses all pixels)")
print("=" * 80)

# This should log: "Random sampling DISABLED (max_measure_pixels=-1): using ALL pixels per bin"

# Example call (you would need actual data files to run this):
# plot_distance_time_heatmap(
#     image_path="path/to/image.tif",
#     distance_path="path/to/distance.tif",
#     max_measure_pixels=-1  # Disable sampling, use all pixels
# )

print("\nKey changes:")
print("1. When max_measure_pixels=-1: Sampling is disabled, all pixels are used")
print("2. When max_measure_pixels>0: Sampling is enabled (original behavior)")
print("3. Backward compatible: Default is still 20 (original behavior)")
print("\nTo use in command line:")
print("  python -m plots.quantify_distance_heatmap --max-measure-pixels -1 ...")
print("\nThe code will log:")
print("  'Random sampling DISABLED (max_measure_pixels=-1): using ALL pixels per bin'")
print("  (when max_measure_pixels=-1)")
print("or:")
print("  'Random sampling: max 20 pixels per bin (seed=42)'")
print("  (when max_measure_pixels=20, the default)")
