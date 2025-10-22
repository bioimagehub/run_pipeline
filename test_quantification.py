"""
Test script for the new signal spread and decay quantification function.

This script demonstrates how to use the quantify_signal_spread_and_decay function
to analyze photoconversion data that spreads spatially and decays temporally.

Usage:
    python test_quantification.py
"""

import sys
import os
from pathlib import Path

# Add standard_code/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))

from plots.plot_distance_heatmap import quantify_signal_spread_and_decay
import bioimage_pipeline_utils as rp

def test_quantification():
    """
    Test the quantification function with example data.
    
    You'll need to provide paths to:
    - An image file (TCZYX format)
    - A distance matrix file (matching dimensions)
    """
    
    # TODO: Update these paths to your actual data
    image_path = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\your_image.tif"
    distance_path = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\your_distance_matrix.tif"
    output_path = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\quantification_test.png"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        print("\nPlease update the paths in test_quantification.py to point to your actual data.")
        return
    
    if not os.path.exists(distance_path):
        print(f"ERROR: Distance file not found: {distance_path}")
        print("\nPlease update the paths in test_quantification.py to point to your actual data.")
        return
    
    print("Running quantification analysis...")
    print(f"Image: {Path(image_path).name}")
    print(f"Distance: {Path(distance_path).name}")
    print()
    
    # Run quantification
    results = quantify_signal_spread_and_decay(
        image_path=image_path,
        distance_path=distance_path,
        output_path=output_path,
        channel=0,
        colormap='viridis',
        normalize_to_t0=False,
        remove_first_n_bins=5,
        smooth_sigma=1.5,
        signal_percentile=80.0,
        force_show=True  # Display the plot
    )
    
    if results:
        print("\n=== QUANTIFICATION RESULTS ===")
        print(f"Temporal decay point: T={results['temporal_decay_point']}")
        print(f"Signal percentile used: {results['signal_percentile']}%")
        print(f"Smoothing sigma: {results['smooth_sigma']}")
        print(f"\nDistance spread statistics:")
        print(f"  Mean spread: {results['distance_spread'].mean():.2f}")
        print(f"  Max spread: {results['distance_spread'].max():.2f}")
        print(f"  Min spread: {results['distance_spread'].min():.2f}")
        print(f"\nResults saved to: {output_path}")
        print(f"Metrics saved to: {output_path.replace('.png', '_metrics.tsv')}")
    else:
        print("ERROR: Quantification failed!")


if __name__ == "__main__":
    print("="*60)
    print("Signal Spread and Decay Quantification Test")
    print("="*60)
    print()
    test_quantification()
