#!/usr/bin/env python3
"""
Test script to verify that the cumulative shift accumulation fix works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))

import numpy as np
import bioimage_pipeline_utils as rp

def test_cumulative_fix():
    """Test that cumulative shift accumulation works for 'previous' reference mode."""
    
    print("=== Testing Cumulative Shift Fix ===")
    
    # Test with actual image files if available
    input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif"
    output_first = "E:/Oyvind/BIP-hub-test-data/drift/output_phase/1_Meng_with_known_drift_corrected_first.tif"
    output_previous = "E:/Oyvind/BIP-hub-test-data/drift/output_phase/1_Meng_with_known_drift_corrected_previous.tif"
    
    try:
        # Import the drift correction score function directly
        sys.path.append(os.path.join(os.path.dirname(__file__), 'standard_code', 'python', 'drift_correction'))
        from phase_cross_correlation import drift_correction_score
        
        # Test scores
        if os.path.exists(output_first) and os.path.exists(output_previous):
            score_first = drift_correction_score(output_first, channel=0, reference='first')
            score_previous = drift_correction_score(output_previous, channel=0, reference='first')
            
            print(f"First reference mode drift correction score: {score_first:.4f}")
            print(f"Previous reference mode drift correction score: {score_previous:.4f}")
            
            if abs(score_first - score_previous) < 0.01:
                print("✅ SUCCESS: Both reference modes achieve similar performance!")
                print("✅ Cumulative shift accumulation is working correctly!")
                return True
            else:
                print(f"❌ DIFFERENCE: {abs(score_first - score_previous):.4f} - further investigation needed")
                return False
        else:
            print(f"❌ Output files not found - please run the main script first")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_cumulative_fix()