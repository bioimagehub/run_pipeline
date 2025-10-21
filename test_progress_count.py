"""
Test progress bar counts are correct - Sequential mode for easier verification.

MIT License  
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import logging
import argparse
from drift_correction import process_files

# Set up logging to INFO to see progress details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("Testing Progress Bar Counts - SEQUENTIAL MODE")
print("=" * 60)
print("\nThis will process 1 file sequentially to verify counts")
print("Expected: T (detection) + T*C (application)")
print("For 181 timepoints, 2 channels: 181 + 181*2 = 543 total\n")

args = argparse.Namespace(
    input_search_pattern=r"Z:\Schink\Meng\driftcor\input\1.nd2",  # Just process one file
    output_folder=r"Z:\Schink\Meng\driftcor\test_count",
    output_suffix="_count_test",
    method="phase_cross_correlation",
    reference_channel=0,
    reference="first",
    no_save_tmats=True,
    no_gpu=False,
    no_parallel=True,  # Sequential for easier verification
    crop_fraction=0.8,
    upsample_factor=10,
    max_shift=50.0
)

print("Starting processing...\n")
process_files(args)

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("Check that the progress bar showed the correct total (543 for this file)")
