"""
Quick test to see progress bars working on real data.

Run directly with Python to see the progress bars update properly.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import logging
import argparse
from drift_correction import process_files

# Set up logging to WARNING level to avoid interference with progress bars
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("Testing Progress Bar Updates")
print("=" * 60)
print("\nThis will process 3 files in PARALLEL")
print("You should see 3 independent progress bars updating!\n")

args = argparse.Namespace(
    input_search_pattern=r"Z:\Schink\Meng\driftcor\input\*.nd2",
    output_folder=r"Z:\Schink\Meng\driftcor\test_progress",
    output_suffix="_test",
    method="phase_cross_correlation",
    reference_channel=0,
    reference="first",
    no_save_tmats=True,  # Skip saving tmats for speed
    no_gpu=False,
    no_parallel=False,  # PARALLEL MODE!
    crop_fraction=0.8,
    upsample_factor=10,
    max_shift=50.0
)

print("Starting processing...\n")
process_files(args)

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("Check that you saw 3 progress bars updating independently!")
