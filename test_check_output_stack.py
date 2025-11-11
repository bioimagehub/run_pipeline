"""Quick test to check if saved TIFF files are actually 3D stacks."""

import tifffile
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python test_check_output_stack.py <path_to_tiff>")
    sys.exit(1)

tiff_path = sys.argv[1]

if not Path(tiff_path).exists():
    print(f"File not found: {tiff_path}")
    sys.exit(1)

# Load and check shape
with tifffile.TiffFile(tiff_path) as tif:
    data = tif.asarray()
    print(f"File: {tiff_path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Number of pages (frames): {len(tif.pages)}")
    print(f"ImageJ metadata: {tif.imagej_metadata}")
    
    if len(data.shape) == 2:
        print("\n⚠️ WARNING: This is a 2D image, not a 3D stack!")
    elif len(data.shape) == 3:
        print(f"\n✓ This is a 3D stack with {data.shape[0]} Z-slices")
    else:
        print(f"\n? Unexpected dimensionality: {len(data.shape)}D")
