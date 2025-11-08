"""Quick test of show_image blocking behavior."""

import bioimage_pipeline_utils as rp
import numpy as np

# Load a test image
img_path = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv_global\cellprofiler_input\SP20250625__3SA__R2__SP20250625_PC_R2_3SA_001_bleach_corrected.tif"

print("Loading image...")
img = rp.load_tczyx_image(img_path)
print(f"Image shape: {img.shape}")

print("Showing image - window should appear and script will wait...")
rp.show_image(img, title="Test Image - Close this window to continue")

print("Window was closed! Script continuing...")
print("Done!")
