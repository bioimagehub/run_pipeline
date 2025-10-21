"""Quick script to check image dimensions."""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
import bioimage_pipeline_utils as rp

path = r"Z:\Schink\Meng\driftcor\input\1.nd2"
img = rp.load_tczyx_image(path)
print(f"Image shape: {img.shape}")
print(f"T={img.dims.T}, C={img.dims.C}, Z={img.dims.Z}, Y={img.dims.Y}, X={img.dims.X}")
print(f"\nExpected progress steps:")
print(f"  Detection phase: T = {img.dims.T}")
print(f"  Application phase: T×C = {img.dims.T} × {img.dims.C} = {img.dims.T * img.dims.C}")
print(f"  TOTAL: {img.dims.T + img.dims.T * img.dims.C}")
