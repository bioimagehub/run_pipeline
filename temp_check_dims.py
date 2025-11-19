import tifffile
import sys

tiff_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/temp/ij_bridge_output/ing_3.tif'

print(f"Checking: {tiff_path}")
img = tifffile.imread(tiff_path)
print(f"TIFF shape: {img.shape}")
print(f"TIFF dtype: {img.dtype}")

with tifffile.TiffFile(tiff_path) as tif:
    print(f"Number of pages: {len(tif.pages)}")
    if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
        print(f"ImageJ metadata: {tif.imagej_metadata}")
    else:
        print("No ImageJ metadata found")
    
    # Check first page
    page = tif.pages[0]
    print(f"First page shape: {page.shape}")
    if hasattr(page, 'tags'):
        if 'ImageDescription' in page.tags:
            print(f"ImageDescription: {page.tags['ImageDescription'].value}")
