"""Quick test to check if imwrite creates proper OME-TIFF."""
import tempfile
from pathlib import Path
from bioio import BioImage
from tifffile import imwrite, TiffFile
import numpy as np

test_file = r'E:\Oyvind\BIP-hub-test-data\drift\input\live_cells\1_Meng.nd2'

print("Loading test file...")
img = BioImage(test_file)
data = np.asarray(img.data)

# Just use first frame for speed
data_sample = data[:1]

out1 = Path(tempfile.gettempdir()) / 'test_without_ome.tif'
out2 = Path(tempfile.gettempdir()) / 'test_with_ome.ome.tif'
out3 = Path(tempfile.gettempdir()) / 'test_bioimage_save.ome.tif'

# Test 1: imwrite without ome flag
print("\n1. Testing imwrite WITHOUT ome flag...")
imwrite(out1, data_sample, compression='zlib', metadata={'axes': 'TCZYX'})

# Test 2: imwrite WITH ome=True flag
print("2. Testing imwrite WITH ome=True...")
imwrite(out2, data_sample, compression='zlib', ome=True, metadata={'axes': 'TCZYX'})

# Test 3: BioImage.save() for comparison  
print("3. Testing BioImage.save()...")
# BioImage doesn't allow modifying data, so just save the original
img.save(out3)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n{'Method':<30} {'Size (KB)':<12} {'Is OME?':<10} {'Has OME-XML?'}")
print("-"*60)

with TiffFile(out1) as tif:
    size1 = out1.stat().st_size / 1024
    is_ome1 = tif.is_ome
    has_xml1 = 'ImageDescription' in tif.pages[0].tags and 'OME' in str(tif.pages[0].tags['ImageDescription'].value)[:100]
    print(f"{'imwrite (no ome flag)':<30} {size1:<12.1f} {str(is_ome1):<10} {str(has_xml1)}")

with TiffFile(out2) as tif:
    size2 = out2.stat().st_size / 1024
    is_ome2 = tif.is_ome
    has_xml2 = 'ImageDescription' in tif.pages[0].tags and 'OME' in str(tif.pages[0].tags['ImageDescription'].value)[:100]
    print(f"{'imwrite (ome=True)':<30} {size2:<12.1f} {str(is_ome2):<10} {str(has_xml2)}")

with TiffFile(out3) as tif:
    size3 = out3.stat().st_size / 1024
    is_ome3 = tif.is_ome
    has_xml3 = 'ImageDescription' in tif.pages[0].tags and 'OME' in str(tif.pages[0].tags['ImageDescription'].value)[:100]
    print(f"{'BioImage.save()':<30} {size3:<12.1f} {str(is_ome3):<10} {str(has_xml3)}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if is_ome2:
    print("✅ imwrite with ome=True DOES create proper OME-TIFF")
else:
    print("❌ imwrite with ome=True does NOT create proper OME-TIFF")

if not is_ome1:
    print("❌ imwrite WITHOUT ome=True does NOT create OME-TIFF")
else:
    print("✅ imwrite WITHOUT ome=True still creates OME-TIFF (auto-detected)")

print("\n📝 Recommendation: Use ome=True explicitly for OME-TIFF output")

# Cleanup
out1.unlink(missing_ok=True)
out2.unlink(missing_ok=True)
out3.unlink(missing_ok=True)
