"""Test file grouping logic"""
import sys
sys.path.insert(0, 'standard_code/python')

import bioimage_pipeline_utils as rp
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

search_patterns = {
    'input': r'E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250702\input_tif\**\*.tif',
    'mask': r'E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250702\output_masks\*_segmentation.tif'
}

print("\n" + "="*80)
print("Testing file grouping with ** pattern...")
print("="*80)

grouped_files = rp.get_grouped_files_to_process(search_patterns, search_subfolders=True)

print(f"\n\nTotal groups: {len(grouped_files)}")
print(f"Groups with 'input': {sum(1 for g in grouped_files.values() if 'input' in g)}")
print(f"Groups with 'mask': {sum(1 for g in grouped_files.values() if 'mask' in g)}")
print(f"Groups with both: {sum(1 for g in grouped_files.values() if 'input' in g and 'mask' in g)}")

print("\n\nFirst 5 groups:")
for i, (basename, group) in enumerate(list(grouped_files.items())[:5]):
    print(f"\nGroup {i}: basename='{basename}'")
    for key, path in group.items():
        import os
        print(f"  {key}: {os.path.basename(path)}")
