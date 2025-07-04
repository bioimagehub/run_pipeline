

import run_pipeline_helper_functions as rp

path = r"C:\Users\oodegard\Documents\BIPHub_files\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\SP20250627\output_masks\SP20250625_PC_R3_WT011_mask.tif"
mask = rp.load_bioio(path)  # TCZYX

if mask is None or mask.data is None:
    print(f"Error: Mask data is None for file {path}.") 

print(mask)
