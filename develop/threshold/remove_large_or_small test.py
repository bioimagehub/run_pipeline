from segment_threshold import remove_small_or_large_labels, LabelInfo
from run_pipeline_helper_functions import load_bioio
import yaml
mask_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\colaboration_user_data\20250424_Julia\output_threshold\tmp\Rep3_LaminAKD_001_S00_ch0-1-2_03_apply_threshold.tif"
label_info_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\colaboration_user_data\20250424_Julia\output_threshold\tmp\Rep3_LaminAKD_001_S00_ch0-1-2_03_apply_threshold_labelinfo.yaml"


# with open(label_info_path, 'r') as file:
#     labelinfo = yaml.safe_load(file)
labelinfo = LabelInfo.load(label_info_path)

mask = load_bioio(mask_path)

mask, new_label_info = remove_small_or_large_labels(mask.data, label_info_list = labelinfo, min_sizes=[50], max_sizes=[100], channels= [2])




