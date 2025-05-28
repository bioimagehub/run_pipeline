from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import run_pipeline_helper_functions as rp

img_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\colaboration_user_data\20250424_Julia\input_tif\Rep3_LaminAKD_001__1.tif"

img = rp.load_bioio(img_path).data[0, 0, 0, :, :]

img = median_filter(img, size=5)

print(img.shape)


fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()