"""Quick verification of drift correction quality by comparing input vs output scores."""

import logging
import sys
from pathlib import Path
import bioimage_pipeline_utils as rp
from drift_correction_utils.drift_correct_utils import drift_correction_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
input_file = Path(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15.tif")
output_file = Path(r"E:\Oyvind\BIP-hub-test-data\drift\output\test_2\1_Meng_timecrop_template_rolled-t15_phase_cor_gpu_first.tif")

if not input_file.exists():
    logging.error(f"Input file not found: {input_file}")
    sys.exit(1)
    
if not output_file.exists():
    logging.error(f"Output file not found: {output_file}")
    sys.exit(1)

# Load images
logging.info("Loading input image...")
input_img = rp.load_tczyx_image(str(input_file))
logging.info(f"Input shape: {input_img.shape}")

logging.info("Loading output (corrected) image...")
output_img = rp.load_tczyx_image(str(output_file))
logging.info(f"Output shape: {output_img.shape}")

# Prepare data for scoring (squeeze to remove singleton Z dim)
input_data = input_img.get_image_data("CZYX")[:, 0, :, :]  # Shape: (C, Y, X) per frame
output_data = output_img.get_image_data("CZYX")[:, 0, :, :]

# Compute scores
logging.info("\n" + "="*80)
logging.info("DRIFT CORRECTION QUALITY COMPARISON")
logging.info("="*80)

# Score vs first frame
input_score_first = drift_correction_score(img_data=input_img.data, channel=0, reference='first', central_crop=0.8)
output_score_first = drift_correction_score(img_data=output_img.data, channel=0, reference='first', central_crop=0.8)

logging.info(f"\nScore vs FIRST frame:")
logging.info(f"  Input (uncorrected):  {input_score_first:.4f}")
logging.info(f"  Output (corrected):   {output_score_first:.4f}")
logging.info(f"  Improvement:          {output_score_first - input_score_first:+.4f}")

# Score vs previous frame
input_score_prev = drift_correction_score(img_data=input_img.data, channel=0, reference='previous', central_crop=0.8)
output_score_prev = drift_correction_score(img_data=output_img.data, channel=0, reference='previous', central_crop=0.8)

logging.info(f"\nScore vs PREVIOUS frame:")
logging.info(f"  Input (uncorrected):  {input_score_prev:.4f}")
logging.info(f"  Output (corrected):   {output_score_prev:.4f}")
logging.info(f"  Improvement:          {output_score_prev - input_score_prev:+.4f}")

# Overall assessment
logging.info("\n" + "="*80)
if output_score_first > 0.98:
    logging.info("✅ EXCELLENT: Drift correction achieved high alignment (>0.98)")
elif output_score_first > 0.95:
    logging.info("✅ GOOD: Drift correction shows clear improvement")
elif output_score_first > input_score_first + 0.02:
    logging.info("⚠️  MODERATE: Some improvement but may need parameter tuning")
else:
    logging.info("❌ POOR: Minimal or no improvement - check input data quality")
logging.info("="*80)
