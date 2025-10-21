"""
Drift Correction Entry Point - Translation-Only Registration

⚠️ TRANSLATION ONLY: This module only corrects for XYZ shifts.
No rotation, scaling, or non-linear transformations are applied.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Literal, Optional, List, Union
import numpy as np

from bioio import BioImage
import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def process_files(args):

    # Expand glob pattern using standardized helper function
    input_files = rp.get_files_to_process2(args.input_search_pattern, search_subfolders=False)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        exit(1)
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")
    
    # Process each file sequentially
    if args.no_parallel:
        for file_path in input_files:

            output_file_path = os.path.join(
                output_folder,
                f"{Path(file_path).stem}{args.output_suffix}.tif"
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file: {file_path}")
            try:
                img = rp.load_tczyx_image(file_path)
            except Exception as e:
                logger.error(f"Failed to load image {file_path}: {e}")
                continue
            
            # load the appropriate registration function
            if args.method == "phase_cross_correlation":
                from drift_correction_utils.phase_cross_correlation import register_image_xy
            elif args.method == "stackreg":
                from drift_correction_utils.translation_pystackreg import register_image_xy
            elif args.method == "phase_cross_correlation_v2":
                from drift_correction_utils.phase_cross_correlation_v2 import register_image_xy
            elif args.method == "phase_cross_correlation_v3":
                from drift_correction_utils.phase_cross_correlation_v3 import register_image_xy
            else:
                logger.error(f"Unknown method: {args.method}")
                continue        
            
            
            try:
                registered_img, tmats = register_image_xy(
                    img,
                    channel=args.reference_channel,
                    show_progress=True,
                    no_gpu=args.no_gpu,
                    reference=args.reference,
                    crop_fraction=args.crop_fraction
                )

                registered_img.save(output_file_path)

                if not args.no_save_tmats:
                    tmats_file = output_file_path.replace('.tif', '_tmats.npy')
                    np.save(tmats_file, tmats)
                    logger.info(f"Saved transformation matrices to: {tmats_file}")

                logger.info(f"Saved corrected image to: {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to process image {file_path}: {e}")
                continue
        

    else:
        raise NotImplementedError("Parallel processing not implemented in this script version.")



    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete! {len(input_files)} file(s) processed")
    logger.info(f"{'='*60}")



if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Apply translation-only drift correction to TCZYX images"
    )
    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards, e.g., 'data/*.tif')"
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for corrected images"
    )
    parser.add_argument(
        "--output-suffix",
        default="_drift_corrected",
        help="Suffix to add to output filenames (default: _drift_corrected). Example: '_cor' or '_phase_cor'"
    )
    parser.add_argument(
        "--method",
        default="phase_cross_correlation",
        choices=["phase_cross_correlation", "stackreg", "phase_cross_correlation_v2", "phase_cross_correlation_v3"],
        help="Registration method (default: phase_cross_correlation). v3 = GPU-optimized with v2 accuracy"
    )

    parser.add_argument(
        "--reference-channel",
        type=int,
        default=0,
        help="Channel index to use for registration (0-based, default: 0)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="first",
        choices=["first", "previous", "median"],
        help="Reference frame for registration (default: first). 'first' = register all to first frame, 'previous' = register each to previous frame, 'median' = register to median projection"
    )
    parser.add_argument(
        "--no-save-tmats",
        action="store_true",
        help="Do not save transformation matrices (default: False)"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=1.0,
        help="Fraction of image to use for registration (0.0-1.0, default: 1.0). Values < 1.0 crop edges to speed up registration (e.g., 0.8 uses center 80%)"
    )
    
    
    # parser.add_argument(
    #     "--upsample-factor",
    #     type=int,
    #     default=10,
    #     help="Subpixel accuracy for phase_cross_correlation (10 = 0.1 pixel, default: 10)"
    # )
    # parser.add_argument(
    #     "--max-shift",
    #     type=float,
    #     default=50.0,
    #     help="Maximum expected shift in pixels. Warning issued if exceeded (default: 50.0)"
    # )
    # parser.add_argument(
    #     "--bandpass-low-sigma",
    #     type=float,
    #     default=None,
    #     help="Lower sigma for DoG bandpass filter (suppresses structures smaller than this, e.g., 20 for vesicles). Must be used with --bandpass-high-sigma"
    # )
    # parser.add_argument(
    #     "--bandpass-high-sigma",
    #     type=float,
    #     default=None,
    #     help="Upper sigma for DoG bandpass filter (preserves structures larger than this, e.g., 100 for cells). Must be used with --bandpass-low-sigma"
    # )


    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    process_files(args)
    
    
