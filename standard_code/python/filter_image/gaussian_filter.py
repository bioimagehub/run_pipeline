"""
Gaussian Filter Module

Applies Gaussian blur filter to bioimage data.
Part of the BIPHUB Pipeline System.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_gaussian_filter(
    input_pattern: str,
    output_folder: str,
    sigma: float = 1.0
) -> None:
    """
    Apply Gaussian blur filter to images.
    
    Args:
        input_pattern: Glob pattern or path to input images
        output_folder: Output folder for filtered images
        sigma: Standard deviation for Gaussian kernel
    """
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=False)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Processing {len(input_files)} files with Gaussian filter (sigma={sigma})...")
    
    for i, input_file in enumerate(input_files, 1):
        try:
            logger.info(f"[{i}/{len(input_files)}] Processing: {input_file}")
            
            # Load image
            img = rp.load_tczyx_image(input_file)
            filtered_data = np.zeros_like(img.data)
            
            # Process each timepoint and channel
            for t in range(img.dims.T):
                for c in range(img.dims.C):
                    # Get 3D or 2D slice
                    if img.dims.Z > 1:
                        image_slice = img.get_image_data("ZYX", T=t, C=c)
                    else:
                        image_slice = img.get_image_data("YX", T=t, C=c, Z=0)
                    
                    # Apply Gaussian filter
                    filtered_slice = gaussian_filter(image_slice.astype(np.float32), sigma=sigma)
                    
                    # Store filtered result back
                    if img.dims.Z > 1:
                        filtered_data[t, c, :, :, :] = filtered_slice
                    else:
                        filtered_data[t, c, 0, :, :] = filtered_slice
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_gaussian{input_path.suffix}"
            
            # Save filtered image (save_tczyx_image accepts numpy arrays directly)
            rp.save_tczyx_image(filtered_data, str(output_file), dim_order="TCZYX")
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    logger.info("Gaussian filtering complete!")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Apply Gaussian blur filter to bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python gaussian_filter.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --sigma 2.0
        """
    )
    
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=True,
        help="Glob pattern for input images (e.g., './input/*.tif')"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for filtered images"
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Standard deviation for Gaussian kernel (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    apply_gaussian_filter(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        sigma=args.sigma
    )


if __name__ == "__main__":
    main()
