"""
Sobel Edge Detection Filter Module

Applies Sobel edge detection filter to bioimage data.
Part of the BIPHUB Pipeline System.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_sobel_filter(
    input_pattern: str,
    output_folder: str,
    axis: int = -1
) -> None:
    """
    Apply Sobel edge detection filter to images.
    
    Args:
        input_pattern: Glob pattern or path to input images
        output_folder: Output folder for filtered images
        axis: Axis along which to compute gradient (-1 for magnitude, 0 for first axis, 1 for second axis)
    """
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=False)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Processing {len(input_files)} files with Sobel filter (axis={axis})...")
    
    for i, input_file in enumerate(input_files, 1):
        try:
            logger.info(f"[{i}/{len(input_files)}] Processing: {input_file}")
            
            # Load image
            img = rp.load_tczyx_image(input_file)
            filtered_data = np.zeros_like(img.data, dtype=np.float32)
            
            # Process each timepoint and channel
            for t in range(img.dims.T):
                for c in range(img.dims.C):
                    # Get 3D or 2D slice
                    if img.dims.Z > 1:
                        image_slice = img.get_image_data("ZYX", T=t, C=c)
                    else:
                        image_slice = img.get_image_data("YX", T=t, C=c, Z=0)
                    
                    # Apply Sobel filter
                    if axis == -1:
                        # Compute gradient magnitude
                        sx = sobel(image_slice.astype(np.float32), axis=-2)  # Y axis
                        sy = sobel(image_slice.astype(np.float32), axis=-1)  # X axis
                        filtered_slice = np.sqrt(sx**2 + sy**2)
                    else:
                        filtered_slice = sobel(image_slice.astype(np.float32), axis=axis)
                    
                    # Store filtered result back
                    if img.dims.Z > 1:
                        filtered_data[t, c, :, :, :] = filtered_slice
                    else:
                        filtered_data[t, c, 0, :, :] = filtered_slice
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_sobel{input_path.suffix}"
            
            # Save filtered image
            rp.save_tczyx_image(filtered_data, str(output_file), dim_order="TCZYX")
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    logger.info("Sobel filtering complete!")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Apply Sobel edge detection filter to bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python sobel_filter.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --axis -1
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
        "--axis",
        type=int,
        default=-1,
        help="Axis for gradient: 0, 1, or -1 for magnitude (default: -1)"
    )
    
    args = parser.parse_args()
    
    apply_sobel_filter(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        axis=args.axis
    )


if __name__ == "__main__":
    main()
