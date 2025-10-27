"""
Top-Hat Morphological Filter Module

Applies morphological top-hat filter to bioimage data.
Part of the BIPHUB Pipeline System.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import white_tophat, black_tophat, generate_binary_structure, binary_dilation
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_tophat_filter(
    input_pattern: str,
    output_folder: str,
    size: int = 3,
    mode: str = 'white'
) -> None:
    """
    Apply morphological top-hat filter to images.
    
    Args:
        input_pattern: Glob pattern or path to input images
        output_folder: Output folder for filtered images
        size: Size of the structuring element
        mode: 'white' for white top-hat, 'black' for black top-hat
    """
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=False)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Processing {len(input_files)} files with {mode} top-hat filter (size={size})...")
    
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
                    
                    # Create structuring element
                    rank = image_slice.ndim
                    footprint = generate_binary_structure(rank, 1)
                    
                    # Scale footprint to desired size
                    for _ in range(size - 1):
                        footprint = binary_dilation(footprint)
                    
                    # Apply top-hat filter
                    if mode == 'white':
                        filtered_slice = white_tophat(image_slice, footprint=footprint)
                    else:
                        filtered_slice = black_tophat(image_slice, footprint=footprint)
                    
                    # Store filtered result back
                    if img.dims.Z > 1:
                        filtered_data[t, c, :, :, :] = filtered_slice
                    else:
                        filtered_data[t, c, 0, :, :] = filtered_slice
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_tophat_{mode}{input_path.suffix}"
            
            # Save filtered image
            rp.save_tczyx_image(filtered_data, str(output_file), dim_order="TCZYX")
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    logger.info("Top-hat filtering complete!")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Apply morphological top-hat filter to bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tophat_filter.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --size 5 --mode white
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
        "--size",
        type=int,
        default=3,
        help="Size of the structuring element (default: 3)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Mode for tophat filter: 'white' or 'black' (default: white)"
    )
    
    args = parser.parse_args()
    
    apply_tophat_filter(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        size=args.size,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
