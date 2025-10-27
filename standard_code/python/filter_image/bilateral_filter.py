"""
Bilateral Filter Module

Applies bilateral (edge-preserving) filter to bioimage data.
Part of the BIPHUB Pipeline System.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_bilateral_filter(
    input_pattern: str,
    output_folder: str,
    sigma_spatial: float = 5.0,
    sigma_intensity: float = 50.0
) -> None:
    """
    Apply bilateral (edge-preserving) filter to images.
    
    Args:
        input_pattern: Glob pattern or path to input images
        output_folder: Output folder for filtered images
        sigma_spatial: Spatial sigma for bilateral filter
        sigma_intensity: Intensity sigma for bilateral filter
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) not installed. Install with: pip install opencv-python")
        return
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=False)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Processing {len(input_files)} files with bilateral filter...")
    logger.info(f"Parameters: sigma_spatial={sigma_spatial}, sigma_intensity={sigma_intensity}")
    
    for i, input_file in enumerate(input_files, 1):
        try:
            logger.info(f"[{i}/{len(input_files)}] Processing: {input_file}")
            
            # Load image
            img = rp.load_tczyx_image(input_file)
            filtered_data = np.zeros_like(img.data)
            
            # Calculate diameter from spatial sigma
            d = int(sigma_spatial * 2)
            
            # Process each timepoint and channel
            for t in range(img.dims.T):
                for c in range(img.dims.C):
                    # Get 3D or 2D slice
                    if img.dims.Z > 1:
                        image_slice = img.get_image_data("ZYX", T=t, C=c)
                        # Process slice by slice for 3D
                        filtered_slice = np.zeros_like(image_slice)
                        for z in range(image_slice.shape[0]):
                            # Normalize to uint8 for OpenCV
                            slice_2d = image_slice[z]
                            if slice_2d.dtype != np.uint8:
                                slice_normalized = ((slice_2d - slice_2d.min()) / 
                                                    (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)
                            else:
                                slice_normalized = slice_2d.astype(np.uint8)
                            
                            filtered_slice[z] = cv2.bilateralFilter(
                                slice_normalized,
                                d=d,
                                sigmaColor=sigma_intensity,
                                sigmaSpace=sigma_spatial
                            )
                    else:
                        image_slice = img.get_image_data("YX", T=t, C=c, Z=0)
                        # Normalize to uint8 for OpenCV
                        if image_slice.dtype != np.uint8:
                            image_normalized = ((image_slice - image_slice.min()) / 
                                                (image_slice.max() - image_slice.min()) * 255).astype(np.uint8)
                        else:
                            image_normalized = image_slice.astype(np.uint8)
                        
                        filtered_slice = cv2.bilateralFilter(
                            image_normalized,
                            d=d,
                            sigmaColor=sigma_intensity,
                            sigmaSpace=sigma_spatial
                        )
                    
                    # Store filtered result back
                    if img.dims.Z > 1:
                        filtered_data[t, c, :, :, :] = filtered_slice
                    else:
                        filtered_data[t, c, 0, :, :] = filtered_slice
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_bilateral{input_path.suffix}"
            
            # Save filtered image
            rp.save_tczyx_image(filtered_data, str(output_file), dim_order="TCZYX")
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    logger.info("Bilateral filtering complete!")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Apply bilateral (edge-preserving) filter to bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python bilateral_filter.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --sigma-spatial 5 --sigma-intensity 50
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
        "--sigma-spatial",
        type=float,
        default=5.0,
        help="Spatial sigma for bilateral filter (default: 5.0)"
    )
    
    parser.add_argument(
        "--sigma-intensity",
        type=float,
        default=50.0,
        help="Intensity sigma for bilateral filter (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    apply_bilateral_filter(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        sigma_spatial=args.sigma_spatial,
        sigma_intensity=args.sigma_intensity
    )


if __name__ == "__main__":
    main()
