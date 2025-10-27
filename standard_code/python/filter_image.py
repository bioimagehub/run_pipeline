"""
Image Filtering Module

Applies various image filters including Gaussian, median, mean, bilateral, and more.
Part of the BIPHUB Pipeline System.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB
"""

import argparse
import logging
from pathlib import Path
from typing import Literal
import numpy as np
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_gaussian_filter(
    image: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Apply Gaussian blur filter.
    
    Args:
        image: Input image array (2D or 3D)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(np.float32), sigma=sigma)


def apply_median_filter(
    image: np.ndarray,
    size: int
) -> np.ndarray:
    """
    Apply median filter.
    
    Args:
        image: Input image array (2D or 3D)
        size: Size of the median filter kernel
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import median_filter
    return median_filter(image, size=size)


def apply_mean_filter(
    image: np.ndarray,
    size: int
) -> np.ndarray:
    """
    Apply mean (uniform) filter.
    
    Args:
        image: Input image array (2D or 3D)
        size: Size of the mean filter kernel
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(image.astype(np.float32), size=size)


def apply_bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_intensity: float
) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving).
    
    Args:
        image: Input image array (2D only)
        sigma_spatial: Spatial sigma for bilateral filter
        sigma_intensity: Intensity sigma for bilateral filter
        
    Returns:
        Filtered image
    """
    try:
        import cv2
        # cv2.bilateralFilter requires uint8 or float32
        if image.dtype != np.uint8 and image.dtype != np.float32:
            image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_normalized = image.astype(np.uint8)
        
        # Calculate diameter from spatial sigma
        d = int(sigma_spatial * 2)
        
        filtered = cv2.bilateralFilter(
            image_normalized,
            d=d,
            sigmaColor=sigma_intensity,
            sigmaSpace=sigma_spatial
        )
        return filtered.astype(image.dtype)
    except ImportError:
        logger.error("OpenCV (cv2) not installed. Install with: pip install opencv-python")
        raise


def apply_sobel_filter(
    image: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Sobel edge detection filter.
    
    Args:
        image: Input image array (2D or 3D)
        axis: Axis along which to compute gradient (-1 for magnitude)
        
    Returns:
        Edge-detected image
    """
    from scipy.ndimage import sobel
    if axis == -1:
        # Compute gradient magnitude
        sx = sobel(image.astype(np.float32), axis=0)
        sy = sobel(image.astype(np.float32), axis=1)
        return np.sqrt(sx**2 + sy**2)
    else:
        return sobel(image.astype(np.float32), axis=axis)


def apply_laplacian_filter(
    image: np.ndarray
) -> np.ndarray:
    """
    Apply Laplacian filter for edge detection.
    
    Args:
        image: Input image array (2D or 3D)
        
    Returns:
        Edge-detected image
    """
    from scipy.ndimage import laplace
    return laplace(image.astype(np.float32))


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float,
    amount: float
) -> np.ndarray:
    """
    Apply unsharp mask filter for sharpening.
    
    Args:
        image: Input image array (2D or 3D)
        sigma: Sigma for Gaussian blur
        amount: Amount of sharpening (typically 0.5-2.0)
        
    Returns:
        Sharpened image
    """
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    sharpened = image.astype(np.float32) + amount * (image.astype(np.float32) - blurred)
    return np.clip(sharpened, image.min(), image.max()).astype(image.dtype)


def apply_tophat_filter(
    image: np.ndarray,
    size: int,
    mode: Literal['white', 'black'] = 'white'
) -> np.ndarray:
    """
    Apply morphological top-hat filter.
    
    Args:
        image: Input image array (2D or 3D)
        size: Size of the structuring element
        mode: 'white' for white top-hat, 'black' for black top-hat
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import white_tophat, black_tophat, generate_binary_structure
    
    # Create structuring element
    rank = image.ndim
    footprint = generate_binary_structure(rank, 1)
    
    # Scale footprint to desired size
    from scipy.ndimage import binary_dilation
    for _ in range(size - 1):
        footprint = binary_dilation(footprint)
    
    if mode == 'white':
        return white_tophat(image, footprint=footprint)
    else:
        return black_tophat(image, footprint=footprint)


def filter_image(
    input_pattern: str,
    output_folder: str,
    filter_type: str,
    sigma: float = 1.0,
    size: int = 3,
    sigma_spatial: float = 5.0,
    sigma_intensity: float = 50.0,
    amount: float = 1.0,
    tophat_mode: str = 'white',
    axis: int = -1
) -> None:
    """
    Apply filtering to images.
    
    Args:
        input_pattern: Glob pattern or path to input images
        output_folder: Output folder for filtered images
        filter_type: Type of filter to apply
        sigma: Sigma parameter for Gaussian/unsharp filters
        size: Size parameter for median/mean/tophat filters
        sigma_spatial: Spatial sigma for bilateral filter
        sigma_intensity: Intensity sigma for bilateral filter
        amount: Amount for unsharp mask
        tophat_mode: Mode for tophat filter ('white' or 'black')
        axis: Axis for Sobel filter (-1 for magnitude)
    """
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = rp.get_files_from_pattern(input_pattern)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Processing {len(input_files)} files with {filter_type} filter...")
    
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
                    
                    # Apply selected filter
                    if filter_type == "gaussian":
                        filtered_slice = apply_gaussian_filter(image_slice, sigma)
                    elif filter_type == "median":
                        filtered_slice = apply_median_filter(image_slice, size)
                    elif filter_type == "mean":
                        filtered_slice = apply_mean_filter(image_slice, size)
                    elif filter_type == "bilateral":
                        if image_slice.ndim > 2:
                            logger.warning("Bilateral filter only supports 2D images. Processing slice by slice...")
                            filtered_slice = np.zeros_like(image_slice)
                            for z in range(image_slice.shape[0]):
                                filtered_slice[z] = apply_bilateral_filter(
                                    image_slice[z], sigma_spatial, sigma_intensity
                                )
                        else:
                            filtered_slice = apply_bilateral_filter(
                                image_slice, sigma_spatial, sigma_intensity
                            )
                    elif filter_type == "sobel":
                        filtered_slice = apply_sobel_filter(image_slice, axis)
                    elif filter_type == "laplacian":
                        filtered_slice = apply_laplacian_filter(image_slice)
                    elif filter_type == "unsharp":
                        filtered_slice = apply_unsharp_mask(image_slice, sigma, amount)
                    elif filter_type == "tophat":
                        filtered_slice = apply_tophat_filter(image_slice, size, tophat_mode)
                    else:
                        logger.error(f"Unknown filter type: {filter_type}")
                        return
                    
                    # Store filtered result back
                    if img.dims.Z > 1:
                        filtered_data[t, c, :, :, :] = filtered_slice
                    else:
                        filtered_data[t, c, 0, :, :] = filtered_slice
            
            # Create output image
            output_img = rp.ImageData(
                data=filtered_data,
                dims=img.dims,
                metadata=img.metadata
            )
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_{filter_type}{input_path.suffix}"
            
            # Save filtered image
            rp.save_tczyx_image(output_img, str(output_file))
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    logger.info("Filtering complete!")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Apply various image filters to bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply Gaussian blur
  python filter_image.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --filter-type gaussian --sigma 2.0
  
  # Apply median filter
  python filter_image.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --filter-type median --size 5
  
  # Apply bilateral filter (edge-preserving)
  python filter_image.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --filter-type bilateral --sigma-spatial 5 --sigma-intensity 50
  
  # Apply unsharp mask for sharpening
  python filter_image.py --input-pattern "./input/*.tif" --output-folder "./output/filtered" --filter-type unsharp --sigma 1.0 --amount 1.5
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
        "--filter-type",
        type=str,
        required=True,
        choices=["gaussian", "median", "mean", "bilateral", "sobel", "laplacian", "unsharp", "tophat"],
        help="Type of filter to apply"
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian/unsharp filters (default: 1.0)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=3,
        help="Size for median/mean/tophat filters (default: 3)"
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
    
    parser.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="Amount for unsharp mask sharpening (default: 1.0)"
    )
    
    parser.add_argument(
        "--tophat-mode",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Mode for tophat filter: 'white' or 'black' (default: white)"
    )
    
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        help="Axis for Sobel filter: 0, 1, or -1 for magnitude (default: -1)"
    )
    
    args = parser.parse_args()
    
    filter_image(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        filter_type=args.filter_type,
        sigma=args.sigma,
        size=args.size,
        sigma_spatial=args.sigma_spatial,
        sigma_intensity=args.sigma_intensity,
        amount=args.amount,
        tophat_mode=args.tophat_mode,
        axis=args.axis
    )


if __name__ == "__main__":
    main()
