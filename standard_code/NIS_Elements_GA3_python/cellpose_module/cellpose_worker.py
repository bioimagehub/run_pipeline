#!/usr/bin/env python
"""
Cellpose Worker Script - Runs in UV-managed isolated environment

This script is called by GA3 as a subprocess with image data passed via
temporary numpy files. It performs Cellpose segmentation and returns masks.

Author: BIPHUB Team
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from cellpose import models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def segment_image(
    image: np.ndarray,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    channels: Optional[list] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    use_gpu: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Segment cells using Cellpose.

    Args:
        image: Input image as numpy array (2D or 3D)
        model_type: Cellpose model ('cyto', 'cyto2', 'cyto3', 'nuclei', etc.)
        diameter: Expected cell diameter in pixels (None for auto-estimate)
        channels: [cytoplasm, nucleus] channels (None for grayscale)
        flow_threshold: Flow error threshold (all cells with errors below threshold are kept)
        cellprob_threshold: Cell probability threshold
        use_gpu: Whether to use GPU acceleration

    Returns:
        tuple: (masks, info_dict)
            - masks: Labeled mask array (0=background, 1,2,3...=cell IDs)
            - info_dict: Additional information (flows, diameters, etc.)
    """
    logger.info(f"Initializing Cellpose model: {model_type}")
    logger.info(f"GPU enabled: {use_gpu}")
    
    # Initialize model
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    
    logger.info(f"Running segmentation on image shape: {image.shape}")
    logger.info(f"Parameters: diameter={diameter}, flow_threshold={flow_threshold}, "
                f"cellprob_threshold={cellprob_threshold}")
    
    # Run segmentation
    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    
    n_cells = masks.max()
    logger.info(f"Segmentation complete. Found {n_cells} cells.")
    
    info = {
        "n_cells": int(n_cells),
        "estimated_diameter": float(diams[0]) if isinstance(diams, np.ndarray) else float(diams),
        "flows_shape": flows[0].shape if isinstance(flows, tuple) else None,
    }
    
    return masks, info


def main():
    parser = argparse.ArgumentParser(
        description="Cellpose segmentation worker for GA3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .npy file containing image"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output .npy file for masks"
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        help="Optional path to save segmentation info as JSON"
    )
    
    # Cellpose parameters
    parser.add_argument(
        "--model",
        type=str,
        default="cyto3",
        choices=["cyto", "cyto2", "cyto3", "nuclei", "tissuenet", "livecell", "cyto2torch", "CP", "CPx", "TN1", "TN2", "TN3", "LC1", "LC2", "LC3", "LC4"],
        help="Cellpose model type"
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Expected cell diameter in pixels (None for auto-estimate)"
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Flow error threshold"
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file exists
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        logger.info(f"Loading image from: {args.input}")
        image = np.load(args.input)
        logger.info(f"Image loaded. Shape: {image.shape}, dtype: {image.dtype}")
        
        # Run segmentation
        masks, info = segment_image(
            image=image,
            model_type=args.model,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            use_gpu=not args.no_gpu,
        )
        
        # Save masks
        logger.info(f"Saving masks to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, masks)
        
        # Save info if requested
        if args.info_output:
            import json
            logger.info(f"Saving info to: {args.info_output}")
            args.info_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.info_output, 'w') as f:
                json.dump(info, f, indent=2)
        
        logger.info("✓ Cellpose segmentation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"✗ Cellpose segmentation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
