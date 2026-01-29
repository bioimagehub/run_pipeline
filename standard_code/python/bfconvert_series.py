#!/usr/bin/env python3
"""
bfconvert wrapper to split LIF files by series using bioformats bfconvert tool.

Processes all .lif files in input folder and converts each series to separate OME-TIFF files.
"""

import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_bfconvert(
    input_folder: Path,
    output_folder: Path,
    pattern: str = "*.lif",
    compression: str = "LZW",
    bigtiff: bool = True,
    padded: bool = True,
    dry_run: bool = False
) -> None:
    """
    Convert all LIF files in input folder to OME-TIFF by series.
    
    Args:
        input_folder: Path to folder containing .lif files
        output_folder: Path to output folder for OME-TIFF files
        pattern: File pattern to match (default: *.lif)
        compression: Compression type (default: LZW)
        bigtiff: Use BigTIFF format (default: True)
        padded: Use padded filenames (default: True)
        dry_run: Print commands without executing (default: False)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        logger.error(f"Input folder does not exist: {input_path}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_path}")
    
    # Find all matching files
    lif_files = list(input_path.glob(pattern))
    
    if not lif_files:
        logger.warning(f"No files matching '{pattern}' found in {input_path}")
        return
    
    logger.info(f"Found {len(lif_files)} file(s) to process")
    
    # Process each LIF file
    for lif_file in lif_files:
        logger.info(f"Processing: {lif_file.name}")
        
        # Build bfconvert command
        cmd = ["bfconvert"]
        
        if padded:
            cmd.append("-padded")
        if bigtiff:
            cmd.append("-bigtiff")
        
        cmd.extend(["-compression", compression])
        cmd.append(str(lif_file))
        
        # Output pattern: input_name_S%s.ome.tif
        # %s will be replaced by bfconvert with the series number
        output_pattern = output_path / f"{lif_file.stem}_S%s.ome.tif"
        cmd.append(str(output_pattern))
        
        if dry_run:
            logger.info(f"[DRY RUN] Command: {' '.join(cmd)}")
        else:
            try:
                logger.debug(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"✓ Successfully converted: {lif_file.name}")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to convert {lif_file.name}")
                logger.error(f"  Error: {e.stderr}")
            except FileNotFoundError:
                logger.error("bfconvert not found. Ensure bioformats tools are installed and on PATH.")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIF files by series to OME-TIFF using bfconvert",
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert LIF files to OME-TIFF by series
  environment: uv@3.11:convert_to_tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_series.py'
  - --input-folder: '%YAML%/bfconvert_input'
  - --output-folder: '%YAML%/bfconvert_output'
  - --compression: LZW
  - --bigtiff
  - --padded

- name: Convert with custom parameters
  environment: uv@3.11:convert_to_tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_series.py'
  - --input-folder: '%YAML%/input'
  - --output-folder: '%YAML%/output'
  - --pattern: '*.lif'
  - --compression: JPEG
  - --dry-run
"""
    )
    
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Input folder containing .lif files"
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Output folder for converted OME-TIFF files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.lif",
        help="File pattern to match (default: *.lif)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="LZW",
        choices=["LZW", "JPEG", "JPEG-2000", "None"],
        help="Compression type (default: LZW)"
    )
    parser.add_argument(
        "--bigtiff",
        action="store_true",
        default=True,
        help="Use BigTIFF format (default: enabled)"
    )
    parser.add_argument(
        "--no-bigtiff",
        action="store_false",
        dest="bigtiff",
        help="Disable BigTIFF format"
    )
    parser.add_argument(
        "--padded",
        action="store_true",
        default=True,
        help="Use padded filenames (default: enabled)"
    )
    parser.add_argument(
        "--no-padded",
        action="store_false",
        dest="padded",
        help="Disable padded filenames"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    args = parser.parse_args()
    
    run_bfconvert(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        compression=args.compression,
        bigtiff=args.bigtiff,
        padded=args.padded,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
