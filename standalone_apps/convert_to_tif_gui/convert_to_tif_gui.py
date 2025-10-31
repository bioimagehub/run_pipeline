"""
BIPHUB Image Converter - Standalone GUI Application
====================================================

This is a GUI wrapper around the core convert_to_tif.py module,
designed to be packaged as a standalone .exe for users without Python.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Import local copies of core modules
# These are copied during build process with 'local_' prefix
try:
    import local_convert_to_tif as convert_to_tif
except ImportError as e:
    print(f"ERROR: Failed to import local_convert_to_tif: {e}")
    print("Make sure to run the build script first to copy the core modules.")
    sys.exit(1)

# Now import Gooey
try:
    from gooey import Gooey, GooeyParser
except ImportError:
    print("ERROR: Gooey is not installed.")
    print("Install with: uv pip install gooey")
    sys.exit(1)

import argparse


@Gooey(
    program_name="BIPHUB Image Converter",
    program_description="Convert microscopy images to OME-TIFF format with optional Z-projection",
    default_size=(900, 700),
    navigation='TABBED',
    tabbed_groups=True,
    show_success_modal=True,
    show_failure_modal=True,
    richtext_controls=True,
    menu=[{
        'name': 'Help',
        'items': [{
            'type': 'AboutDialog',
            'menuTitle': 'About',
            'name': 'BIPHUB Image Converter',
            'description': 'Convert microscopy images to OME-TIFF format',
            'version': '1.0.0',
            'copyright': '2024-2025 BIPHUB, University of Oslo',
            'website': 'https://www.uio.no/tjenester/it/forskning/kompetansehuber/biphub/',
            'license': 'MIT'
        }]
    }]
)
def main():
    """Main GUI application entry point."""
    
    parser = GooeyParser(
        description="Convert microscopy images to OME-TIFF format with optional Z-projection.\n\n"
                    "Supports multiple input formats: .czi, .nd2, .tif, .lif, and more."
    )
    
    # === INPUT/OUTPUT GROUP ===
    io_group = parser.add_argument_group(
        "Input/Output Settings",
        "Specify input files and output location",
        gooey_options={'columns': 1}
    )
    
    io_group.add_argument(
        "--input-search-pattern",
        metavar="Input Files",
        widget="FileChooser",
        help="Select input image file(s). Supports wildcards like 'data/*.czi' or 'data/**/*.tif' for recursive search.",
        gooey_options={
            'wildcard': "Image files (*.czi;*.nd2;*.tif;*.tiff;*.lif;*.ims)|*.czi;*.nd2;*.tif;*.tiff;*.lif;*.ims|All files (*.*)|*.*",
            'message': "Select input image file(s)"
        }
    )
    
    io_group.add_argument(
        "--output-folder",
        metavar="Output Folder",
        widget="DirChooser",
        help="Folder where converted images will be saved. If not specified, creates a '_tif' folder next to input.",
        gooey_options={
            'message': "Select output folder"
        },
        default=""
    )
    
    io_group.add_argument(
        "--output-file-name-extension",
        metavar="Output Filename Extension",
        help="Optional text to append to output filenames (e.g., '_converted')",
        default=""
    )
    
    # === PROCESSING OPTIONS GROUP ===
    proc_group = parser.add_argument_group(
        "Processing Options",
        "Configure image processing parameters"
    )
    
    proc_group.add_argument(
        "--projection-method",
        metavar="Z-Projection Method",
        widget="Dropdown",
        choices=['None', 'max', 'sum', 'mean', 'median', 'min', 'std'],
        default='None',
        help="Method for Z-projection (collapsing Z-stacks into 2D). Use 'None' to keep 3D stacks."
    )
    
    # === ADVANCED OPTIONS GROUP ===
    adv_group = parser.add_argument_group(
        "Advanced Options",
        "Optional advanced settings (usually not needed)"
    )
    
    adv_group.add_argument(
        "--collapse-delimiter",
        metavar="Subfolder Delimiter",
        default="__",
        help="When using recursive search (**), this delimiter is used to flatten subfolder names (e.g., 'folder1__folder2__image.tif')"
    )
    
    adv_group.add_argument(
        "--dry-run",
        metavar="Dry Run (Preview Only)",
        widget="CheckBox",
        action="store_true",
        help="Preview what would be done without actually converting any files"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # === POST-PROCESSING OF ARGUMENTS ===
    
    # Convert 'None' string back to actual None for projection method
    if args.projection_method == 'None':
        args.projection_method = None
    
    # Handle empty output folder (convert empty string to None so default logic works)
    if not args.output_folder or args.output_folder.strip() == "":
        args.output_folder = None
    
    # Ensure required argument is provided
    if not args.input_search_pattern or args.input_search_pattern.strip() == "":
        print("ERROR: Input files are required!")
        sys.exit(1)
    
    # === CALL CORE PROCESSING LOGIC ===
    try:
        # Configure logging for GUI display
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        print("=" * 60)
        print("BIPHUB Image Converter")
        print("=" * 60)
        print(f"Input pattern: {args.input_search_pattern}")
        print(f"Output folder: {args.output_folder or '(auto-generated)'}")
        print(f"Projection: {args.projection_method or 'None'}")
        print(f"Dry run: {args.dry_run}")
        print("=" * 60)
        print()
        
        # Call the core convert_to_tif processing function
        convert_to_tif.process_pattern(args)
        
        print()
        print("=" * 60)
        print("[SUCCESS] CONVERSION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("[ERROR] CONVERSION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
