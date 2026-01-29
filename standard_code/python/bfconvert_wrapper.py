"""
Simple CLI wrapper for BioFormats bfconvert tool.

Provides a convenient interface to run bfconvert on multiple files with any flags.
Handles Java installation checks and bfconvert path detection.

Supports all formats readable by BioFormats including:
- OME-TIFF, standard TIFF
- Zeiss (LSM, CZI)
- Leica (LEI, LIF)
- Olympus (OIB, OIF)
- Nikon (ND2)
- And many others

MIT License - BIPHUB, University of Oslo
"""
import argparse
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from glob import glob
import shutil

# Configure logging
logger = logging.getLogger(__name__)


def find_bfconvert() -> str:
    """
    Find bfconvert executable in the standard locations.
    
    Returns:
        Path to bfconvert executable
        
    Raises:
        FileNotFoundError: If bfconvert cannot be found
    """
    # Check if running from repo context
    repo_root = Path(__file__).parent.parent.parent
    potential_paths = [
        repo_root / "external" / "bftools" / "bfconvert.bat",
        repo_root / "external" / "bftools" / "bfconvert",
        shutil.which("bfconvert"),
        shutil.which("bfconvert.bat"),
    ]
    
    for path in potential_paths:
        if path and os.path.exists(path):
            logger.info(f"Found bfconvert at: {path}")
            return str(path)
    
    raise FileNotFoundError(
        "bfconvert not found. Ensure BioFormats is installed or available in the repo at "
        "external/bftools/bfconvert.bat"
    )


def check_java_installed() -> bool:
    """
    Check if Java is installed and accessible.
    
    Returns:
        True if Java is found, False otherwise
        
    Raises:
        RuntimeError: If Java is not found with installation instructions
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Java prints version to stderr
        version_output = result.stderr + result.stdout
        
        if result.returncode == 0 and "version" in version_output.lower():
            # Extract version info
            version_line = version_output.split('\n')[0]
            logger.info(f"Java found: {version_line}")
            return True
        else:
            raise RuntimeError("Java command executed but returned unexpected output")
            
    except FileNotFoundError:
        error_msg = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          JAVA NOT FOUND                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

BioFormats tools (bfconvert, showinf) require Java to be installed.

INSTALLATION INSTRUCTIONS:

Windows:
  1. Download Java from: https://adoptium.net/temurin/releases/
  2. Choose the latest LTS version (Java 17 or 21)
  3. Download the Windows .msi installer
  4. Run the installer and check "Set JAVA_HOME variable"
  5. Restart your terminal/command prompt after installation
  6. Verify installation: java -version

macOS:
  1. Using Homebrew: brew install openjdk
  2. Or download from: https://adoptium.net/temurin/releases/
  3. Verify installation: java -version

Linux:
  Ubuntu/Debian: sudo apt-get install default-jdk
  Fedora/RHEL:   sudo dnf install java-17-openjdk
  Verify installation: java -version

After installation, restart your terminal and try again.
"""
        logger.error(error_msg)
        raise RuntimeError("Java is not installed or not in PATH. See installation instructions above.")
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("Java check timed out. Java may be installed but not responding.")
    
    except Exception as e:
        raise RuntimeError(f"Error checking for Java: {e}")


def run_bfconvert(
    input_file: str,
    output_file: str,
    bfconvert_path: str,
    extra_flags: list[str] = None
) -> bool:
    """
    Run bfconvert on a single file.
    
    Args:
        input_file: Path to input image file
        output_file: Path to output file
        bfconvert_path: Path to bfconvert executable
        extra_flags: Additional flags to pass to bfconvert
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = [str(bfconvert_path)]
        
        # Add user-specified flags
        if extra_flags:
            cmd.extend(extra_flags)
        
        # Add input and output
        cmd.extend([str(input_file), output_file])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None
        )
        
        if result.returncode != 0:
            logger.error(f"bfconvert failed for {input_file}:")
            logger.error(result.stderr)
            return False
        
        logger.info(f"✓ Successfully converted: {input_file} -> {output_file}")
        if result.stdout and result.stdout.strip():
            logger.debug(result.stdout)
        return True
    
    except Exception as e:
        logger.error(f"Error converting file {input_file}: {e}")
        return False


def process_files(
    input_pattern: str,
    output_pattern: str,
    extra_flags: list[str] = None,
    dry_run: bool = False
) -> None:
    """
    Process multiple image files with bfconvert.
    
    Args:
        input_pattern: Glob pattern for input files
        output_pattern: Output pattern with optional placeholders
        extra_flags: Additional flags to pass to bfconvert
        dry_run: Print planned actions without executing
    """
    # Expand glob pattern
    input_files = glob(input_pattern, recursive=True)
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Check Java installation first
    try:
        check_java_installed()
    except RuntimeError as e:
        logger.error(f"Cannot proceed without Java: {e}")
        return
    
    # Find bfconvert
    try:
        bfconvert_path = find_bfconvert()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for input_file in input_files:
        # Determine output file path
        # If output_pattern has bfconvert placeholders (%s, %c, etc.), use it directly
        # Otherwise, use it as a simple output path
        if '%' in output_pattern:
            # Pattern with bfconvert placeholders - use as-is
            output_file = output_pattern
        else:
            # Simple output path - replace input filename
            input_name = Path(input_file).stem
            output_ext = Path(output_pattern).suffix or '.ome.tif'
            output_dir = Path(output_pattern).parent
            output_file = str(output_dir / f"{input_name}{output_ext}")
        
        logger.info(f"Processing: {input_file}")
        logger.info(f"Output: {output_file}")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would convert: {input_file} -> {output_file}")
            logger.info(f"[DRY RUN] Flags: {extra_flags}")
            success_count += 1
            continue
        
        if run_bfconvert(input_file, output_file, bfconvert_path, extra_flags):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"\nCompleted: {success_count} successful, {fail_count} failed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple wrapper for BioFormats bfconvert tool. "
                    "Pass any bfconvert flags directly to the tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert to OME-TIFF
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.nd2'
  - --output-pattern: '%YAML%/output/converted.ome.tif'
  - --flag: -padded
  - --flag: -no-upgrade
  - --flag: -overwrite

- name: Split by channel and timepoint
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.czi'
  - --output-pattern: '%YAML%/output/img_c%c_t%t.ome.tif'
  - --flag: -series
  - --flag: '0'
  - --flag: -padded

- name: Convert with compression
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.oib'
  - --output-pattern: '%YAML%/output/compressed.ome.tif'
  - --flag: -compression
  - --flag: LZW
  - --flag: -overwrite

Common bfconvert flags:
  -series N               Select specific series
  -timepoint N            Select specific timepoint
  -channel N              Select specific channel
  -z N                    Select specific Z slice
  -range START END        Select range of planes
  -tilex N -tiley N       Set tile dimensions
  -compression TYPE       Set compression (LZW, JPEG-2000, JPEG, Uncompressed)
  -padded                 Zero-pad series/channel/z/t numbers
  -overwrite              Overwrite existing files
  -no-upgrade             Don't upgrade file format
  -noflat                 Don't flatten RGB images

For full bfconvert documentation, see:
https://docs.openmicroscopy.org/bio-formats/latest/users/comlinetools/conversion.html
"""
        )
    )
    
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-pattern",
        type=str,
        required=True,
        help="Output file pattern. Can include bfconvert placeholders like %%c (channel), "
             "%%t (timepoint), %%z (Z), %%s (series). Examples: "
             "'output/img_c%%c.ome.tif' or 'output/converted.ome.tif'"
    )
    
    parser.add_argument(
        "--flag",
        action="append",
        dest="flags",
        metavar="ARG",
        help="Add a flag or argument to pass to bfconvert. Can be used multiple times. "
             "For flags starting with -, use quotes or =: --flag=-padded or --flag=\"-padded\". "
             "Example: --flag=-padded --flag=-overwrite --flag=-compression --flag=LZW"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"bfconvert_wrapper.py version: {version}")
        return
    
    # Process files
    process_files(
        input_pattern=args.input_pattern,
        output_pattern=args.output_pattern,
        extra_flags=args.flags or [],
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
