"""
Organize Andor Fusion files into snapshots and acquisitions folders.
Analyzes _metadata.txt files to distinguish between snapshots and protocol acquisitions.

Snapshots: Files without 'Protocol Name=MultiFieldProtocolWizard' in metadata
Acquisitions: Files with 'Protocol Name=MultiFieldProtocolWizard' in metadata

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def is_snapshot(metadata_path: str) -> bool:
    """
    Determine if a metadata file represents a snapshot.
    
    Snapshots do not have 'Protocol Name=MultiFieldProtocolWizard' in their metadata.
    
    Args:
        metadata_path: Path to _metadata.txt file
    
    Returns:
        True if file is a snapshot, False if it's a protocol acquisition
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Check if the protocol line exists
            return 'Protocol Name=MultiFieldProtocolWizard' not in content
    except Exception as e:
        logger.error(f"Failed to read {metadata_path}: {e}")
        return False


def find_paired_files(base_path: str, metadata_file: str) -> List[str]:
    """
    Find all files associated with a metadata file.
    
    Looks for files with the same basename (removing _metadata.txt suffix).
    
    Args:
        base_path: Directory containing the files
        metadata_file: Name of the metadata file (e.g., 'file_metadata.txt')
    
    Returns:
        List of file paths that should be moved together
    """
    # Extract basename by removing _metadata.txt
    if metadata_file.endswith('_metadata.txt'):
        basename = metadata_file[:-13]  # Remove '_metadata.txt'
    else:
        logger.warning(f"Unexpected metadata filename format: {metadata_file}")
        return [metadata_file]
    
    # Find all files with this basename
    paired_files = []
    
    # Add the metadata file itself
    metadata_path = os.path.join(base_path, metadata_file)
    if os.path.exists(metadata_path):
        paired_files.append(metadata_path)
    
    # Look for image files with matching basename
    # Common extensions: .ims, .tif, .tiff, etc.
    common_extensions = ['.ims', '.tif', '.tiff', '.nd2', '.czi', '.lif']
    
    for ext in common_extensions:
        image_path = os.path.join(base_path, basename + ext)
        if os.path.exists(image_path):
            paired_files.append(image_path)
    
    return paired_files


def organize_files(
    input_folder: str,
    snapshot_folder_name: str = "snapshots",
    dry_run: bool = False
) -> None:
    """
    Organize Fusion files into snapshots and leave acquisitions in place.
    
    Args:
        input_folder: Folder containing _metadata.txt and image files
        snapshot_folder_name: Name of subfolder for snapshot files
        dry_run: If True, only print planned actions without moving files
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Find all metadata files
    metadata_files = list(input_path.glob("*_metadata.txt"))
    
    if not metadata_files:
        logger.warning(f"No *_metadata.txt files found in {input_folder}")
        return
    
    logger.info(f"Found {len(metadata_files)} metadata file(s)")
    
    # Categorize files
    snapshot_files = []
    acquisition_files = []
    
    for metadata_file in metadata_files:
        metadata_name = metadata_file.name
        
        if is_snapshot(str(metadata_file)):
            # Find all associated files
            paired = find_paired_files(input_folder, metadata_name)
            snapshot_files.extend(paired)
            logger.info(f"Snapshot: {metadata_name} (+ {len(paired) - 1} paired file(s))")
        else:
            paired = find_paired_files(input_folder, metadata_name)
            acquisition_files.extend(paired)
            logger.info(f"Acquisition: {metadata_name} (+ {len(paired) - 1} paired file(s))")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Snapshots: {len(snapshot_files)} file(s)")
    logger.info(f"  Acquisitions: {len(acquisition_files)} file(s)")
    
    if not snapshot_files:
        logger.info("No snapshot files to move. Nothing to do.")
        return
    
    # Create snapshots folder
    snapshot_folder = input_path / snapshot_folder_name
    
    if dry_run:
        print(f"\n[DRY RUN] Would create folder: {snapshot_folder}")
        print(f"[DRY RUN] Would move {len(snapshot_files)} snapshot file(s):")
        for file_path in snapshot_files:
            dest = snapshot_folder / Path(file_path).name
            print(f"  {file_path} -> {dest}")
        print(f"\n[DRY RUN] Would leave {len(acquisition_files)} acquisition file(s) in place")
        return
    
    # Create snapshot folder
    snapshot_folder.mkdir(exist_ok=True)
    logger.info(f"\nCreated folder: {snapshot_folder}")
    
    # Move snapshot files
    moved_count = 0
    failed_count = 0
    
    for file_path in snapshot_files:
        try:
            file_name = Path(file_path).name
            dest_path = snapshot_folder / file_name
            
            # Check if destination already exists
            if dest_path.exists():
                logger.warning(f"Destination already exists, skipping: {dest_path}")
                failed_count += 1
                continue
            
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Moved: {file_name} -> {snapshot_folder_name}/")
            moved_count += 1
            
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
            failed_count += 1
    
    # Final summary
    logger.info(f"\nCompleted:")
    logger.info(f"  Moved: {moved_count} file(s)")
    if failed_count > 0:
        logger.warning(f"  Failed: {failed_count} file(s)")
    logger.info(f"  Remaining acquisitions: {len(acquisition_files)} file(s)")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Organize Andor Fusion files by moving snapshots to a subfolder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize files in a folder
  python organize_fusion_files.py --input-folder "Z:\\experiment_data"
  
  # Dry run to preview actions
  python organize_fusion_files.py --input-folder "Z:\\data" --dry-run
  
  # Use custom snapshot folder name
  python organize_fusion_files.py --input-folder "Z:\\data" --snapshot-folder-name "snap_images"
        """
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Folder containing _metadata.txt and image files"
    )
    
    parser.add_argument(
        "--snapshot-folder-name",
        type=str,
        default="snapshots",
        help="Name of subfolder for snapshot files (default: 'snapshots')"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without moving files"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"organize_fusion_files.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Organize files
    organize_files(
        input_folder=args.input_folder,
        snapshot_folder_name=args.snapshot_folder_name,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
