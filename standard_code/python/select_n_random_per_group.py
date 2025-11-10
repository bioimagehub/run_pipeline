"""
Select random samples from grouped files for training set creation.
Groups files by pattern, selects N random samples per group, and copies to output folder.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import shutil
import random
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
from tqdm import tqdm

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def group_files_by_pattern(
    files: List[str],
    delimiter: str = "__",
    num_segments: int = 2
) -> Dict[str, List[str]]:
    """
    Group files based on filename pattern segments.
    
    Args:
        files: List of file paths to group
        delimiter: Delimiter to split filenames by
        num_segments: Number of segments to use for grouping (from start)
    
    Returns:
        Dictionary mapping group keys to lists of file paths
    """
    groups = defaultdict(list)
    
    for file_path in files:
        # Get just the filename without path
        filename = os.path.basename(file_path)
        
        # Split by delimiter
        segments = filename.split(delimiter)
        
        # Take first num_segments to create group key
        if len(segments) >= num_segments:
            group_key = delimiter.join(segments[:num_segments])
        else:
            # If not enough segments, use full filename (minus extension)
            group_key = os.path.splitext(filename)[0]
        
        groups[group_key].append(file_path)
    
    return dict(groups)


def select_random_from_groups(
    groups: Dict[str, List[str]],
    n_per_group: int,
    seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Select N random files from each group.
    
    Args:
        groups: Dictionary of grouped files
        n_per_group: Number of files to select per group
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of selected files per group
    """
    if seed is not None:
        random.seed(seed)
    
    selected = {}
    
    for group_key, file_list in groups.items():
        # Sort for reproducibility when seed is set
        file_list_sorted = sorted(file_list)
        
        # Select min(n_per_group, available files)
        n_to_select = min(n_per_group, len(file_list_sorted))
        selected[group_key] = random.sample(file_list_sorted, n_to_select)
        
        logger.warning(f"Group '{group_key}': Selected {n_to_select}/{len(file_list_sorted)} files")
    
    return selected


def copy_selected_files(
    selected_groups: Dict[str, List[str]],
    output_folder: str,
    preserve_structure: bool = False,
    base_folder: Optional[str] = None
) -> None:
    """
    Copy selected files to output folder.
    
    Args:
        selected_groups: Dictionary of selected files per group
        output_folder: Destination folder
        preserve_structure: If True, preserve subfolder structure
        base_folder: Base folder for preserving structure (required if preserve_structure=True)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    total_copied = 0
    
    # Count total files to copy
    total_files = sum(len(file_list) for file_list in selected_groups.values())
    
    # Add progress bar for file copying
    with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
        for group_key, file_list in selected_groups.items():
            for src_path in file_list:
                if preserve_structure and base_folder:
                    # Preserve relative path structure
                    rel_path = os.path.relpath(src_path, base_folder)
                    dst_path = os.path.join(output_folder, rel_path)
                else:
                    # Flat structure - just copy to output folder
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(output_folder, filename)
                
                # Create destination directory if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
                pbar.set_postfix_str(f"{os.path.basename(src_path)}")
                total_copied += 1
                pbar.update(1)
    
    logger.warning(f"Total files copied: {total_copied}")


def process_files(
    input_pattern: str,
    output_folder: str,
    n_per_group: int = 5,
    group_delimiter: str = "__",
    group_segments: int = 2,
    preserve_structure: bool = False,
    random_seed: Optional[int] = None,
    dry_run: bool = False
) -> None:
    """
    Process files: find, group, select random samples, and copy.
    
    Args:
        input_pattern: File search pattern (supports ** for recursive)
        output_folder: Output directory for selected files
        n_per_group: Number of random files to select per group
        group_delimiter: Delimiter to split filenames by (default: '__')
        group_segments: Number of segments to use for grouping (default: 2)
        preserve_structure: Preserve subfolder structure in output
        random_seed: Random seed for reproducibility
        dry_run: Only print planned actions without executing
    """
    # Find files
    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.warning(f"Found {len(files)} file(s) to process")
    
    # Determine base folder (for structure preservation)
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)
    
    logger.warning(f"Base folder: {base_folder}")
    
    # Group files by pattern
    logger.warning(f"Grouping files by first {group_segments} segments (delimiter: '{group_delimiter}')")
    groups = group_files_by_pattern(files, group_delimiter, group_segments)
    
    logger.warning(f"Found {len(groups)} group(s):")
    for group_key, file_list in sorted(groups.items()):
        logger.warning(f"  {group_key}: {len(file_list)} files")
    
    # Select random samples from each group
    logger.warning(f"Selecting {n_per_group} random file(s) per group (seed: {random_seed})")
    selected = select_random_from_groups(groups, n_per_group, random_seed)
    
    # Calculate total selections
    total_selected = sum(len(files) for files in selected.values())
    logger.warning(f"Total files selected: {total_selected}")
    
    # Dry run - just print plans
    if dry_run:
        print(f"\n[DRY RUN] Would copy {total_selected} files to: {output_folder}")
        print(f"[DRY RUN] Preserve structure: {preserve_structure}")
        print(f"[DRY RUN] Random seed: {random_seed}")
        print("\n[DRY RUN] Selected files by group:")
        for group_key, file_list in sorted(selected.items()):
            print(f"\n  Group: {group_key} ({len(file_list)} files)")
            for file_path in sorted(file_list):
                print(f"    - {os.path.basename(file_path)}")
        return
    
    # Copy selected files
    logger.warning(f"Copying files to: {output_folder}")
    copy_selected_files(
        selected,
        output_folder,
        preserve_structure=preserve_structure,
        base_folder=base_folder if preserve_structure else None
    )
    
    logger.warning("Selection complete!")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Select N random samples per group for training set creation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Select random samples for training (5 per group)
  environment: uv@3.11:basic
  commands:
  - python
  - '%REPO%/standard_code/python/select_n_random_per_group.py'
  - --input-search-pattern: '%YAML%/bleach_corrected/**/*.tif'
  - --output-folder: '%YAML%/training_data'
  - --n-per-group: 5
  - --group-delimiter: __
  - --group-segments: 2

- name: Select random samples (reproducible with seed)
  environment: uv@3.11:basic
  commands:
  - python
  - '%REPO%/standard_code/python/select_n_random_per_group.py'
  - --input-search-pattern: '%YAML%/bleach_corrected/**/*.tif'
  - --output-folder: '%YAML%/training_data'
  - --n-per-group: 10
  - --random-seed: 42
  - --preserve-structure

- name: Preview selection (dry run)
  environment: uv@3.11:basic
  commands:
  - python
  - '%REPO%/standard_code/python/select_n_random_per_group.py'
  - --input-search-pattern: '%YAML%/bleach_corrected/**/*.tif'
  - --output-folder: '%YAML%/training_data'
  - --n-per-group: 5
  - --dry-run

File Grouping Example:
  Given files like:
    SP20250625__3SA__R2__SP20250625_PC_R2_3SA_001_bleach_corrected.tif
    SP20250625__3SA__R2__SP20250625_PC_R2_3SA_002_bleach_corrected.tif
    SP20250625__4SB__R1__SP20250625_PC_R1_4SB_001_bleach_corrected.tif
    
  With --group-delimiter "__" and --group-segments 2:
    - Group 1: "SP20250625__3SA" (all files starting with these segments)
    - Group 2: "SP20250625__4SB" (all files starting with these segments)
  
  Then --n-per-group 5 will select 5 random files from each group.
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for selected files"
    )
    
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=5,
        help="Number of random files to select per group (default: 5)"
    )
    
    parser.add_argument(
        "--group-delimiter",
        type=str,
        default="__",
        help="Delimiter to split filenames by for grouping (default: '__')"
    )
    
    parser.add_argument(
        "--group-segments",
        type=int,
        default=2,
        help="Number of segments to use for grouping from start (default: 2)"
    )
    
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve subfolder structure in output folder"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection (default: None)"
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
    
    args = parser.parse_args()
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"select_n_random_per_group.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        n_per_group=args.n_per_group,
        group_delimiter=args.group_delimiter,
        group_segments=args.group_segments,
        preserve_structure=args.preserve_structure,
        random_seed=args.random_seed,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
