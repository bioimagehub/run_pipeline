import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import bioimage_pipeline_utils as rp


logger = logging.getLogger(__name__)


def _ensure_unique_path(path: str) -> str:
    """Return a unique path by appending _N if the file already exists."""
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)
    counter = 1
    candidate = f"{base_name}_{counter}{ext}"
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base_name}_{counter}{ext}"
    return candidate


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    collapse_delimiter: str = "__",
    dry_run: bool = False,
) -> None:
    """Copy files matching a search pattern into a single output folder."""

    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error("No files found matching pattern: %s", input_pattern)
        return

    # Determine base folder (mirrors convert_to_tif.py logic)
    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)

    # Default output folder
    if output_folder is None:
        output_folder = base_folder + "_copied"

    os.makedirs(output_folder, exist_ok=True)
    logger.info("Copying %d file(s) to %s", len(files), output_folder)

    # Prepare source/destination pairs
    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        dst = os.path.join(output_folder, collapsed)
        dst = _ensure_unique_path(dst)
        file_pairs.append((src, dst))

    if dry_run:
        print(f"[DRY RUN] Would copy {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    # Copy files
    for src, dst in file_pairs:
        shutil.copy2(src, dst)
        logger.debug("Copied %s -> %s", src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy files matching a search pattern into a single folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Copy masks into one folder
  environment: uv@3.11:copy-files
  commands:
  - python
  - '%REPO%/standard_code/python/copy_files_with_extension.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/merged_masks'
  - --collapse-delimiter: '__'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: base_folder + '_copied')",
    )
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter used when flattening subfolder structure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy actions without executing",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"copy_files_with_extension.py version: {version}")
        return

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        collapse_delimiter=args.collapse_delimiter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

