"""
Fill missing timepoints in binary masks (TCZYX-aware).

For frames with empty masks:
- Leading empty frames copy the first non-empty mask.
- Trailing empty frames copy the last non-empty mask.
- Middle gaps are interpolated between the nearest non-empty frames.

Designed for use with run_pipeline.exe YAML configs.

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np

import bioimage_pipeline_utils as rp


logger = logging.getLogger(__name__)


def _parse_channels(channel_str: str | None) -> list[int] | None:
    """
    Parse channels from string format like '0 2', '0,2', or None.
    Handles both space-separated and comma-separated formats.
    """
    if channel_str is None:
        return None
    parts = str(channel_str).replace(',', ' ').split()
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse channels: {e}")


def _interpolate_gap(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    n_steps: int,
    blend_threshold: float,
    foreground_value: int | bool,
) -> list[np.ndarray]:
    """Return interpolated masks between two binary masks."""
    before_bin = before_mask > 0
    after_bin = after_mask > 0

    if not np.any(before_bin) or not np.any(after_bin):
        return [np.zeros_like(before_mask) for _ in range(n_steps)]

    interpolated = []
    for step in range(n_steps):
        alpha = (step + 1) / (n_steps + 1)
        blended = before_bin.astype(float) * (1 - alpha) + after_bin.astype(float) * alpha
        interp_bin = blended > blend_threshold
        result = np.zeros_like(before_mask)
        result[interp_bin] = foreground_value
        interpolated.append(result)

    return interpolated


def _fill_channel_timepoints(
    data: np.ndarray,
    max_gap_size: int | None,
    blend_threshold: float,
) -> tuple[np.ndarray, int, int, int]:
    """
    Fill missing timepoints for a single channel.

    Args:
        data: 4D TZYX array
        max_gap_size: Maximum gap size to fill (None = no limit)
        blend_threshold: Threshold for interpolation blending

    Returns:
        (filled_data, filled_gaps, skipped_gaps, copied_frames)
    """
    filled = data.copy()
    T = filled.shape[0]

    nonempty = np.any(filled > 0, axis=(1, 2, 3))
    if not np.any(nonempty):
        logger.warning("All timepoints are empty; nothing to fill")
        return filled, 0, 0, 0

    first_nonempty = int(np.argmax(nonempty))
    last_nonempty = int(T - 1 - np.argmax(nonempty[::-1]))

    filled_gaps = 0
    skipped_gaps = 0
    copied_frames = 0

    # Leading empty frames
    if first_nonempty > 0:
        logger.info(f"Leading empty frames: 0..{first_nonempty - 1}, copying T={first_nonempty}")
        for t in range(0, first_nonempty):
            filled[t] = filled[first_nonempty]
            copied_frames += 1

    # Trailing empty frames
    if last_nonempty < T - 1:
        logger.info(f"Trailing empty frames: {last_nonempty + 1}..{T - 1}, copying T={last_nonempty}")
        for t in range(last_nonempty + 1, T):
            filled[t] = filled[last_nonempty]
            copied_frames += 1

    # Middle gaps
    t = first_nonempty + 1
    while t < last_nonempty:
        if nonempty[t]:
            t += 1
            continue

        gap_start = t
        while t < last_nonempty and not nonempty[t]:
            t += 1
        gap_end = t - 1

        gap_size = gap_end - gap_start + 1
        if max_gap_size is not None and gap_size > max_gap_size:
            logger.warning(
                f"Gap T={gap_start}-{gap_end} (size {gap_size}) exceeds max ({max_gap_size}), skipping"
            )
            skipped_gaps += 1
            continue

        before_t = gap_start - 1
        after_t = gap_end + 1

        before_mask = filled[before_t]
        after_mask = filled[after_t]

        if filled.dtype == bool:
            foreground_value = True
        else:
            foreground_value = int(max(np.max(before_mask), np.max(after_mask), 1))

        interpolated = _interpolate_gap(
            before_mask=before_mask,
            after_mask=after_mask,
            n_steps=gap_size,
            blend_threshold=blend_threshold,
            foreground_value=foreground_value,
        )

        for i, interp_mask in enumerate(interpolated):
            filled[gap_start + i] = interp_mask

        filled_gaps += 1

    return filled, filled_gaps, skipped_gaps, copied_frames


def process_single_file(
    input_path: str,
    output_path: str,
    channels: list[int] | None,
    max_gap_size: int | None,
    blend_threshold: float,
    force: bool,
) -> bool:
    """Load one mask, fill missing timepoints, and save."""
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")

        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True

        img = rp.load_tczyx_image(input_path)
        data = img.data.copy()
        T, C, Z, Y, X = data.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}, dtype={data.dtype}")

        if channels is None:
            channels_to_process = list(range(C))
        else:
            channels_to_process = [c for c in channels if 0 <= c < C]
            if not channels_to_process:
                raise ValueError(f"No valid channels found. Image has {C} channels, requested {channels}")

        logger.info(f"Channels to process: {channels_to_process}")

        for c in channels_to_process:
            filled_c, filled_gaps, skipped_gaps, copied_frames = _fill_channel_timepoints(
                data[:, c, :, :, :],
                max_gap_size=max_gap_size,
                blend_threshold=blend_threshold,
            )
            data[:, c, :, :, :] = filled_c
            logger.info(
                f"Channel {c}: filled_gaps={filled_gaps}, skipped_gaps={skipped_gaps}, copied_frames={copied_frames}"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(data, output_path)
        logger.info(f"Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    input_pattern: str,
    output_folder: str | None,
    channels: list[int] | None,
    max_gap_size: int | None,
    blend_threshold: float,
    no_parallel: bool,
    dry_run: bool,
    force: bool,
    suffix: str,
) -> None:
    """Process many files matching the search pattern."""
    search_subfolders = "**" in input_pattern
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not input_files:
        logger.error(f"No files matched pattern: {input_pattern}")
        return

    logger.info(f"Found {len(input_files)} files")

    if output_folder is None:
        base_dir = os.path.dirname(input_pattern.replace("**/", "").replace("*", ""))
        output_folder = (base_dir or ".") + "_filled_timepoints"

    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    for input_path in input_files:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{basename}{suffix}.tif"
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path))

    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Max gap size: {max_gap_size}")
        print(f"[DRY RUN] Blend threshold: {blend_threshold}")
        if channels is None:
            print("[DRY RUN] Channels: all")
        else:
            print(f"[DRY RUN] Channels: {channels}")
        for inp, out in tasks[:5]:
            print(f"[DRY RUN]   {os.path.basename(inp)} -> {os.path.basename(out)}")
        if len(tasks) > 5:
            print(f"[DRY RUN]   ... and {len(tasks) - 5} more files")
        return

    if no_parallel or len(tasks) == 1:
        logger.info("Processing files sequentially")
        ok = 0
        for inp, out in tasks:
            if process_single_file(
                inp, out, channels, max_gap_size, blend_threshold, force
            ):
                ok += 1
        logger.info(f"Done: {ok} succeeded, {len(tasks) - ok} failed")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    cpu_count = os.cpu_count() or 1
    max_workers = max(cpu_count - 1, 1)
    logger.info(f"Processing files in parallel (workers={max_workers})")

    ok = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_file,
                inp, out, channels, max_gap_size, blend_threshold, force,
            )
            for inp, out in tasks
        ]
        for f in as_completed(futures):
            try:
                if f.result():
                    ok += 1
            except Exception as e:  # pragma: no cover
                logger.error(f"Task failed: {e}")

    logger.info(f"Done: {ok} succeeded, {len(tasks) - ok} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing timepoints in binary masks (TCZYX-aware).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fill missing timepoints (all channels)
  environment: uv@3.11:fill-mask-timepoints
  commands:
  - python
  - '%REPO%/standard_code/python/fill_mask_timepoints.py'
  - --input-search-pattern: '%YAML%/input_masks/**/*.tif'
  - --output-folder: '%YAML%/output_filled'
  - --blend-threshold: 0.3

- name: Fill missing timepoints (channels 0 and 2, max gap 5)
  environment: uv@3.11:fill-mask-timepoints
  commands:
  - python
  - '%REPO%/standard_code/python/fill_mask_timepoints.py'
  - --input-search-pattern: '%YAML%/input_masks/**/*.tif'
  - --output-folder: '%YAML%/output_filled'
  - --channels: '0 2'
  - --max-gap-size: 5
  - --blend-threshold: 0.4
  - --no-parallel
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards; use '**' to recurse)",
    )
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: <input_root>_filled_timepoints)",
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=None,
        help=(
            "Channel indices to process (0-based). Space or comma separated. "
            "Examples: --channels '0 2' or --channels '0,2'"
        ),
    )
    parser.add_argument(
        "--max-gap-size",
        type=int,
        default=None,
        help="Maximum gap size (in frames) to fill. Default: no limit",
    )
    parser.add_argument(
        "--blend-threshold",
        type=float,
        default=0.3,
        help="Threshold for interpolated blending (0-1). Default: 0.3",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_filled_timepoints",
        help="Suffix to add to output filenames",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if outputs exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned actions without executing",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args()

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"fill_mask_timepoints.py version: {version}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        channels=args.channels,
        max_gap_size=args.max_gap_size,
        blend_threshold=args.blend_threshold,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
