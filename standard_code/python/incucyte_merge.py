"""
Merge Incucyte archive channel splits into multi-channel OME-TIFF.

Incucyte archives split images by channel (e.g., B2-1-C1.tif, B2-1-Ph.tif).
This script groups channel splits by well/image ID and merges them into a single
multi-channel OME-TIFF per well-image.

Lenient by default: merges whatever channels exist, logs missing channels when
an expected list is provided, and keeps going unless --strict is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from bioio_base.types import PhysicalPixelSizes

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)

# Incucyte objective specifications from user manual
OBJECTIVE_SPECS = {
    "4x": {"um_per_pixel": 2.82, "position": "Slot C", "image_size": (1536, 1152)},
    "10x": {"um_per_pixel": 1.24, "position": "Slot B", "image_size": (1408, 1040)},
    "20x": {"um_per_pixel": 0.62, "position": "Slot A", "image_size": (1408, 1040)},
}


CHANNEL_REGEX = re.compile(r"^(?P<base>.+)-(?P<channel>[^-]+)$")


def safe_dtype_convert(
    data: np.ndarray,
    target_dtype: np.dtype,
    source_name: str = "source"
) -> np.ndarray:
    """Safely convert array to target dtype with value clamping.
    
    If values exceed the range of target_dtype, clamp them to max/min
    and log a warning.
    
    Args:
        data: Array to convert
        target_dtype: Target numpy dtype
        source_name: Name for warning messages (e.g., "mask", "seed")
    
    Returns:
        Data in target dtype, with values clamped to valid range
    """
    if data.dtype == target_dtype:
        return data
    
    # For integer dtypes, check value ranges
    if np.issubdtype(target_dtype, np.integer) and np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(target_dtype)
        min_val, max_val = info.min, info.max
        
        data_min, data_max = data.min(), data.max()
        
        if data_min < min_val or data_max > max_val:
            logger.warning(
                "%s dtype conversion: values range [%d, %d] exceed target dtype %s range [%d, %d]. "
                "Clamping to valid range.",
                source_name,
                int(data_min), int(data_max),
                target_dtype,
                int(min_val), int(max_val),
            )
    
    # Clamp and convert
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        converted = np.clip(data, info.min, info.max).astype(target_dtype)
    else:
        converted = data.astype(target_dtype)
    
    return converted


@dataclass(frozen=True)
class ChannelFile:
    base_id: str
    channel: str
    path: str
    time_key: str


def extract_time_key_from_scandata(path: str) -> str:
    parts = Path(path).parts
    lower_parts = [part.lower() for part in parts]
    if "scandata" in lower_parts:
        scan_index = lower_parts.index("scandata")
        time_parts = parts[scan_index + 1:-2]
        if time_parts:
            return "/".join(time_parts)
    return "unknown"


def extract_time_key_from_sidecar(path: str) -> str:
    parts = Path(path).parts
    lower_parts = [part.lower() for part in parts]
    if "jobs" in lower_parts:
        jobs_index = lower_parts.index("jobs")
        if len(parts) > jobs_index + 2:
            return normalize_sidecar_time_key(parts[jobs_index + 2])
    return normalize_sidecar_time_key(Path(path).parent.name)


def normalize_sidecar_time_key(time_key: str) -> str:
    """Normalize sidecar time key to match ScanData time key format.

    Example: 26y01m30d11h44 -> 2601/30/1144
    """
    match = re.match(r"^(?P<yy>\d{2})y(?P<mm>\d{2})m(?P<dd>\d{2})d(?P<hh>\d{2})h(?P<min>\d{2})$", time_key)
    if not match:
        return time_key

    yy = match.group("yy")
    mm = match.group("mm")
    dd = match.group("dd")
    hh = match.group("hh")
    minute = match.group("min")
    return f"{yy}{mm}/{dd}/{hh}{minute}"


def process_group_worker(
    args: Tuple[
        str,
        List[ChannelFile],
        List[str],
        Set[str],
        str,
        str,
        List[str],
        List[str],
        List[str],
        bool,
        bool,
        bool,
        Optional[PhysicalPixelSizes],
        bool,
        str,
    ]
) -> Tuple[str, bool, str]:
    (
        base_id,
        channels,
        search_roots,
        primary_paths,
        output_folder,
        output_suffix,
        channel_order,
        expected_channels,
        acq_expected_channels,
        strict,
        dry_run,
        embed_mask_channels,
        physical_pixel_sizes,
        quiet_parallel,
        log_level,
    ) = args

    worker_log_level = "ERROR" if quiet_parallel else log_level
    logging.basicConfig(
        level=getattr(logging, worker_log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    sidecar_groups = find_sidecar_files(
        base_id=base_id,
        search_roots=search_roots,
        primary_paths=primary_paths,
    )

    ok, output_path = process_group(
        base_id=base_id,
        channels=channels,
        output_folder=output_folder,
        output_suffix=output_suffix,
        channel_order=channel_order,
        expected_channels=expected_channels,
        acq_expected_channels=acq_expected_channels,
        sidecar_groups=sidecar_groups,
        strict=strict,
        dry_run=dry_run,
        embed_mask_channels=embed_mask_channels,
        override_physical_sizes=physical_pixel_sizes,
    )

    return base_id, ok, output_path


def render_progress(completed: int, total: int) -> None:
    if total <= 0:
        return
    percent = int((completed / total) * 100)
    bar_width = 30
    filled = int(bar_width * completed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stderr.write(f"\r[{bar}] {completed}/{total} ({percent}%)")
    sys.stderr.flush()


def parse_channel_file(path: str) -> Optional[ChannelFile]:
    """Parse channel token from filename.

    Expected pattern: <base>-<channel>.tif
    Example: B2-1-C1.tif -> base_id=B2-1, channel=C1
    """
    stem = Path(path).stem
    match = CHANNEL_REGEX.match(stem)
    if not match:
        return None
    return ChannelFile(
        base_id=match.group("base"),
        channel=match.group("channel"),
        path=path,
        time_key=extract_time_key_from_scandata(path),
    )


def list_input_files(search_pattern: str) -> List[str]:
    files = sorted(glob(search_pattern, recursive=True))
    return [f.replace("\\", "/") for f in files if f.lower().endswith(".tif")]


def find_essen_roots(input_files: List[str]) -> List[str]:
    roots: Set[str] = set()
    for file_path in input_files:
        for parent in Path(file_path).parents:
            if parent.name.lower() == "essenfiles":
                roots.add(str(parent))
                break
        else:
            roots.add(str(Path(file_path).parent))
    return sorted(roots)


def group_by_base(files: Iterable[str]) -> Dict[str, List[ChannelFile]]:
    grouped: Dict[str, List[ChannelFile]] = {}
    for file_path in files:
        parsed = parse_channel_file(file_path)
        if not parsed:
            logger.warning("Skipping unmatched filename: %s", file_path)
            continue
        grouped.setdefault(parsed.base_id, []).append(parsed)
    return grouped


def parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def find_acq_xml_for_path(file_path: str) -> Optional[str]:
    for parent in Path(file_path).parents:
        candidate = parent / "Acq.xml"
        if candidate.exists():
            return str(candidate)
    return None


def parse_acq_expected_channels(acq_path: str) -> List[str]:
    try:
        tree = ET.parse(acq_path)
    except (ET.ParseError, OSError) as exc:
        logger.warning("Failed to parse Acq.xml at %s: %s", acq_path, exc)
        return []

    root = tree.getroot()
    expected: List[str] = []

    for acq_chan in root.findall(".//AcqChan"):
        index_node = acq_chan.find("Index")
        if index_node is None:
            continue
        index_text = (index_node.text or "").strip()
        if not index_text.isdigit():
            continue
        channel_index = int(index_text)

        has_signal = False
        for time_node in acq_chan.findall(".//ExposureTimes/Time"):
            try:
                if float(time_node.text or 0) > 0:
                    has_signal = True
                    break
            except ValueError:
                continue

        if has_signal:
            expected.append(f"C{channel_index}")

    return expected


def extract_sidecar_suffix(base_id: str, filename_stem: str) -> Optional[str]:
    prefix = f"{base_id}-"
    if not filename_stem.startswith(prefix):
        return None
    suffix = filename_stem[len(prefix):]
    if not suffix:
        return None
    return suffix


def find_sidecar_files(
    base_id: str,
    search_roots: List[str],
    primary_paths: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    
    # Pattern to match channel-like suffixes (C1, C2, Ph, GFP, etc.)
    channel_pattern = re.compile(r"^(C\d+|Ph|GFP|RFP|mCherry|YFP|CFP|DAPI|Seed)$", re.IGNORECASE)

    for root in search_roots:
        pattern = os.path.join(root, "**", f"{base_id}-*.tif")
        for path in glob(pattern, recursive=True):
            normalized = path.replace("\\", "/")
            if normalized in primary_paths:
                continue

            parts_lower = [p.lower() for p in Path(normalized).parts]
            if "scandata" in parts_lower:
                continue

            stem = Path(normalized).stem
            suffix = extract_sidecar_suffix(base_id, stem)
            if not suffix:
                continue
            
            # Skip files that look like channel splits (C1, C2, Ph, etc.)
            if channel_pattern.match(suffix):
                continue
            
            time_key = extract_time_key_from_sidecar(normalized)
            grouped.setdefault(suffix, {}).setdefault(time_key, []).append(normalized)

    for suffix in grouped:
        for time_key in grouped[suffix]:
            grouped[suffix][time_key] = sorted(grouped[suffix][time_key])
    return grouped


def embed_sidecars_into_channels(
    sidecar_groups: Dict[str, Dict[str, List[str]]],
    main_channel_time_keys: List[str],
    ref_shape: Tuple[int, int, int, int, int],
    ref_dtype: np.dtype,
) -> List[Tuple[str, np.ndarray]]:
    """Embed sidecar files as additional channels into the main image.
    
    For each sidecar group (suffix like "mask", "seed"):
    - Check XY match with reference
    - Pad Z if needed
    - Fill missing timepoints with zeros
    - Return list of (suffix_name, TCZYX_array) tuples
    
    Returns empty list if no sidecars can be embedded.
    """
    embedded: List[Tuple[str, np.ndarray]] = []
    
    for suffix in sorted(sidecar_groups.keys()):
        time_to_files = sidecar_groups[suffix]
        sidecar_time_keys = sorted(time_to_files.keys())
        
        if not sidecar_time_keys:
            logger.warning("Sidecar group '%s' has no files, skipping", suffix)
            continue
        
        # Get first file to check dimensions
        first_files = time_to_files[sidecar_time_keys[0]]
        if not first_files:
            continue
            
        first_img = rp.load_tczyx_image(first_files[0])
        first_shape = first_img.data.shape  # TCZYX
        
        # Check XY dimensions (must match)
        ref_Y, ref_X = ref_shape[3], ref_shape[4]
        sidecar_Y, sidecar_X = first_shape[3], first_shape[4]
        
        if (sidecar_Y, sidecar_X) != (ref_Y, ref_X):
            logger.warning(
                "Sidecar '%s': XY mismatch (reference: %dx%d, sidecar: %dx%d). "
                "Cannot embed as channel, will keep as separate file.",
                suffix,
                ref_X, ref_Y,
                sidecar_X, sidecar_Y,
            )
            continue
        
        # XY matches, proceed with embedding
        sidecar_dtype = first_img.data.dtype
        stacks = []
        
        for main_time_key in main_channel_time_keys:
            # Find matching sidecar timepoint
            files = time_to_files.get(main_time_key, [])
            
            if not files:
                # Missing timepoint: fill with zeros in ref_dtype
                # Shape: T=1, C=1 (default), Z=ref_Z, Y=ref_Y, X=ref_X
                logger.debug(
                    "Sidecar '%s' timepoint '%s' not found; creating empty mask channel(s)",
                    suffix,
                    main_time_key,
                )
                empty = np.zeros(
                    (1, 1, ref_shape[2], ref_shape[3], ref_shape[4]),
                    dtype=ref_dtype
                )
                stacks.append(empty)
                logger.debug(
                    "Sidecar '%s' timepoint '%s': missing, filling with zeros",
                    suffix, main_time_key
                )
                continue
            
            # Load and stack all channels from this timepoint
            loaded = []
            for file_path in files:
                img = rp.load_tczyx_image(file_path)
                data = img.data  # TCZYX
                
                ref_Z = ref_shape[2]
                sidecar_Z = data.shape[2]
                
                # Pad Z if needed
                if sidecar_Z != ref_Z:
                    if sidecar_Z > ref_Z:
                        logger.warning(
                            "Sidecar '%s' has more Z slices (%d) than reference (%d), "
                            "cropping to match",
                            suffix, sidecar_Z, ref_Z
                        )
                        data = data[:, :, :ref_Z, :, :]
                    else:
                        pad_amount = ref_Z - sidecar_Z
                        logger.debug(
                            "Sidecar '%s' timepoint '%s': padding Z by %d slices",
                            suffix, main_time_key, pad_amount
                        )
                        # Pad on the end (after last Z slice)
                        data = np.pad(
                            data,
                            ((0, 0), (0, 0), (0, pad_amount), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )
                
                # Safely convert sidecar dtype to match reference dtype before concatenation
                if data.dtype != ref_dtype:
                    data = safe_dtype_convert(data, ref_dtype, source_name=f"sidecar '{suffix}'")
                
                loaded.append(data)
            
            # Concatenate channels from this timepoint
            stacked_channels = np.concatenate(loaded, axis=1)  # Merge along C
            stacks.append(stacked_channels)
        
        # Concatenate all timepoints
        merged = np.concatenate(stacks, axis=0)  # TCZYX
        embedded.append((suffix, merged))
    
    return embedded


def merge_sidecar_stack(
    sidecar_groups: Dict[str, List[str]],
    output_path: str,
    time_keys: List[str],
    ref_shape: Tuple[int, int, int, int, int],
    ref_dtype: np.dtype,
    ref_physical_sizes: Optional[PhysicalPixelSizes],
) -> bool:
    if not sidecar_groups:
        return False

    # Use actual time keys from sidecar files, not from main channels
    sidecar_time_keys = sorted(sidecar_groups.keys())
    if not sidecar_time_keys:
        logger.warning("No sidecar files found for %s, skipping", output_path)
        return False

    stacks = []
    sidecar_dtype = None
    sidecar_physical_sizes = None

    for time_key in sidecar_time_keys:
        files = sidecar_groups.get(time_key, [])
        if not files:
            if sidecar_dtype is None:
                # Will be set on first real file
                continue
            logger.debug(
                "Sidecar stack timepoint '%s' not found; creating empty mask channel(s) in %s",
                time_key,
                output_path,
            )
            empty = np.zeros((1, 1, ref_shape[2], ref_shape[3], ref_shape[4]), dtype=sidecar_dtype)
            stacks.append(empty)
            continue

        loaded = []
        for file_path in files:
            img = rp.load_tczyx_image(file_path)
            data = img.data
            
            # Preserve original dtype from sidecar files (e.g., uint32 for labels)
            if sidecar_dtype is None:
                sidecar_dtype = data.dtype
                sidecar_physical_sizes = img.physical_pixel_sizes
            
            # Verify spatial dimensions match (ZYX)
            if data.shape[2:] != ref_shape[2:]:
                logger.error(
                    "Sidecar shape mismatch for %s: expected ZYX=%s, got %s (full shape: %s)",
                    file_path,
                    ref_shape[2:],
                    data.shape[2:],
                    data.shape,
                )
                return False
            loaded.append(data)

        stacked = np.concatenate(loaded, axis=1)
        stacks.append(stacked)

    if not stacks:
        logger.warning("No sidecar files found for %s, skipping", output_path)
        return False

    merged = np.concatenate(stacks, axis=0)  # TCZYX
    rp.save_tczyx_image(merged, output_path, physical_pixel_sizes=sidecar_physical_sizes or ref_physical_sizes)
    return True


def channel_sort_key(channel: str) -> Tuple[int, int, str]:
    match = re.match(r"^C(\d+)$", channel, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), "")
    return (1, 0, channel.lower())


def order_channels(
    channels: List[ChannelFile],
    channel_order: List[str]
) -> List[ChannelFile]:
    if not channel_order:
        return sorted(channels, key=lambda c: channel_sort_key(c.channel))

    order_map = {name: idx for idx, name in enumerate(channel_order)}
    return sorted(
        channels,
        key=lambda c: (
            order_map.get(c.channel, len(order_map)),
            channel_sort_key(c.channel)[1],
            c.channel.lower(),
        ),
    )


def merge_channels(
    channels: List[ChannelFile],
    ordered_channels: List[str],
    output_path: str,
    strict: bool,
    override_physical_sizes: Optional[PhysicalPixelSizes] = None,
) -> Tuple[bool, Optional[Tuple[int, int, int, int, int]], Optional[np.dtype], Optional[PhysicalPixelSizes]]:
    if not channels:
        return False, None, None, None

    time_map: Dict[str, Dict[str, ChannelFile]] = {}
    channel_map: Dict[str, ChannelFile] = {}
    for channel_file in channels:
        time_map.setdefault(channel_file.time_key, {})[channel_file.channel] = channel_file
        channel_map.setdefault(channel_file.channel, channel_file)

    time_keys = sorted(time_map.keys())
    if not ordered_channels:
        ordered_channels = order_channels(list(channel_map.values()), [])
        ordered_channels = [c.channel for c in ordered_channels]

    reference_path = next(iter(channel_map.values())).path
    reference_img = rp.load_tczyx_image(reference_path)
    ref_shape = reference_img.data.shape
    ref_dtype = reference_img.data.dtype
    physical_sizes = override_physical_sizes or reference_img.physical_pixel_sizes

    stacks = []

    for time_key in time_keys:
        time_channels = time_map[time_key]
        time_stack = []
        for channel in ordered_channels:
            channel_file = time_channels.get(channel)
            if channel_file is None:
                empty = np.zeros((1, 1, ref_shape[2], ref_shape[3], ref_shape[4]), dtype=ref_dtype)
                time_stack.append(empty)
                continue

            img = rp.load_tczyx_image(channel_file.path)
            data = img.data
            if data.shape[2:] != ref_shape[2:]:
                logger.error(
                    "Shape mismatch for %s: expected %s, got %s",
                    channel_file.path,
                    ref_shape,
                    data.shape,
                )
                return False, None, None, None
            time_stack.append(data)

        stacked_channels = np.concatenate(time_stack, axis=1)
        stacks.append(stacked_channels)

    merged = np.concatenate(stacks, axis=0)  # TCZYX
    rp.save_tczyx_image(merged, output_path, physical_pixel_sizes=physical_sizes)
    return True, ref_shape, ref_dtype, physical_sizes


def process_group(
    base_id: str,
    channels: List[ChannelFile],
    output_folder: str,
    output_suffix: str,
    channel_order: List[str],
    expected_channels: List[str],
    acq_expected_channels: List[str],
    sidecar_groups: Dict[str, Dict[str, List[str]]],
    strict: bool,
    dry_run: bool,
    embed_mask_channels: bool = False,
    override_physical_sizes: Optional[PhysicalPixelSizes] = None,
) -> Tuple[bool, str]:
    channel_index: Dict[str, ChannelFile] = {}
    for channel_file in channels:
        channel_index.setdefault(channel_file.channel, channel_file)
    ordered = order_channels(list(channel_index.values()), channel_order)
    
    # Filter to only include channels in channel_order if specified
    if channel_order:
        ordered = [c for c in ordered if c.channel in channel_order]
    
    present = [c.channel for c in ordered]

    time_keys = sorted({c.time_key for c in channels})

    if expected_channels:
        missing = [c for c in expected_channels if c not in present]
        if missing:
            message = f"{base_id}: missing channels {missing}"
            if strict:
                logger.error(message)
                return False, message
            logger.warning(message)

    if acq_expected_channels:
        missing_acq = [c for c in acq_expected_channels if c not in present]
        if missing_acq:
            message = f"{base_id}: missing Acq.xml channels {missing_acq}"
            if strict:
                logger.error(message)
                return False, message
            logger.warning(message)

    output_path = os.path.join(output_folder, f"{base_id}{output_suffix}")

    if dry_run:
        logger.info(
            "[DRY RUN] %s -> %s | channels: %s",
            base_id,
            os.path.basename(output_path),
            ", ".join(present),
        )
        if sidecar_groups:
            for suffix, files in sorted(sidecar_groups.items()):
                file_count = sum(len(group) for group in files.values())
                if embed_mask_channels:
                    logger.info(
                        "[DRY RUN] %s | embed sidecar: %s (%d files)",
                        base_id,
                        suffix,
                        file_count,
                    )
                else:
                    logger.info(
                        "[DRY RUN] %s -> %s | sidecar: %s (%d files)",
                        base_id,
                        f"{base_id}_{suffix}{output_suffix}",
                        suffix,
                        file_count,
                    )
        return True, "dry-run"

    ok, ref_shape, ref_dtype, ref_physical_sizes = merge_channels(
        channels=channels,
        ordered_channels=present,
        output_path=output_path,
        strict=strict,
        override_physical_sizes=override_physical_sizes,
    )
    if not ok:
        return False, output_path

    if ref_shape is None or ref_dtype is None:
        return False, output_path

    # Handle sidecar embedding or separate files
    if embed_mask_channels and sidecar_groups:
        # Embed sidecars into main image
        embedded_sidecars = embed_sidecars_into_channels(
            sidecar_groups=sidecar_groups,
            main_channel_time_keys=time_keys,
            ref_shape=ref_shape,
            ref_dtype=ref_dtype,
        )
        
        if embedded_sidecars:
            # Load the main merged image and append embedded channels
            main_img = rp.load_tczyx_image(output_path)
            main_data = main_img.data  # TCZYX
            
            # Append each embedded sidecar as additional channels
            for suffix, sidecar_data in embedded_sidecars:
                # sidecar_data is TCZYX and already converted to ref_dtype, append along C axis
                # Double-check dtype match before concatenation
                if sidecar_data.dtype != main_data.dtype:
                    sidecar_data = safe_dtype_convert(sidecar_data, main_data.dtype, source_name=f"sidecar '{suffix}'")
                
                main_data = np.concatenate([main_data, sidecar_data], axis=1)
                logger.info(
                    "%s: embedded sidecar '%s' as additional channel(s) (dtype: %s)",
                    base_id, suffix, main_data.dtype
                )
            
            # Re-save the combined image
            rp.save_tczyx_image(main_data, output_path, physical_pixel_sizes=ref_physical_sizes)
            logger.info(
                "%s: saved combined image with embedded sidecars",
                base_id
            )
    else:
        # Original behavior: save sidecars as separate files
        for suffix, files in sorted(sidecar_groups.items()):
            sidecar_output = os.path.join(output_folder, f"{base_id}_{suffix}{output_suffix}")
            if not merge_sidecar_stack(
                files,
                sidecar_output,
                time_keys=time_keys,
                ref_shape=ref_shape,
                ref_dtype=ref_dtype,
                ref_physical_sizes=ref_physical_sizes,
            ):
                if strict:
                    return False, sidecar_output
                logger.warning("Failed to merge sidecar %s for %s", suffix, base_id)

    return True, output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Incucyte archive channel splits into multi-channel OME-TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Merge Incucyte archive channels (lenient)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/incucyte_merge.py'
  - --input-search-pattern: '%YAML%/Incucyte/Archive/**/ScanData/**/96/*.tif'
  - --output-folder: '%YAML%/merged_output'
  - --objective: '20x'

- name: Merge with expected channels and order
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/incucyte_merge.py'
  - --input-search-pattern: '%YAML%/Incucyte/Archive/**/ScanData/**/96/*.tif'
  - --output-folder: '%YAML%/merged_output'
  - --expected-channels: 'C1,C2,Ph,Seed'
  - --channel-order: 'Ph,C1,C2,Seed'
  - --objective: '20x'

- name: Merge with embedded mask channels
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/incucyte_merge.py'
  - --input-search-pattern: '%YAML%/Incucyte/Archive/**/ScanData/**/96/*.tif'
  - --output-folder: '%YAML%/merged_output'
  - --channel-order: 'Ph,C1,C2'
  - --embed-mask-channels
  - --objective: '20x'

- name: Strict mode with 4x objective
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/incucyte_merge.py'
  - --input-search-pattern: '%YAML%/Incucyte/Archive/**/ScanData/**/96/*.tif'
  - --output-folder: '%YAML%/merged_output'
  - --expected-channels: 'C1,C2,Ph,Seed'
  - --objective: '4x'
  - --strict
""",
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help=(
            "Glob pattern for Incucyte channel TIFFs. "
            "Example: '%YAML%/Incucyte/Archive/**/ScanData/**/96/*.tif'"
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for merged OME-TIFF files",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=".ome.tif",
        help="Output file suffix (default: .ome.tif)",
    )
    parser.add_argument(
        "--expected-channels",
        type=str,
        default=None,
        help="Comma-separated list of expected channel tokens (warn if missing)",
    )
    parser.add_argument(
        "--channel-order",
        type=str,
        default=None,
        help="Comma-separated channel ordering; extra channels appended",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a group if any expected channels are missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview grouping and ordering without writing outputs",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["4x", "10x", "20x"],
        default=None,
        help="Incucyte objective (4x, 10x, or 20x). Sets correct pixel size calibration in output metadata.",
    )
    parser.add_argument(
        "--embed-mask-channels",
        action="store_true",
        help="Embed sidecar mask channels into the main image as additional channels instead of separate files",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process groups sequentially)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)",
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

    expected_channels = parse_list(args.expected_channels)
    channel_order = parse_list(args.channel_order)

    # Set up physical pixel sizes if objective specified
    physical_pixel_sizes: Optional[PhysicalPixelSizes] = None
    if args.objective:
        spec = OBJECTIVE_SPECS[args.objective]
        um_per_pixel = spec["um_per_pixel"]
        # Z is not available, Y and X are the same for most objectives
        physical_pixel_sizes = PhysicalPixelSizes(Z=None, Y=um_per_pixel, X=um_per_pixel)
        logger.info(
            "Using %s objective: %.2f um/pixel (Position: %s)",
            args.objective,
            um_per_pixel,
            spec["position"],
        )

    input_files = list_input_files(args.input_search_pattern)
    if not input_files:
        raise ValueError(f"No files found for pattern: {args.input_search_pattern}")

    groups = group_by_base(input_files)
    if not groups:
        raise ValueError("No valid channel files matched the naming pattern.")

    acq_expected_channels: List[str] = []
    acq_paths: Set[str] = set()
    for file_path in input_files[:20]:
        acq_path = find_acq_xml_for_path(file_path)
        if acq_path:
            acq_paths.add(acq_path)
    if not acq_paths:
        logger.warning("Acq.xml not found. Skipping acquisition validation.")
    else:
        acq_path = sorted(acq_paths)[0]
        acq_expected_channels = parse_acq_expected_channels(acq_path)
        if acq_expected_channels:
            logger.info(
                "Acq.xml found: %s | expected channels: %s",
                acq_path,
                ", ".join(acq_expected_channels),
            )

    os.makedirs(args.output_folder, exist_ok=True)

    search_roots = find_essen_roots(input_files)
    primary_paths = {p.replace("\\", "/") for p in input_files}

    success = 0
    failed = 0

    group_items = sorted(groups.items())

    if args.no_parallel or len(group_items) == 1:
        for base_id, channels in group_items:
            sidecar_groups = find_sidecar_files(
                base_id=base_id,
                search_roots=search_roots,
                primary_paths=primary_paths,
            )
            ok, _ = process_group(
                base_id=base_id,
                channels=channels,
                output_folder=args.output_folder,
                output_suffix=args.output_suffix,
                channel_order=channel_order,
                expected_channels=expected_channels,
                acq_expected_channels=acq_expected_channels,
                sidecar_groups=sidecar_groups,
                strict=args.strict,
                dry_run=args.dry_run,
                embed_mask_channels=args.embed_mask_channels,
                override_physical_sizes=physical_pixel_sizes,
            )
            if ok:
                success += 1
            else:
                failed += 1
    else:
        max_workers = args.max_workers or min(os.cpu_count() or 4, len(group_items))
        logging.getLogger().setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)

        worker_args = [
            (
                base_id,
                channels,
                search_roots,
                primary_paths,
                args.output_folder,
                args.output_suffix,
                channel_order,
                expected_channels,
                acq_expected_channels,
                args.strict,
                args.dry_run,
                args.embed_mask_channels,
                physical_pixel_sizes,
                True,
                args.log_level,
            )
            for base_id, channels in group_items
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_group_worker, item): item[0] for item in worker_args}
            completed = 0
            total = len(futures)
            render_progress(completed, total)
            for future in as_completed(futures):
                base_id = futures[future]
                try:
                    _, ok, _ = future.result()
                    if ok:
                        success += 1
                    else:
                        failed += 1
                except Exception as exc:
                    failed += 1
                    logger.error("%s: failed in worker: %s", base_id, exc)
                completed += 1
                render_progress(completed, total)
            sys.stderr.write("\n")

    logger.info("%s", "=" * 60)
    logger.info("Completed: %d successful, %d failed", success, failed)
    logger.info("Output folder: %s", args.output_folder)
    logger.info("%s", "=" * 60)

    if failed:
        raise RuntimeError(f"{failed} group(s) failed. See logs for details.")


if __name__ == "__main__":
    main()
