import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path
from pathlib import Path
from typing import Optional, Union
import logging
import re
import numpy as np
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

import bioimage_pipeline_utils as rp
from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter


def resolve_rgb(img: BioImage) -> BioImage:
    """
    Convert TCZYXS RGB data (S=3) into TCZYX by folding S into C.
    Returns a BioImage with updated channel metadata when possible.
    """
    if len(img.shape) != 6 or img.shape[-1] != 3:
        return img

    data = img.dask_data.compute()  # TCZYXS
    t, c, z, y, x, s = data.shape
    data_tczyx = data.transpose(0, 1, 5, 2, 3, 4).reshape(t, c * s, z, y, x)

    base_channel_names = None
    try:
        if hasattr(img, "channel_names") and img.channel_names is not None:
            base_channel_names = [str(name) for name in img.channel_names]
    except Exception:
        base_channel_names = None

    if base_channel_names and len(base_channel_names) == c:
        channel_names = []
        for channel_name in base_channel_names:
            channel_names.extend([f"{channel_name}_R", f"{channel_name}_G", f"{channel_name}_B"])
    else:
        channel_names = [f"C{i}" for i in range(c * s)]

    physical_pixel_sizes = None
    try:
        if hasattr(img, "physical_pixel_sizes"):
            physical_pixel_sizes = img.physical_pixel_sizes
    except Exception:
        physical_pixel_sizes = None

    try:
        return BioImage(
            data_tczyx,
            dim_order="TCZYX",
            channel_names=channel_names,
            physical_pixel_sizes=physical_pixel_sizes,
        )
    except TypeError:
        # Fallback for BioImage versions that do not accept physical_pixel_sizes kwarg.
        return BioImage(
            data_tczyx,
            dim_order="TCZYX",
            channel_names=channel_names,
        )

def _filter_scenes(
    scenes: tuple[str, ...] | list[str],
    img: BioImage,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
) -> list[str]:
    """
    Filter scenes according to the requested strategy.

    Args:
        scenes: All available scene names.
        img: BioImage object (used for dimension-based filters).
        scene_filter: Filtering mode.
            - ``all``      – keep every scene.
            - ``largest``  – keep scenes whose YX pixel count equals the maximum (default).
            - ``smallest`` – keep scenes whose YX pixel count equals the minimum.
            - ``includes`` – keep scenes whose name contains ANY of *scene_filter_strings*.
            - ``excludes`` – keep scenes whose name does NOT contain ANY of *scene_filter_strings*.
        scene_filter_strings: Required for ``includes`` / ``excludes`` modes.

    Returns:
        Filtered list of scene names.
    """
    if scene_filter == "all":
        return list(scenes)

    if scene_filter in ("largest", "smallest"):
        scene_dims = {}
        for scene_id in scenes:
            dims = _get_scene_dimensions(img, scene_id)
            scene_dims[scene_id] = dims
            logger.info(f"Scene '{scene_id}': {dims[0]}x{dims[1]} pixels")

        pixel_counts = [dims[0] * dims[1] for dims in scene_dims.values()]
        if scene_filter == "largest":
            target = max(pixel_counts)
            label = "lower resolution pyramid level"
        else:
            target = min(pixel_counts)
            label = "higher resolution level"

        kept = []
        for scene_id, dims in scene_dims.items():
            if dims[0] * dims[1] == target:
                kept.append(scene_id)
            else:
                logger.info(f"Skipping scene '{scene_id}' - {label}")
        return kept

    if scene_filter == "includes":
        if not scene_filter_strings:
            logger.warning("scene_filter='includes' requires --scene-filter-strings; returning all scenes")
            return list(scenes)
        kept = [
            s for s in scenes
            if any(f in s for f in scene_filter_strings)
        ]
        skipped = [s for s in scenes if s not in kept]
        for s in skipped:
            logger.info(f"Skipping scene '{s}' - does not match includes filter")
        return kept

    if scene_filter == "excludes":
        if not scene_filter_strings:
            logger.warning("scene_filter='excludes' requires --scene-filter-strings; returning all scenes")
            return list(scenes)
        kept = [
            s for s in scenes
            if not any(f in s for f in scene_filter_strings)
        ]
        skipped = [s for s in scenes if s not in kept]
        for s in skipped:
            logger.info(f"Skipping scene '{s}' - matches excludes filter")
        return kept

    logger.warning(f"Unknown scene_filter '{scene_filter}', falling back to 'largest'")
    return _filter_scenes(scenes, img, scene_filter="largest")

def _group_scenes_by_timestamp(scenes: list[str]) -> list[tuple[str, list[str]]]:
    """
    Group scene names by timestamp while preserving original scene order.

    Scenes without timestamp are kept as single-item groups.
    """
    groups: dict[str, list[str]] = {}
    ordered_keys: list[str] = []

    for scene_name in scenes:
        timestamp = _extract_scene_timestamp(scene_name)
        key = timestamp if timestamp is not None else f"no_timestamp::{scene_name}"
        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(scene_name)

    return [(key, groups[key]) for key in ordered_keys]

def _normalize_scene_channel_label(scene_name: str) -> str:
    """Extract a stable channel label from the scene tail token."""
    tail = scene_name.split("/")[-1].strip()
    # Common OBF pattern: MLE(STAR GREEN) -> STAR GREEN
    if tail.startswith("MLE(") and tail.endswith(")") and len(tail) > 5:
        return tail[4:-1].strip()
    return tail

def _split_tiff_extension(output_path: str) -> tuple[str, str]:
    """Split output path into base and TIFF extension (supports .ome.tif/.ome.tiff)."""
    lower = output_path.lower()
    if lower.endswith(".ome.tif"):
        return output_path[:-8], output_path[-8:]
    if lower.endswith(".ome.tiff"):
        return output_path[:-9], output_path[-9:]
    if lower.endswith(".tiff"):
        return output_path[:-5], output_path[-5:]
    if lower.endswith(".tif"):
        return output_path[:-4], output_path[-4:]
    return output_path, ".ome.tif"

def _build_group_output_path(output_path: str, group_key: str, group_index: int) -> str:
    """Build one output path per group with sequential numeric suffixes."""
    base, ext = _split_tiff_extension(output_path)
    suffix = f"{group_index + 1}"
    return f"{base}_{suffix}{ext}"

def _build_metadata_output_path(output_path: str) -> str:
    """Build metadata sidecar path for a TIFF output path."""
    base, _ = _split_tiff_extension(output_path)
    return f"{base}_metadata.yaml"

def get_core_metadata(img: Union[BioImage, str]) -> dict:

    if isinstance(img, str):
        img = bioio_reader(img)
        if img is None:
            logger.warning(f"Could not read image for metadata extraction: {img}")
            return {}    
    

    # Image dimensions
    t, c, z, y, x = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X

    # Physical dimensions
    z_um, y_um, x_um = img.physical_pixel_sizes.Z, img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X

    # TODO Find out if time is possible to find

    # Channel info
    channel_info = [str(n) for n in img.channel_names]

    # Extract metadata
    image_metadata = {
        'Image metadata': {
            'Channels': [{'Name': f'{name}'} for name in channel_info],
            'Image dimensions': {'C': c, 'T': t, 'X': x, 'Y': y, 'Z': z},
            'Physical dimensions': {'T_ms': None, 'X_um': x_um, 'Y_um': y_um, 'Z_um': z_um},
        }
    }
    return image_metadata

def _apply_channel_names_to_metadata(metadata: dict, channel_names: list[str], channel_count: Optional[int] = None) -> dict:
    """Overwrite channel-related metadata with the channel names used for output."""
    image_meta = metadata.get("Image metadata")
    if not isinstance(image_meta, dict):
        return metadata

    image_meta["Channels"] = [{"Name": str(name)} for name in channel_names]

    image_dims = image_meta.get("Image dimensions")
    if isinstance(image_dims, dict):
        image_dims["C"] = int(channel_count if channel_count is not None else len(channel_names))

    return metadata

def _convert_rgb_tczyxs_to_tczyx(
    data: np.ndarray,
    channel_names: Optional[list[str]],
) -> tuple[np.ndarray, Optional[list[str]]]:
    """Convert TCZYXS (S=3 RGB) to TCZYX by folding S into C."""
    if data.ndim != 6 or data.shape[-1] != 3:
        return data, channel_names

    t, c, z, y, x, s = data.shape
    out = data.transpose(0, 1, 5, 2, 3, 4).reshape(t, c * s, z, y, x)

    if channel_names and len(channel_names) == c:
        expanded: list[str] = []
        for channel_name in channel_names:
            expanded.extend([f"{channel_name}_R", f"{channel_name}_G", f"{channel_name}_B"])
        return out, expanded

    return out, [f"C{i}" for i in range(c * s)]

def _get_scene_dimensions(img: BioImage, scene_id: str) -> tuple[int, int]:
    """
    Get the physical dimensions (Y, X pixel count) of a scene.
    
    Args:
        img: BioImage object
        scene_id: Scene identifier
    
    Returns:
        Tuple of (height, width) in pixels
    """
    img.set_scene(scene_id)
    shape = img.shape  # TCZYX
    return (shape[-2], shape[-1])  # Y, X

def _extract_scene_timestamp(scene_name: str) -> Optional[str]:
    """Extract timestamp in HH:MM:SS format from a scene name."""
    match = re.search(r"\d{2}:\d{2}:\d{2}", scene_name)
    if match:
        return match.group(0)
    return None

def bioio_reader(input_path: str) ->  Union[BioImage, None]:  
    if input_path.endswith(".nd2"):
        import bioio_nd2
        img = BioImage(input_path, reader=bioio_nd2.Reader)

    elif input_path.endswith(".czi"):
        import bioio_czi
        img = BioImage(input_path, reader=bioio_czi.Reader)

    elif input_path.endswith(".dv"):
        import bioio_dv
        img = BioImage(input_path, reader=bioio_dv.Reader)

    elif input_path.endswith(".lif"):
        import bioio_lif
        img = BioImage(input_path, reader=bioio_lif.Reader)

        # # print all the attributes of the BioImage object to find metadata
        # for attr in dir(img):
        #     if not attr.startswith("_"):
        #         try:
        #             value = getattr(img, attr)
        #             print(f"{attr}: {value}")
        #         except Exception as e:
        #             print(f"{attr}: <error reading attribute: {e}>")
        # sys.exit(0)
        # Could not find channel names for .lif


    elif input_path.endswith(".czi"):
        import bioio_czi
        img = BioImage(input_path, reader=bioio_czi.Reader)

    elif input_path.endswith((".png", ".gif", ".jpg", ".jpeg", ".bmp", ".webp")):
        import bioio_imageio
        img = BioImage(input_path, reader=bioio_imageio.Reader)
        if len(img.shape) == 6:
            img = resolve_rgb(img)

    elif input_path.endswith((".ome.tif", ".ome.tiff")):
        import bioio_ome_tiff
        img = BioImage(input_path, reader=bioio_ome_tiff.Reader)

    # elif input_path.endswith((".ome.tf8", ".ome.btf")):
    #     import bioio_ome_tiled_tiff
    #     img = BioImage(input_path, reader=bioio_ome_tiled_tiff.Reader)

    elif input_path.endswith(".zarr"):
        import bioio_ome_zarr
        img = BioImage(input_path, reader=bioio_ome_zarr.Reader)

    # elif input_path.endswith(".sldy"):
    #     import bioio_sldy
    #     img = BioImage(input_path, reader=bioio_sldy.Reader)

    elif input_path.endswith((".tif", ".tiff")):
        import bioio_tifffile
        img = BioImage(input_path, reader=bioio_tifffile.Reader)

    # elif input_path.endswith(".tifglob"):
    #     import bioio_tiff_glob
    #     img = BioImage(input_path, reader=bioio_tiff_glob.Reader)

    else:
        try:
            import bioio_bioformats
            import jdk4py
            import scyjava 
            import fix_java_home_problem  # calls fix_java_home_problem() on import 
            img = BioImage(input_path, reader=bioio_bioformats.Reader)

        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return None
    # print(img.shape)
    return img

def process_file(input_path: str, output_path: str,
                 scene_filter: str = "largest",
                 scene_filter_strings: Optional[list[str]] = None,
                 scene_group_by_timestamp: bool = False,
                 split_series: bool = True,
                 save_metadata: bool = True,
                 ) -> bool:
    '''
            scene_filter: Scene selection strategy ('all', 'largest', 'smallest',
            'includes', 'excludes'). largest should be default
    '''
    # Load a TCZYX image with bioio. (Possobly multiple scenes)
    print(f"Reading {input_path} with bioio...")
    img = bioio_reader(input_path)
    if img is None:
        print(f"Failed to read {input_path}")
        return False

    # print(f"Image shape: {img.shape}, scenes: {img.scenes}")

    # Determine which scenes to keep based on the filter
    scenes = img.scenes

    # If there is only one scene, just save the OME-TIFF
    if len(scenes) == 1:
        logger.info(f"Only one scene to process - saving as {output_path}")
        img.save(output_path)
        if save_metadata:
            metadata = get_core_metadata(img)
            metadata_path = _build_metadata_output_path(output_path)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                yaml.dump(metadata, metadata_file, default_flow_style=False, sort_keys=False)
            
        return True

    # If there are multiple scenes, apply the filter to determine which ones to keep
    logger.info(f"Found {len(scenes)} scene(s)")
    if len(scenes) > 1:
        logger.info(f"Applying scene filter: '{scene_filter}' (strings={scene_filter_strings})")
        scenes_to_process = _filter_scenes(
            scenes=scenes,
            img=img,
            scene_filter=scene_filter,
            scene_filter_strings=scene_filter_strings
        )
    else:
        scenes_to_process = list(scenes)

    if not scenes_to_process:
        logger.warning(f"Scene filter removed all scenes from {input_path} - skipping file")
        return False
    
    # If the user wants a multi-scene file as output and wants to keep all scenes
    if not split_series and len(scenes_to_process) == len(scenes):
        logger.info(f"Saving all {len(scenes_to_process)} scenes into a single OME-TIFF: {output_path}")
        img.save(output_path)
        if save_metadata:
            metadata = get_core_metadata(img)
            metadata_path = _build_metadata_output_path(output_path)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                yaml.dump(metadata, metadata_file, default_flow_style=False, sort_keys=False)
        return True

    if scene_group_by_timestamp:  
        if not input_path.lower().endswith("obf"):
            logger.warning(f"scene_group_by_timestamp=True is intended for OBF files. The input file {input_path} does not have .obf extension. Proceeding anyway, but results may be unexpected.")

        # This is a special case for OBF Abberior STED images
        scene_groups = _group_scenes_by_timestamp(scenes_to_process)
        logger.info(
            f"Grouping by timestamp: {len(scenes_to_process)} scenes -> {len(scene_groups)} output file(s)"
        )

        saved_files: list[str] = []
        for group_idx, (group_key, group_scene_ids) in enumerate(scene_groups):
            # Enforce deterministic channel ordering between files.
            sorted_scene_ids = sorted(
                group_scene_ids,
                key=lambda s: (_normalize_scene_channel_label(s).lower(), s.lower()),
            )

            merged_data: Optional[np.ndarray] = None
            merged_channel_names: list[str] = []
            reference_tzyx: Optional[tuple[int, int, int, int]] = None
            physical_pixel_sizes = None

            logger.info(
                f"Processing timestamp group '{group_key}' with {len(sorted_scene_ids)} scene(s)"
            )

            for scene_id in sorted_scene_ids:
                img.set_scene(scene_id)

                scene_channel_names: Optional[list[str]] = None
                try:
                    if physical_pixel_sizes is None and hasattr(img, "physical_pixel_sizes"):
                        physical_pixel_sizes = img.physical_pixel_sizes
                    if hasattr(img, "channel_names") and img.channel_names is not None:
                        scene_channel_names = [str(name) for name in img.channel_names]
                except Exception as exc:
                    logger.warning(f"Could not read metadata for scene '{scene_id}': {exc}")

                # Load one scene at a time to limit memory footprint.
                data = np.asarray(img.get_image_data("TCZYX"))
                data, scene_channel_names = _convert_rgb_tczyxs_to_tczyx(data, scene_channel_names)

                if data.ndim != 5:
                    logger.warning(
                        f"Skipping scene '{scene_id}' due to unsupported ndim={data.ndim}; expected TCZYX"
                    )
                    continue

                current_tzyx = (data.shape[0], data.shape[2], data.shape[3], data.shape[4])
                if reference_tzyx is None:
                    reference_tzyx = current_tzyx
                elif current_tzyx != reference_tzyx:
                    logger.warning(
                        f"Skipping scene '{scene_id}' due to shape mismatch: "
                        f"{current_tzyx} vs expected {reference_tzyx}"
                    )
                    continue

                scene_label = _normalize_scene_channel_label(scene_id)
                scene_c = data.shape[1]
                if scene_c == 1:
                    ordered_channel_names = [scene_label]
                else:
                    ordered_channel_names = [f"{scene_label}_{i + 1}" for i in range(scene_c)]

                if merged_data is None:
                    merged_data = data
                else:
                    merged_data = np.concatenate([merged_data, data], axis=1)

                merged_channel_names.extend(ordered_channel_names)

            if merged_data is None:
                logger.warning(f"No valid scene data for timestamp group '{group_key}', skipping output")
                continue

            if len(merged_channel_names) != merged_data.shape[1]:
                logger.warning(
                    f"Channel name count ({len(merged_channel_names)}) does not match C ({merged_data.shape[1]}). "
                    "Falling back to generated names."
                )
                merged_channel_names = [f"C{i}" for i in range(merged_data.shape[1])]

            scene_output_path = _build_group_output_path(output_path, group_key, group_idx)
            os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)

            kwargs = {"channel_names": merged_channel_names}
            if physical_pixel_sizes is not None:
                kwargs["physical_pixel_sizes"] = physical_pixel_sizes

            logger.info(
                f"Saving grouped timestamp '{group_key}' to {scene_output_path} with shape {merged_data.shape} "
                f"and channels {merged_channel_names}"
            )
            OmeTiffWriter.save(merged_data, scene_output_path, dim_order="TCZYX", **kwargs)
            if save_metadata:
                metadata = get_core_metadata(scene_output_path)
                metadata = _apply_channel_names_to_metadata(
                    metadata,
                    merged_channel_names,
                    channel_count=merged_data.shape[1],
                )
                metadata_path = _build_metadata_output_path(scene_output_path)
                with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                    yaml.dump(metadata, metadata_file, default_flow_style=False, sort_keys=False)


            saved_files.append(scene_output_path)

        if not saved_files:
            logger.warning(f"No grouped outputs were written for {input_path}")
            return False
        else:
            logger.info(f"Wrote {len(saved_files)} grouped output file(s)")
        return True
        

    # This is the default behavior for multi-scene files
    # To ensure that the metadata is correct for each scene we use tifffile
    # Otherwise, save each scene as a separate OME-TIFF file with _S0, _S1 suffixes before ome.tif
    for i, scene_id in enumerate(scenes_to_process):
        print(f"Processing scene '{scene_id}'...")
        img.set_scene(scene_id)
        scene_output_path = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_S{i}.ome.tif"
        )
        logger.info(f"Saving scene '{scene_id}' as {scene_output_path}")

        # Attempt to extract metadta 
        kwargs = {}
        if hasattr(img, "physical_pixel_sizes") and img.physical_pixel_sizes is not None: 
            logger.info(f"Scene '{scene_id}' physical pixel sizes: {img.physical_pixel_sizes}")
        if hasattr(img, "channel_names") and img.channel_names is not None:
            logger.info(f"Scene '{scene_id}' channel names: {img.channel_names}")
            kwargs["channel_names"] = img.channel_names

        # TODO: Find out how to transfer the lookup tables

        dask_data = img.dask_data
        data = dask_data.compute()
        OmeTiffWriter.save(data, scene_output_path, dim_order="TCZYX", **kwargs)
        if save_metadata:
            metadata = get_core_metadata(scene_output_path)
            metadata_path = _build_metadata_output_path(scene_output_path)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                yaml.dump(metadata, metadata_file, default_flow_style=False, sort_keys=False)



    
    # img.save(output_path)
    return True


def _process_file_task(
    input_path: str,
    output_path: str,
    scene_filter: str,
    scene_filter_strings: Optional[list[str]],
    scene_group_by_timestamp: bool,
    split_series: bool,
    save_metadata: bool,
) -> bool:
    try:
        return process_file(
            input_path=input_path,
            output_path=output_path,
            scene_filter=scene_filter,
            scene_filter_strings=scene_filter_strings,
            scene_group_by_timestamp=scene_group_by_timestamp,
            split_series=split_series,
            save_metadata=save_metadata,
        )
    except Exception as exc:
        logger.error(f"Failed to process {input_path}: {exc}")
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    save_metadata: bool = True,
    dry_run: bool = False,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
    scene_group_by_timestamp: bool = False,
    split_series: bool = True,
) -> None:
    """Process multiple files matching a search pattern."""
    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return

    logger.info(f"Found {len(files)} file(s) to process")

    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)

    if output_folder is None:
        output_folder = base_folder + "_tif"

    logger.info(f"Output folder: {output_folder}")

    file_pairs: list[tuple[str, str]] = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        out_name = os.path.splitext(collapsed)[0] + "_ome.tif"
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))

    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Scene filter: {scene_filter} (strings={scene_filter_strings})")
        print(f"[DRY RUN] Group scenes by timestamp: {scene_group_by_timestamp}")
        print(f"[DRY RUN] Split series: {split_series}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    os.makedirs(output_folder, exist_ok=True)

    if no_parallel or len(file_pairs) == 1:
        failures = 0
        for src, dst in tqdm(file_pairs, desc="Processing files", unit="file"):
            success = _process_file_task(
                input_path=src,
                output_path=dst,
                scene_filter=scene_filter,
                scene_filter_strings=scene_filter_strings,
                scene_group_by_timestamp=scene_group_by_timestamp,
                split_series=split_series,
                save_metadata=save_metadata,
            )
            if not success:
                failures += 1
        if failures:
            logger.error(f"Failed processing {failures} file(s)")
        return

    max_workers = min(os.cpu_count() or 4, len(file_pairs))
    logger.info(f"Processing with {max_workers} workers")

    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_file_task,
                src,
                dst,
                scene_filter,
                scene_filter_strings,
                scene_group_by_timestamp,
                split_series,
                save_metadata,
            ): (src, dst)
            for src, dst in file_pairs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            src, _ = futures[future]
            try:
                success = future.result()
                if not success:
                    failures += 1
                    logger.error(f"Failed: {src}")
            except Exception as exc:
                failures += 1
                logger.error(f"Exception processing {src}: {exc}")

    if failures:
        logger.error(f"Failed processing {failures} file(s)")

def process_folder(input_dir: str, output_dir: str, split_series: bool = True, scene_filter: str = "largest", scene_filter_strings: Optional[list[str]] = None, scene_group_by_timestamp: bool = False, save_metadata: bool = True):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir)]
    
    for file in files:
        input_path = path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_ome.tif")

        process_file(input_path, output_path, split_series=split_series, scene_filter=scene_filter, scene_filter_strings=scene_filter_strings, scene_group_by_timestamp=scene_group_by_timestamp, save_metadata=save_metadata)
        


def main():
    parser = argparse.ArgumentParser(
        description="Convert microscopy image files to OME-TIFF using the original EDNA conversion logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert files with default multi-scene handling
  environment: uv@3.11:convert-to-tif-edna
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif_edna.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --log-level: INFO

- name: Convert OBF files and group matching scenes by timestamp
  environment: uv@3.11:convert-to-tif-edna
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif_edna.py'
  - --input-search-pattern: '%YAML%/input/**/*.obf'
  - --output-folder: '%YAML%/output'
  - --scene-filter: includes
  - --scene-filter-strings: /MLE(
  - --scene-group-by-timestamp
  - --log-level: INFO

- name: Convert files sequentially and keep all scenes in one output when possible
  environment: uv@3.11:convert-to-tif-edna
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif_edna.py'
  - --input-search-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --scene-filter: all
  - --no-split-series
  - --no-parallel
  - --log-level: INFO
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
        help="Output folder (default: input_folder + '_tif')",
    )
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder paths (default: '__')",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip saving metadata YAML sidecars",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing",
    )
    parser.add_argument(
        "--scene-filter",
        type=str,
        default="largest",
        choices=["all", "largest", "smallest", "includes", "excludes"],
        help=(
            "Scene selection strategy for multi-scene files (default: largest). "
            "'all' keeps every scene. "
            "'largest'/'smallest' selects by YX pixel count. "
            "'includes' keeps scenes whose name contains any --scene-filter-strings value. "
            "'excludes' removes scenes whose name contains any --scene-filter-strings value."
        ),
    )
    parser.add_argument(
        "--scene-filter-strings",
        type=str,
        nargs="+",
        default=None,
        metavar="STRING",
        help=(
            "One or more substrings used with --scene-filter includes/excludes. "
            "Example: --scene-filter-strings /MLE(  or  --scene-filter-strings Overview Tile"
        ),
    )
    parser.add_argument(
        "--scene-group-by-timestamp",
        action="store_true",
        help="Group filtered scenes by HH:MM:SS timestamp in scene name and merge each group into channels",
    )
    parser.add_argument(
        "--no-split-series",
        action="store_true",
        help="Keep all scenes in one output only when all scenes are selected; otherwise preserves current split-series behavior",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"convert_to_tif_edna.py version: {version}")
        return

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        save_metadata=not args.no_metadata,
        dry_run=args.dry_run,
        scene_filter=args.scene_filter,
        scene_filter_strings=args.scene_filter_strings,
        scene_group_by_timestamp=args.scene_group_by_timestamp,
        split_series=not args.no_split_series,
    )

if __name__ == "__main__":
    main()