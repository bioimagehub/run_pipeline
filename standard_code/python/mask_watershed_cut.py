"""
Watershed-based mask cutting with automatic fragment re-merging.

This script provides a watershed-based segmentation refinement workflow:
1. Apply watershed algorithm to binary masks (splits touching objects)
2. Identify and merge large fragments that are likely erroneous cuts
3. Save labeled watershed output and merged results
4. Generate ImageJ ROIs for merged results

Useful for refining segmentation masks with touching/overlapping objects.
"""

from __future__ import annotations
import os
import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import h_maxima


# Local imports
import bioimage_pipeline_utils as rp
import imageJ_view_server as ivs


# Configure logging
logger = logging.getLogger(__name__)


def watershed_xy(
	mask_yx: np.ndarray,
	tolerance: float = 0.5,
) -> np.ndarray:
	"""
	ImageJ-like 2D binary watershed on a single YX slice.

	Parameters
	----------
	mask_yx : np.ndarray
		2D binary mask (Y, X).
	tolerance : float
		Absolute EDM peak suppression (ImageJ MaximumFinder tolerance).
		Typical range: 0.3 – 1.0

	Returns
	-------
	np.ndarray
		Labeled 2D mask (Y, X), dtype=int32
	"""

	if mask_yx.ndim != 2:
		raise ValueError("mask_yx must be 2D (Y, X)")

	binary = mask_yx > 0

	if not np.any(binary):
		return np.zeros_like(mask_yx, dtype=np.int32)

	# 1. Euclidean Distance Transform (2D)
	dist = ndi.distance_transform_edt(binary)

	# 2. Absolute peak suppression (ImageJ-like behavior)
	maxima = h_maxima(dist, h=tolerance)

	# 3. Label maxima as watershed markers
	markers, _ = ndi.label(maxima)

	# 4. Watershed on negative distance
	labels = watershed(-dist, markers=markers, mask=binary, watershed_line=True)

	return labels.astype(np.int32)


def watershed_tczyx(
	mask_tczyx: np.ndarray,
	tolerance: float = 0.5,
) -> np.ndarray:
	"""
	Apply ImageJ-like watershed to a TCZYX mask.

	All looping over T, C, Z is handled here.
	watershed_xy handles only 2D logic.

	Parameters
	----------
	mask_tczyx : np.ndarray
		Binary mask of shape (T, C, Z, Y, X).
	tolerance : float
		Absolute EDM peak suppression parameter.

	Returns
	-------
	np.ndarray
		Labeled mask (T, C, Z, Y, X), dtype=int32
	"""

	if mask_tczyx.ndim != 5:
		raise ValueError("mask_tczyx must have shape (T, C, Z, Y, X)")

	T, C, Z, Y, X = mask_tczyx.shape
	output = np.zeros((T, C, Z, Y, X), dtype=np.int32)

	for t in range(T):
		for c in range(C):
			for z in range(Z):
				output[t, c, z] = watershed_xy(
					mask_tczyx[t, c, z],
					tolerance=tolerance,
				)

	return output


def merge_large_watershed_fragments(
	original_mask: np.ndarray,
	labeled_watershed: np.ndarray,
	max_fragment_length: int = 50,
) -> np.ndarray:
	"""
	Remove long separator lines to merge regions separated by watershed.

	After watershed with watershed_line=True, regions are separated by 0-valued
	"separator lines". This function identifies separator lines as pixels that were
	non-zero in the original mask but became 0 in the watershed result, then removes
	(merges) any separator lines that are longer than max_fragment_length.

	Similar to ImageJ's approach:
	1. Find pixels that changed from >0 to 0 (these are the actual separator lines)
	2. Label connected components of these separator lines
	3. Measure the size of each separator line
	4. Remove long separator lines by merging adjacent regions

	This function processes each (T, C, Z) plane independently.

	Parameters
	----------
	original_mask : np.ndarray
		Original binary mask before watershed (5D TCZYX).
	labeled_watershed : np.ndarray
		Labeled watershed output with separator lines (background=0, regions=1,2,...).
		5D array in TCZYX order.
	max_fragment_length : int
		Minimum separator line length in pixels. Separator lines longer than this
		will be removed to merge the regions they separate. Typical: 50-150 pixels.

	Returns
	-------
	np.ndarray
		Modified labeled mask with long separator lines removed.
		dtype=int32, same shape as input.
	"""

	output = original_mask.copy()
	
	# Ensure 5D
	if labeled_watershed.ndim != 5:
		raise ValueError("labeled_watershed must be 5D (TCZYX)")
	if original_mask.ndim != 5:
		raise ValueError("original_mask must be 5D (TCZYX)")
	
	T, C, Z, Y, X = labeled_watershed.shape
	
	# Create 5D binary mask to track which separators should be removed/merged
	separators = np.zeros((T, C, Z, Y, X), dtype=np.int32)
	
	logger.info(f"  Step 1: Identifying long separator lines (> {max_fragment_length} pixels)...")
	
	# Process each plane independently for labeling and merging
	for t in range(T):
		for c in range(C):
			for z in range(Z):
				original_plane = original_mask[t, c, z].astype(np.int32)
				watershed_plane = labeled_watershed[t, c, z].astype(np.int32)
				
				# Find separator pixels (were >0 in original but became 0 in watershed)
				separators[t, c, z] = (original_plane > 0) & (watershed_plane == 0)
				
				# Label connected components of separator lines (8-connectivity)
				structure = ndi.generate_binary_structure(2, 2)
				separators[t,c,z], n_separators = ndi.label(separators[t, c, z], structure=structure)
				
				
				# Remove long separator lines
				if n_separators > 0:
					# Calculate size of each separator component
					sep_ids = np.unique(separators[t,c,z])
					sep_ids = sep_ids[sep_ids > 0] # Exclude background=0

					

					sep_sizes = np.array([np.sum(separators[t,c,z] == sep_id) for sep_id in sep_ids])
					# print(f"Separator IDs: {sep_ids}, Sizes: {sep_sizes}") # Debug print to verify sizes
					long_sep_ids = sep_ids[sep_sizes >= max_fragment_length]
			
					for sep_id in long_sep_ids:
						separators[t, c, z][separators[t, c, z] == sep_id] = 0 # Mark for removal
						#ivs.show_image(labeled_separators_for_removal) # This is correct



	# ivs.show_image(separators)

	# Convert to binary explicitly (if not already)
	binary_output = (output > 0).astype(np.uint8)

	binary_output[separators > 0] = 0  # Add watershed splits to original mask to create a corrected watershed

	# 4-connected in 2D
	structure = ndi.generate_binary_structure(2, 1)
	relabeled = np.zeros_like(binary_output, dtype=np.int32)

	for t in range(T):
		for c in range(C):
			for z in range(Z):
				labeled_plane, _ = ndi.label(
					binary_output[t, c, z],
					structure=structure
				)
				relabeled[t, c, z] = labeled_plane.astype(np.int32)


	return relabeled


def remove_edge_objects(mask: np.ndarray, remove_xy: bool = True, remove_z: bool = False) -> np.ndarray:
	"""Remove objects touching image edges."""
	cleaned = np.zeros_like(mask)
	t, c, z, y, x = mask.shape
	
	for ti in range(t):
		for ci in range(c):
			for zi in range(z):
				plane = mask[ti, ci, zi]
				for label_id in np.unique(plane):
					if label_id == 0:
						continue
					
					region_mask = plane == label_id
					
					# Check XY edges
					touches_xy = (
						np.any(region_mask[0, :]) or np.any(region_mask[-1, :]) or
						np.any(region_mask[:, 0]) or np.any(region_mask[:, -1])
					)
					
					if remove_xy and touches_xy:
						continue
					
					# Check Z edges (across all z-planes for this object)
					if remove_z and (zi == 0 or zi == z - 1):
						if np.any(region_mask):
							continue
					
					cleaned[ti, ci, zi][region_mask] = label_id
	
	return cleaned


def remove_small_objects_from_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
	"""Remove objects smaller than min_size pixels."""
	if min_size <= 0:
		return mask
	
	cleaned = np.zeros_like(mask)
	t, c, z, y, x = mask.shape
	
	for ti in range(t):
		for ci in range(c):
			for zi in range(z):
				plane = mask[ti, ci, zi]
				for label_id in np.unique(plane):
					if label_id == 0:
						continue
					
					region_mask = plane == label_id
					if np.sum(region_mask) >= min_size:
						cleaned[ti, ci, zi][region_mask] = label_id
	
	return cleaned


def remove_large_objects_from_mask(mask: np.ndarray, max_size: float) -> np.ndarray:
	"""Remove objects larger than max_size pixels."""
	if max_size >= float('inf'):
		return mask
	
	cleaned = np.zeros_like(mask)
	t, c, z, y, x = mask.shape
	
	for ti in range(t):
		for ci in range(c):
			for zi in range(z):
				plane = mask[ti, ci, zi]
				for label_id in np.unique(plane):
					if label_id == 0:
						continue
					
					region_mask = plane == label_id
					if np.sum(region_mask) <= max_size:
						cleaned[ti, ci, zi][region_mask] = label_id
	
	return cleaned


def process_image(
	input_path: str,
	output_folder: str,
	tolerance: float = 0.5,
	max_fragment_length: int = 100,
	save_watershed: bool = True,
	save_rois: bool = False,
	remove_xy_edges: bool = False,
	remove_z_edges: bool = False,
	min_size: int = 0,
	max_size: float = float('inf'),
) -> None:
	"""Process a single mask image with watershed and merging.
	
	Always outputs binary masks (0/255) with separator lines (0 pixels) between regions
	created natively by the watershed algorithm (watershed_line=True).
	"""
	logger.info(f"Processing: {input_path}")

	# Prepare output paths
	input_name = Path(input_path).stem
	merged_path = os.path.join(output_folder, f"{input_name}_ws.tif")
	failed_mask_path_basename = os.path.join(output_folder, f"{input_name}")

	# Load image
	img = rp.load_tczyx_image(input_path)
	mask_data = img.data

	# exit early if mask is empty (no objects to segment)
	if not np.any(mask_data > 0):
		logger.info("  INFO: Input mask is empty. Saving empty output and skipping processing.")
		rp.save_mask(mask_data, merged_path, as_binary=True)
		if save_rois:
			logger.info("  No ROIs generated (empty mask)")
		else:
			logger.info("  Skipping ROI export (set --save-rois to enable)")
		return
	
	# Log how many objects in original mask
	n_objects_original = len(np.unique(mask_data)) - 1
	logger.info(f"  Original mask has {n_objects_original} objects")
	
	# Apply watershed
	logger.info(f"  Applying watershed (tolerance={tolerance})...")
	watershed_result = watershed_tczyx(mask_data, tolerance=tolerance)
	n_regions_watershed = len(np.unique(watershed_result)) - 1
	logger.info(f"    Created {n_regions_watershed} regions")

	
	# Merge long fragments
	logger.info(f"  Removing long separator lines (> {max_fragment_length} pixels)...")
	merged_result = merge_large_watershed_fragments(
		mask_data, watershed_result, max_fragment_length=max_fragment_length
	)
	n_regions_merged = len(np.unique(merged_result)) - 1
	logger.info(f"    After removing long separators: {n_regions_merged} regions")


	# Remove small objects
	if min_size > 0:
		logger.info(f"  Removing objects < {min_size} pixels...")
		merged_result_tmp = remove_small_objects_from_mask(merged_result, min_size)
		n_regions_filtered = len(np.unique(merged_result_tmp)) - 1
		logger.info(f"    Filtered to {n_regions_filtered} regions")
	

		# Early exit: this step emptied mask
		if np.any(merged_result > 0) and not np.any(merged_result_tmp > 0):
			logger.warning("  WARNING: min_size filtering emptied mask. Saving *_failed and early exit.")
			rp.save_mask(merged_result, failed_mask_path_basename + "_failed_rm_small.tif", as_binary=True)  # previous step
			rp.save_mask(merged_result_tmp, merged_path, as_binary=True) # current (empty) result
			if save_rois:
				logger.info("  No ROIs generated (empty mask)")
			else:
				logger.info("  Skipping ROI export (set --save-rois to enable)")
			return
		merged_result = merged_result_tmp  # IMPORTANT: keep filtered result	

	#print(np.unique(merged_result)	)

	# Remove large objects
	if max_size < float('inf'):
		logger.info(f"  Removing objects > {max_size} pixels...")
		merged_result_tmp = remove_large_objects_from_mask(merged_result, max_size)
		n_regions_filtered = len(np.unique(merged_result_tmp)) - 1
		logger.info(f"    Filtered to {n_regions_filtered} regions")
	

		# Early exit: this step emptied mask
		if np.any(merged_result > 0) and not np.any(merged_result_tmp > 0):
			logger.warning("  WARNING: max_size filtering emptied mask. Saving *_failed and early exit.")
			rp.save_mask(merged_result, failed_mask_path_basename + "_failed_rm_large.tif", as_binary=True)  # previous step
			rp.save_mask(merged_result_tmp, merged_path, as_binary=True) # current (empty) result
			if save_rois:
				logger.info("  No ROIs generated (empty mask)")
			else:
				logger.info("  Skipping ROI export (set --save-rois to enable)")
			return
		merged_result = merged_result_tmp  # IMPORTANT: keep filtered result	



	# Remove edge objects
	if remove_xy_edges or remove_z_edges:
		logger.info(f"  Removing edge objects (XY={remove_xy_edges}, Z={remove_z_edges})...")
		merged_result_tmp = remove_edge_objects(merged_result, remove_xy_edges, remove_z_edges)
		n_regions_filtered = len(np.unique(merged_result_tmp)) - 1
		logger.info(f"    Filtered to {n_regions_filtered} regions")
		
		# Early exit: this step emptied mask
		if np.any(merged_result > 0) and not np.any(merged_result_tmp > 0):
			logger.warning("  WARNING: edge filtering emptied mask. Saving *_failed and early exit.")
			rp.save_mask(merged_result, failed_mask_path_basename + "_failed_rm_edges.tif", as_binary=True)  # previous step
			rp.save_mask(merged_result_tmp, merged_path, as_binary=True) # current (empty) result
			if save_rois:
				logger.info("  No ROIs generated (empty mask)")
			else:
				logger.info("  Skipping ROI export (set --save-rois to enable)")
			return
		merged_result = merged_result_tmp  # IMPORTANT: keep filtered result	
	#print(np.unique(merged_result)	)


	
	if save_watershed:
		watershed_path = os.path.join(output_folder, f"{input_name}_ws_debug.tif")
		logger.info(f"  Saving watershed result to {watershed_path}...")
		rp.save_mask(watershed_result, watershed_path, as_binary=True)
	
	# Save merged result
	logger.info(f"  Saving merged result to {merged_path}...")
	rp.save_mask(merged_result, merged_path, as_binary=True)
	
	# Generate and save ROIs from merged result (optional)
	if save_rois:
		logger.info(f"  Generating ROIs from merged result...")
		rois = rp.mask_to_rois(merged_result)
		if rois:
			roi_path = os.path.join(output_folder, f"{input_name}_wsrois.zip")
			from roifile import roiwrite
			if os.path.exists(roi_path):
				os.remove(roi_path)
			roiwrite(roi_path, rois)
			logger.info(f"  Saved {len(rois)} ROIs to {roi_path}")
		else:
			logger.info("  No ROIs generated (empty mask)")
	else:
		logger.info("  Skipping ROI export (set --save-rois to enable)")
	
	logger.info(f"  ✓ Done")


def process_folder(
	input_pattern: str,
	output_folder: str,
	tolerance: float,
	max_fragment_length: int,
	save_watershed: bool,
	save_rois: bool,
	remove_xy_edges: bool,
	remove_z_edges: bool,
	min_size: int,
	max_size: float,
	no_parallel: bool
) -> None:
	"""Process multiple files matching the input pattern."""
	# Find files
	search_subfolders = '**' in input_pattern
	files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
	
	if not files:
		logger.warning(f"No files found matching: {input_pattern}")
		return
	
	logger.info(f"Found {len(files)} file(s) to process")
	os.makedirs(output_folder, exist_ok=True)
	
	def process_one(file_path):
		try:
			process_image(
				file_path, output_folder, tolerance, max_fragment_length, save_watershed,
				save_rois, remove_xy_edges, remove_z_edges, min_size, max_size
			)
		except Exception as e:
			logger.error(f"ERROR processing {file_path}: {e}")
	
	if not no_parallel and len(files) > 1:
		from joblib import Parallel, delayed
		Parallel(n_jobs=-1)(delayed(process_one)(f) for f in tqdm(files, desc="Processing"))
	else:
		for file_path in files:
			process_one(file_path)


# =============================================================================
# Example YAML configuration for run_pipeline.exe
# =============================================================================
"""
Example YAML config for run_pipeline.exe:

---
run:
  - name: Watershed cut with default merging
	environment: uv@3.11:segmentation
	commands:
	  - python
	  - '%REPO%/standard_code/python/mask_watershed_cut.py'
	  - --input-pattern: '%YAML%/input_masks/**/*.tif'
	  - --output-folder: '%YAML%/output_watershed'
	  - --tolerance: 0.5
	  - --max-fragment-length: 100
	  - --save-watershed

  - name: Watershed cut with size filtering
	environment: uv@3.11:segmentation
	commands:
	  - python
	  - '%REPO%/standard_code/python/mask_watershed_cut.py'
	  - --input-pattern: '%YAML%/masks/**/*.tif'
	  - --output-folder: '%YAML%/output_watershed'
	  - --tolerance: 0.5
	  - --max-fragment-length: 100
	  - --remove-xy-edges
	  - --min-size: 500
	  - --max-size: inf
	  - --save-watershed

  - name: Watershed cut aggressive merging
	environment: uv@3.11:segmentation
	commands:
	  - python
	  - '%REPO%/standard_code/python/mask_watershed_cut.py'
	  - --input-pattern: '%YAML%/masks/**/*.tif'
	  - --output-folder: '%YAML%/output_watershed'
	  - --tolerance: 1.0
	  - --max-fragment-length: 50
	  - --save-watershed



Notes:
  - Use --tolerance to control watershed sensitivity (higher = fewer cuts, typical: 0.3-1.0)
  - Use --max-fragment-length to control which fragments get re-merged (in pixels)
  - Use --remove-xy-edges to remove objects touching XY image boundaries
  - Use --remove-z-edges to remove objects at first/last Z slices
  - Use --min-size to filter out small objects (pixels)
  - Use --max-size to filter out large objects (pixels)
  - Separator lines (0 pixels) are always added between regions (ImageJ-style)
  - Binary mask (0/255) output is always used
  - Use --save-watershed to keep intermediate watershed result (helpful for debugging)
	- Use --save-rois to export ImageJ ROI ZIP files
  - Parallel processing is enabled by default, use --no-parallel to disable
  - Use --verbose to see detailed processing logs
"""


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Watershed-based mask segmentation with automatic fragment re-merging",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Workflow:
  1. Apply watershed to split touching objects (with separator lines via watershed_line=True)
  2. Merge large fragments (likely erroneous cuts) back to neighbors
  3. Filter objects by size and edge position
  4. Save as binary mask with 0/255 values
  5. Generate ImageJ ROIs from merged masks

Parameters:
  --tolerance: Controls watershed peak suppression (0.3-1.0 typical)
			   Lower = more cuts, higher = fewer cuts
  --max-fragment-length: Fragments larger than this are re-merged (in pixels)
  --remove-xy-edges: Remove objects touching XY image boundaries
  --remove-z-edges: Remove objects at first/last Z slices
  --min-size: Minimum object size in pixels
  --max-size: Maximum object size in pixels

Always enabled:
  - Separator lines (0 pixels) between regions (via watershed algorithm)
  - Binary mask output (foreground=255, background=0)

Example usage:
  python mask_watershed_cut.py --input-pattern "masks/*.tif" \
	  --output-folder "output" --tolerance 0.5 --max-fragment-length 100 \
	  --remove-xy-edges --min-size 100 --max-size 100000 --save-rois
		"""
	)
	
	parser.add_argument(
		"--input-pattern", required=True,
		help="Input file pattern for binary masks (supports wildcards, use '**' for recursive)"
	)
	parser.add_argument(
		"--output-folder", required=True,
		help="Output folder for watershed/merged results (and ROIs if --save-rois is set)"
	)
	parser.add_argument(
		"--tolerance", type=float, default=0.5,
		help="EDM peak suppression tolerance for watershed (default: 0.5, range: 0.1-2.0)"
	)
	parser.add_argument(
		"--max-fragment-length", type=int, default=100,
		help="Maximum fragment size for re-merging in pixels (default: 100)"
	)
	parser.add_argument(
		"--remove-xy-edges", action="store_true",
		help="Remove objects touching XY edges"
	)
	parser.add_argument(
		"--remove-z-edges", action="store_true",
		help="Remove objects touching Z edges"
	)
	parser.add_argument(
		"--min-size", type=int, default=0,
		help="Minimum object size in pixels (0 = no filtering, default: 0)"
	)
	parser.add_argument(
		"--max-size", type=str, default="inf",
		help="Maximum object size in pixels ('inf' = no filtering, default: inf)"
	)
	parser.add_argument(
		"--save-watershed", action="store_true",
		help="Save intermediate watershed result (helpful for debugging)"
	)
	parser.add_argument(
		"--save-rois", action="store_true",
		help="Save ImageJ ROIs as .zip files (disabled by default)"
	)
	parser.add_argument(
		"--no-parallel", action="store_true",
		help="Disable parallel processing (parallel is enabled by default)"
	)
	parser.add_argument(
		"--verbose", "-v", action="store_true",
		help="Enable verbose logging output"
	)
	
	args = parser.parse_args()
	
	# Configure logging
	log_level = logging.INFO if args.verbose else logging.WARNING
	logging.basicConfig(
		level=log_level,
		format='%(message)s'
	)
	
	# Parse max_size (handle 'inf' string)
	max_size = float('inf') if args.max_size.lower() == 'inf' else float(args.max_size)
	
	process_folder(
		input_pattern=args.input_pattern,
		output_folder=args.output_folder,
		tolerance=args.tolerance,
		max_fragment_length=args.max_fragment_length,
		save_watershed=args.save_watershed,
		save_rois=args.save_rois,
		remove_xy_edges=args.remove_xy_edges,
		remove_z_edges=args.remove_z_edges,
		min_size=args.min_size,
		max_size=max_size,
		no_parallel=args.no_parallel
	)
	
	logger.info("\n✓ All processing complete!")
