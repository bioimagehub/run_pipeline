from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed

import imageJ_view_server as ijserver
import bioimage_pipeline_utils as rp

MASK_PATH = (
	r"C:\Users\oodegard\Documents\BIPHub_files\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv"
	r"\SP20250627\output_masks\SP20250625_PC_R3_3SA__mask.tif"
)


def _watershed_binary_mask(
	mask_zyx: np.ndarray,
	min_distance: int = 5,
	compactness: float = 0.0,
) -> np.ndarray:
	"""Run watershed on a binary ZYX mask and return labeled components."""
	binary = mask_zyx > 0
	if not np.any(binary):
		return np.zeros(mask_zyx.shape, dtype=np.int32)

	dist = distance_transform_edt(binary)
	coords = peak_local_max(
		dist,
		min_distance=min_distance,
		labels=binary,
	)
	markers = np.zeros(mask_zyx.shape, dtype=np.int32)
	if coords.size == 0:
		# Fallback to connected components when no peaks are found.
		markers = label(binary)
	else:
		markers[tuple(coords.T)] = 1
		markers = label(markers)
	labels = watershed(-dist, markers, mask=binary, compactness=compactness)
	return labels.astype(np.int32, copy=False)


def watershed_mask_tczyx(
	mask_tczyx: np.ndarray,
	min_distance: int = 5,
	compactness: float = 0.0,
) -> np.ndarray:
	"""Apply watershed per (T, C) on a TCZYX mask and return labeled TCZYX."""
	if mask_tczyx.ndim != 5:
		raise ValueError("Expected TCZYX mask array with 5 dimensions.")

	t_size, c_size, z_size, y_size, x_size = mask_tczyx.shape
	out = np.zeros((t_size, c_size, z_size, y_size, x_size), dtype=np.int32)
	for t in range(t_size):
		for c in range(c_size):
			out[t, c] = _watershed_binary_mask(
				mask_tczyx[t, c],
				min_distance=min_distance,
				compactness=compactness,
			)
	return out


def main() -> None:
	img = rp.load_tczyx_image(MASK_PATH)
	mask_tczyx = img.data
	labels_tczyx = watershed_mask_tczyx(mask_tczyx, min_distance=5, compactness=0.0)
	ijserver.show_image(labels_tczyx, verbose=True)


if __name__ == "__main__":
	main()


