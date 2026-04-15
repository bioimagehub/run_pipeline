"""Regression tests for memory-safe filtering."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import filter as filter_module


class _ComputedSlice:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def compute(self) -> np.ndarray:
        return np.asarray(self._array)


class _LazyArray:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.dtype = array.dtype

    def __getitem__(self, key):
        return _ComputedSlice(self._array[key])


class _NoEagerDataImage:
    def __init__(self, array: np.ndarray) -> None:
        self.shape = array.shape
        self.dask_data = _LazyArray(array)

    @property
    def data(self) -> np.ndarray:
        raise MemoryError("eager .data access should not be required for this path")


def test_process_single_file_uses_lazy_loading_for_filtering() -> None:
    data = np.arange(1 * 2 * 3 * 8 * 8, dtype=np.uint16).reshape((1, 2, 3, 8, 8))
    fake_img = _NoEagerDataImage(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "filtered.npy"

        with (
            patch.object(filter_module, "_is_valid_tiff_for_read", return_value=True),
            patch.object(filter_module.rp, "load_tczyx_image", return_value=fake_img),
        ):
            ok = filter_module.process_single_file(
                input_path="synthetic.ome.tif",
                output_path=str(output_path),
                output_format="npy",
                method="median",
                mode="2d",
                channels=[1],
                sigma_xy=1.0,
                sigma_z=0.0,
                truncate=4.0,
                size_y=3,
                size_x=3,
                size_z=3,
                force=True,
            )

        assert ok is True
        saved = np.load(output_path)
        assert saved.shape == data.shape
        assert saved.dtype == data.dtype
        assert np.array_equal(saved[:, 0], data[:, 0])
        assert not np.array_equal(saved[:, 1], data[:, 1])
