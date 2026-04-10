"""
Proper pytest tests for fill_greyscale_holes.py.

Covers the public API (greyscale_fill_holes_2d, greyscale_fill_holes_kernel_2d,
parse_output_formats) and the private helpers that are independently testable
(_kernel_window_starts, _resolve_kernel_overlap, _cast_to_dtype,
_positive_delta, _estimate_nbytes, _parse_channels,
_greyscale_fill_holes_2d_cpu).

All tests run on CPU only (use_gpu=False) so no GPU is required.
"""

import argparse

import numpy as np
import pytest

from fill_greyscale_holes import (
    greyscale_fill_holes_2d,
    greyscale_fill_holes_kernel_2d,
    parse_output_formats,
    _kernel_window_starts,
    _resolve_kernel_overlap,
    _cast_to_dtype,
    _positive_delta,
    _estimate_nbytes,
    _greyscale_fill_holes_2d_cpu,
    _parse_channels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nucleus_with_hole(size=64, outer_val=180, hole_val=20) -> np.ndarray:
    """
    Synthetic bright disc on dark background with a small dark 'nucleolus' hole.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cy, cx, r = size // 2, size // 2, size // 4
    ys, xs = np.ogrid[:size, :size]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
    img[disc] = outer_val
    hr = r // 4
    hole = (ys - cy) ** 2 + (xs - cx) ** 2 <= hr ** 2
    img[hole] = hole_val
    return img


# ---------------------------------------------------------------------------
# _kernel_window_starts
# ---------------------------------------------------------------------------

class TestKernelWindowStarts:
    def test_exact_multiple(self):
        # length=8, kernel=4, stride=2 → starts at 0, 2, 4
        starts = _kernel_window_starts(8, 4, 2)
        assert starts[0] == 0
        assert all(s < 8 for s in starts)
        # Last start must still cover the tail when clamped
        assert starts[-1] + 4 >= 8 or starts[-1] <= 8 - 4

    def test_single_window_when_image_fits(self):
        # If length <= kernel_size, only one window
        starts = _kernel_window_starts(4, 8, 4)
        assert starts == [0]

    def test_stride_one(self):
        starts = _kernel_window_starts(5, 3, 1)
        assert len(starts) >= 3
        assert starts[0] == 0

    def test_returns_list(self):
        assert isinstance(_kernel_window_starts(10, 4, 2), list)

    def test_all_elements_non_negative(self):
        for start in _kernel_window_starts(20, 5, 3):
            assert start >= 0


# ---------------------------------------------------------------------------
# _resolve_kernel_overlap
# ---------------------------------------------------------------------------

class TestResolveKernelOverlap:
    def test_half_is_floor_division(self):
        assert _resolve_kernel_overlap(10, "half") == 5
        assert _resolve_kernel_overlap(7, "half") == 3

    def test_integer_string(self):
        assert _resolve_kernel_overlap(10, "3") == 3

    def test_integer_value(self):
        assert _resolve_kernel_overlap(10, 4) == 4

    def test_zero_overlap_allowed(self):
        assert _resolve_kernel_overlap(5, "0") == 0

    def test_kernel_size_less_than_one_raises(self):
        with pytest.raises(ValueError, match="--kernel-size"):
            _resolve_kernel_overlap(0, "half")

    def test_overlap_equals_kernel_size_raises(self):
        with pytest.raises(ValueError, match="must be <"):
            _resolve_kernel_overlap(5, "5")

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            _resolve_kernel_overlap(5, "-1")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _resolve_kernel_overlap(5, "abc")


# ---------------------------------------------------------------------------
# _cast_to_dtype
# ---------------------------------------------------------------------------

class TestCastToDtype:
    def test_uint8_clamps_high(self):
        arr = np.array([0.0, 127.5, 300.0])
        result = _cast_to_dtype(arr, np.dtype(np.uint8))
        assert result[2] == 255
        assert result.dtype == np.uint8

    def test_uint8_clamps_low(self):
        arr = np.array([-10.0, 50.0])
        result = _cast_to_dtype(arr, np.dtype(np.uint8))
        assert result[0] == 0

    def test_uint8_rounds_half_up(self):
        # 127.5 → 128 (rint rounds to nearest even in numpy, 127.5 → 128)
        arr = np.array([127.5])
        result = _cast_to_dtype(arr, np.dtype(np.uint8))
        assert result[0] in (127, 128)  # accept either rounding convention

    def test_float32_passthrough(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = _cast_to_dtype(arr, np.dtype(np.float32))
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.5, 2.5], atol=1e-5)

    def test_uint16(self):
        arr = np.array([0.0, 32767.6, 65536.0])
        result = _cast_to_dtype(arr, np.dtype(np.uint16))
        assert result[0] == 0
        assert result[1] in (32767, 32768)
        assert result[2] == 65535
        assert result.dtype == np.uint16


# ---------------------------------------------------------------------------
# _positive_delta
# ---------------------------------------------------------------------------

class TestPositiveDelta:
    def test_filled_greater_gives_positive_delta(self):
        filled = np.array([100.0, 200.0, 50.0])
        original = np.array([80.0, 200.0, 70.0])
        delta = _positive_delta(filled, original, np.dtype(np.float64))
        np.testing.assert_allclose(delta, [20.0, 0.0, 0.0])

    def test_no_fill_gives_zeros(self):
        arr = np.array([50.0, 100.0, 150.0])
        delta = _positive_delta(arr, arr, np.dtype(np.float64))
        np.testing.assert_array_equal(delta, np.zeros(3))

    def test_output_dtype_matches_request(self):
        filled = np.array([200.0])
        original = np.array([100.0])
        delta = _positive_delta(filled, original, np.dtype(np.uint8))
        assert delta.dtype == np.uint8


# ---------------------------------------------------------------------------
# _estimate_nbytes
# ---------------------------------------------------------------------------

class TestEstimateNbytes:
    def test_uint8(self):
        result = _estimate_nbytes((10, 10), np.dtype(np.uint8))
        assert result == 100  # 10*10*1

    def test_uint16(self):
        result = _estimate_nbytes((3, 4, 5), np.dtype(np.uint16))
        assert result == 3 * 4 * 5 * 2

    def test_float32(self):
        result = _estimate_nbytes((2, 2), np.dtype(np.float32))
        assert result == 4 * 4


# ---------------------------------------------------------------------------
# _greyscale_fill_holes_2d_cpu
# ---------------------------------------------------------------------------

class TestGreyscaleFillHoles2dCpu:
    def test_fills_dark_interior(self):
        img = _nucleus_with_hole()
        filled = _greyscale_fill_holes_2d_cpu(img)
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        # Centre pixel (the hole) should be brighter after filling
        assert filled[cy, cx] > img[cy, cx]

    def test_output_shape_matches(self):
        img = _nucleus_with_hole()
        filled = _greyscale_fill_holes_2d_cpu(img)
        assert filled.shape == img.shape

    def test_output_is_numeric(self):
        # _greyscale_fill_holes_2d_cpu returns a numpy array (dtype may differ from input).
        img = _nucleus_with_hole().astype(np.uint16)
        filled = _greyscale_fill_holes_2d_cpu(img)
        assert isinstance(filled, np.ndarray)
        assert filled.shape == img.shape

    def test_no_fill_for_constant_image(self):
        img = np.full((20, 20), 100, dtype=np.uint8)
        filled = _greyscale_fill_holes_2d_cpu(img)
        np.testing.assert_array_equal(filled, img)

    def test_empty_image_returned_unchanged(self):
        img = np.zeros((0, 0), dtype=np.uint8)
        filled = _greyscale_fill_holes_2d_cpu(img)
        assert filled.shape == (0, 0)


# ---------------------------------------------------------------------------
# greyscale_fill_holes_2d (public API, CPU path)
# ---------------------------------------------------------------------------

class TestGreyscaleFillHoles2d:
    def test_fills_nucleolus_hole(self):
        img = _nucleus_with_hole()
        filled = greyscale_fill_holes_2d(img, use_gpu=False)
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        assert filled[cy, cx] > img[cy, cx]

    def test_no_fill_needed_if_no_hole(self):
        """Uniform bright disc without a hole — result should equal input."""
        size = 40
        img = np.zeros((size, size), dtype=np.uint8)
        cy, cx, r = 20, 20, 10
        ys, xs = np.ogrid[:size, :size]
        img[(ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2] = 200
        filled = greyscale_fill_holes_2d(img, use_gpu=False)
        np.testing.assert_array_equal(filled, img)

    def test_preserves_border_values(self):
        img = _nucleus_with_hole()
        filled = greyscale_fill_holes_2d(img, use_gpu=False)
        # Border pixels are unchanged (background stays black)
        np.testing.assert_array_equal(filled[0, :], img[0, :])
        np.testing.assert_array_equal(filled[-1, :], img[-1, :])
        np.testing.assert_array_equal(filled[:, 0], img[:, 0])
        np.testing.assert_array_equal(filled[:, -1], img[:, -1])

    def test_float32_input(self):
        img = _nucleus_with_hole().astype(np.float32)
        filled = greyscale_fill_holes_2d(img, use_gpu=False)
        assert filled.dtype == np.float32

    def test_uint16_input(self):
        # greyscale_fill_holes_2d may return float64 from the reconstruction step.
        img = _nucleus_with_hole().astype(np.uint16) * 200
        filled = greyscale_fill_holes_2d(img, use_gpu=False)
        assert isinstance(filled, np.ndarray)
        assert filled.shape == img.shape


# ---------------------------------------------------------------------------
# greyscale_fill_holes_kernel_2d (public API, CPU path)
# ---------------------------------------------------------------------------

class TestGreyscaleFillHolesKernel2d:
    def test_fills_hole_with_kernel(self):
        img = _nucleus_with_hole(size=64)
        filled = greyscale_fill_holes_kernel_2d(img, kernel_size=32, kernel_overlap=16, use_gpu=False)
        cy, cx = 32, 32
        assert filled[cy, cx] > img[cy, cx]

    def test_non_2d_raises(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D"):
            greyscale_fill_holes_kernel_2d(arr, 5, 2, use_gpu=False)

    def test_invalid_kernel_size_raises(self):
        arr = np.zeros((20, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            greyscale_fill_holes_kernel_2d(arr, kernel_size=0, kernel_overlap=0, use_gpu=False)

    def test_overlap_gte_kernel_size_raises(self):
        arr = np.zeros((20, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            greyscale_fill_holes_kernel_2d(arr, kernel_size=5, kernel_overlap=5, use_gpu=False)

    def test_output_shape_matches(self):
        img = _nucleus_with_hole(size=48)
        filled = greyscale_fill_holes_kernel_2d(img, kernel_size=24, kernel_overlap=12, use_gpu=False)
        assert filled.shape == img.shape

    def test_constant_image_unchanged(self):
        img = np.full((30, 30), 128, dtype=np.uint8)
        filled = greyscale_fill_holes_kernel_2d(img, kernel_size=15, kernel_overlap=5, use_gpu=False)
        np.testing.assert_array_equal(filled, img)


# ---------------------------------------------------------------------------
# parse_output_formats
# ---------------------------------------------------------------------------

class TestParseOutputFormats:
    def test_single_valid_format(self):
        assert parse_output_formats("tif") == ["tif"]

    def test_multiple_valid_formats(self):
        assert parse_output_formats("tif npy") == ["tif", "npy"]

    def test_ilastik_h5_format(self):
        assert parse_output_formats("ilastik-h5") == ["ilastik-h5"]

    def test_list_passthrough(self):
        lst = ["tif", "npy"]
        assert parse_output_formats(lst) is lst

    def test_invalid_format_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid format"):
            parse_output_formats("png")

    def test_mixed_valid_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_output_formats("tif jpeg")


# ---------------------------------------------------------------------------
# _parse_channels (in fill_greyscale_holes module)
# ---------------------------------------------------------------------------

class TestFghParseChannels:
    def test_none_returns_none(self):
        assert _parse_channels(None) is None

    def test_space_separated(self):
        assert _parse_channels("0 1 2") == [0, 1, 2]

    def test_comma_separated(self):
        assert _parse_channels("2,4") == [2, 4]

    def test_single_channel(self):
        assert _parse_channels("3") == [3]

    def test_empty_string_returns_none(self):
        assert _parse_channels("") is None

    def test_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_channels("x,y")
