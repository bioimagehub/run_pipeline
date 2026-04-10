"""
Tests for gaussian_blur.py.

Covers: _dtype_limits, gaussian_blur_stack, _parse_channels.

All tests use synthetic numpy arrays — no image files are required.
"""

import numpy as np
import pytest

import gaussian_blur as gb


# ---------------------------------------------------------------------------
# _dtype_limits
# ---------------------------------------------------------------------------

class TestDtypeLimits:
    @pytest.mark.parametrize("dtype,expected_min,expected_max", [
        (np.uint8,  0, 255),
        (np.uint16, 0, 65535),
        (np.int8,  -128, 127),
        (np.int16, -32768, 32767),
        (np.int32, -2147483648, 2147483647),
    ])
    def test_integer_dtypes(self, dtype, expected_min, expected_max):
        lo, hi = gb._dtype_limits(np.dtype(dtype))
        assert lo == expected_min
        assert hi == expected_max

    def test_float32_returns_none(self):
        assert gb._dtype_limits(np.dtype(np.float32)) is None

    def test_float64_returns_none(self):
        assert gb._dtype_limits(np.dtype(np.float64)) is None


# ---------------------------------------------------------------------------
# gaussian_blur_stack
# ---------------------------------------------------------------------------

class TestGaussianBlurStack:
    def _make_5d(self, t=1, c=1, z=1, y=24, x=24, dtype=np.float32):
        return np.zeros((t, c, z, y, x), dtype=dtype)

    def test_rejects_non_5d_array(self):
        arr = np.zeros((10, 10))
        with pytest.raises(ValueError, match="Expected 5D TCZYX array"):
            gb.gaussian_blur_stack(arr, "2d", sigma_xy=1.0, sigma_z=0.0, truncate=4.0)

    def test_2d_mode_preserves_shape(self):
        arr = self._make_5d()
        result = gb.gaussian_blur_stack(arr, "2d", sigma_xy=1.0, sigma_z=0.0, truncate=4.0)
        assert result.shape == arr.shape

    def test_3d_mode_preserves_shape(self):
        arr = self._make_5d()
        result = gb.gaussian_blur_stack(arr, "3d", sigma_xy=1.0, sigma_z=1.0, truncate=4.0)
        assert result.shape == arr.shape

    def test_2d_sigma_zero_is_identity(self):
        arr = np.random.default_rng(0).random((1, 1, 1, 20, 20)).astype(np.float32)
        result = gb.gaussian_blur_stack(arr, "2d", sigma_xy=0.0, sigma_z=0.0, truncate=4.0)
        np.testing.assert_array_almost_equal(result, arr)

    def test_2d_blur_smooths_spike(self):
        arr = self._make_5d()
        arr[0, 0, 0, 12, 12] = 1000.0
        result = gb.gaussian_blur_stack(arr, "2d", sigma_xy=2.0, sigma_z=0.0, truncate=4.0)
        # Peak value must be reduced by blurring
        assert result[0, 0, 0, 12, 12] < 1000.0

    def test_3d_blur_along_z(self):
        arr = self._make_5d(z=6)
        arr[0, 0, 3, 12, 12] = 1000.0  # single bright point
        result = gb.gaussian_blur_stack(arr, "3d", sigma_xy=1.0, sigma_z=1.0, truncate=4.0)
        # Blur should spread to adjacent z-slices
        assert result[0, 0, 2, 12, 12] > 0.0
        assert result[0, 0, 4, 12, 12] > 0.0

    def test_integer_clipping(self):
        arr = np.full((1, 1, 1, 20, 20), 65000, dtype=np.uint16)
        result = gb.gaussian_blur_stack(arr, "2d", sigma_xy=1.0, sigma_z=0.0, truncate=4.0)
        # After blurring a uniform image values should remain constant
        assert result[0, 0, 0, 10, 10] == pytest.approx(65000, abs=10)

    def test_multichannel_processed(self):
        arr = self._make_5d(c=3)
        arr[0, 1, 0, 12, 12] = 500.0  # channel 1 bright
        result = gb.gaussian_blur_stack(arr, "2d", sigma_xy=2.0, sigma_z=0.0, truncate=4.0)
        # Channel 1 should be blurred
        assert result[0, 1, 0, 12, 12] < 500.0
        # Channels 0 and 2 remain zero
        assert result[0, 0, 0, 12, 12] == pytest.approx(0.0, abs=1e-6)
        assert result[0, 2, 0, 12, 12] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _parse_channels
# ---------------------------------------------------------------------------

class TestParseChannels:
    def test_none_returns_none(self):
        assert gb._parse_channels(None) is None

    def test_space_separated(self):
        assert gb._parse_channels("0 2 4") == [0, 2, 4]

    def test_comma_separated(self):
        assert gb._parse_channels("1,3") == [1, 3]

    def test_mixed_separator(self):
        assert gb._parse_channels("0, 2, 4") == [0, 2, 4]

    def test_single_channel(self):
        assert gb._parse_channels("3") == [3]

    def test_empty_string_returns_none(self):
        assert gb._parse_channels("") is None

    def test_invalid_string_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            gb._parse_channels("a,b")
