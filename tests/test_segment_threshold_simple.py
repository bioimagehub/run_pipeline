"""
Tests for segment_threshold_simple.py.

Covers: apply_gaussian_blur, apply_threshold_simple, apply_numeric_threshold,
fill_holes_in_mask, remove_edge_objects, _filter_labels_by_size,
remove_small_objects_from_mask, remove_large_objects_from_mask.

All tests use synthetic numpy arrays — no image files are required.
"""

import numpy as np
import pytest

import segment_threshold_simple as sts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_5d(t=1, c=1, z=1, y=20, x=20, dtype=np.uint16):
    """Return an all-zero 5D TCZYX array."""
    return np.zeros((t, c, z, y, x), dtype=dtype)


def _bright_circle_plane(size=20, cx=10, cy=10, r=6, val=1000):
    """Return a 2D plane with a bright circle."""
    plane = np.zeros((size, size), dtype=np.uint16)
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                plane[y, x] = val
    return plane


def _labeled_5d_with_two_objects():
    """5D mask with two distinct labeled objects not touching the border."""
    mask = _make_5d()
    mask[0, 0, 0, 3:7, 3:7] = 1    # object 1 (16 px)
    mask[0, 0, 0, 12:17, 12:17] = 2  # object 2 (25 px)
    return mask


# ---------------------------------------------------------------------------
# apply_gaussian_blur
# ---------------------------------------------------------------------------

class TestApplyGaussianBlur:
    def test_zero_sigma_returns_unchanged(self):
        img = _make_5d()
        img[:] = 100
        result = sts.apply_gaussian_blur(img, 0)
        np.testing.assert_array_equal(result, img)

    def test_negative_sigma_returns_unchanged(self):
        img = _make_5d()
        img[:] = 50
        result = sts.apply_gaussian_blur(img, -1.0)
        np.testing.assert_array_equal(result, img)

    def test_positive_sigma_smooths_image(self):
        img = np.zeros((1, 1, 1, 30, 30), dtype=np.float32)
        # Single bright pixel → Gaussian spread
        img[0, 0, 0, 15, 15] = 1000.0
        result = sts.apply_gaussian_blur(img.astype(float), 2.0)
        # The bright pixel value should have been reduced
        assert result[0, 0, 0, 15, 15] < 1000.0

    def test_output_shape_preserved(self):
        img = _make_5d(t=2, c=3, z=4, y=16, x=16)
        result = sts.apply_gaussian_blur(img, 1.0)
        assert result.shape == img.shape

    def test_multichannel_all_processed(self):
        img = np.zeros((1, 2, 1, 20, 20), dtype=np.float32)
        img[0, 0, 0, 10, 10] = 500.0
        img[0, 1, 0, 10, 10] = 500.0
        result = sts.apply_gaussian_blur(img, 2.0)
        # Both channels blurred → values spread
        assert result[0, 0, 0, 10, 10] < 500.0
        assert result[0, 1, 0, 10, 10] < 500.0


# ---------------------------------------------------------------------------
# apply_threshold_simple
# ---------------------------------------------------------------------------

class TestApplyThresholdSimple:
    def _image_with_bright_region(self):
        img = _make_5d(y=20, x=20)
        img[0, 0, 0, 5:15, 5:15] = 5000
        return img

    def test_otsu_produces_labeled_mask(self):
        img = self._image_with_bright_region()
        mask = sts.apply_threshold_simple(img, "otsu", channel=0)
        assert mask.shape == img.shape
        assert mask.dtype == np.uint16
        # Label 1 should appear for the bright region
        assert 1 in np.unique(mask)

    def test_invalid_method_raises(self):
        img = _make_5d()
        with pytest.raises(ValueError, match="Unsupported method"):
            sts.apply_threshold_simple(img, "nonexistent_method")

    def test_all_zero_image_gives_zero_mask(self):
        img = _make_5d()
        mask = sts.apply_threshold_simple(img, "otsu")
        # Otsu on all-zero might warn but should not raise
        assert mask.shape == img.shape

    @pytest.mark.parametrize("method", ["otsu", "yen", "li", "triangle", "mean", "isodata"])
    def test_supported_methods_run(self, method):
        img = self._image_with_bright_region()
        mask = sts.apply_threshold_simple(img, method)
        assert mask.shape == img.shape
        assert mask.dtype == np.uint16

    def test_background_stays_zero(self):
        img = _make_5d(y=30, x=30)
        # Only top-left 5×5 is bright, rest is zero
        img[0, 0, 0, 0:5, 0:5] = 8000
        mask = sts.apply_threshold_simple(img, "otsu")
        # The bottom-right corner should be background (0)
        assert mask[0, 0, 0, 25, 25] == 0


# ---------------------------------------------------------------------------
# apply_numeric_threshold
# ---------------------------------------------------------------------------

class TestApplyNumericThreshold:
    def test_above_min_is_labeled(self):
        img = _make_5d(y=20, x=20)
        img[0, 0, 0, 5:10, 5:10] = 500
        mask = sts.apply_numeric_threshold(img, min_val=100, max_val=float("inf"))
        assert mask[0, 0, 0, 7, 7] > 0

    def test_below_min_is_background(self):
        img = _make_5d()
        img[0, 0, 0, :, :] = 50
        mask = sts.apply_numeric_threshold(img, min_val=100, max_val=float("inf"))
        assert np.all(mask == 0)

    def test_between_min_max_is_labeled(self):
        img = _make_5d(y=20, x=20)
        img[0, 0, 0, 5:10, 5:10] = 300
        mask = sts.apply_numeric_threshold(img, min_val=200, max_val=400)
        assert mask[0, 0, 0, 7, 7] > 0

    def test_above_max_is_background(self):
        img = _make_5d(y=20, x=20)
        img[0, 0, 0, 5:10, 5:10] = 1000
        mask = sts.apply_numeric_threshold(img, min_val=0, max_val=500)
        assert mask[0, 0, 0, 7, 7] == 0

    def test_output_dtype_is_uint16(self):
        img = _make_5d()
        mask = sts.apply_numeric_threshold(img, 0, 1000)
        assert mask.dtype == np.uint16

    def test_multiple_timepoints(self):
        img = _make_5d(t=3, y=20, x=20)
        img[:, 0, 0, 5:10, 5:10] = 600
        mask = sts.apply_numeric_threshold(img, 500, float("inf"))
        for t in range(3):
            assert mask[t, 0, 0, 7, 7] > 0


# ---------------------------------------------------------------------------
# fill_holes_in_mask
# ---------------------------------------------------------------------------

class TestFillHolesInMask:
    def test_empty_mask_unchanged(self):
        mask = _make_5d()
        filled = sts.fill_holes_in_mask(mask)
        np.testing.assert_array_equal(filled, mask)

    def test_hole_inside_object_is_filled(self):
        mask = _make_5d(y=20, x=20)
        # Ring: set entire 10×10 block to 1, then zero the interior 4×4
        mask[0, 0, 0, 5:15, 5:15] = 1
        mask[0, 0, 0, 7:12, 7:12] = 0   # hole inside ring
        filled = sts.fill_holes_in_mask(mask)
        # Hole should be filled (nonzero)
        assert filled[0, 0, 0, 9, 9] > 0

    def test_output_shape_unchanged(self):
        mask = _make_5d(t=2, c=2, z=3, y=16, x=16)
        filled = sts.fill_holes_in_mask(mask)
        assert filled.shape == mask.shape

    def test_output_dtype_is_uint16(self):
        mask = _make_5d()
        mask[0, 0, 0, 2:8, 2:8] = 1
        filled = sts.fill_holes_in_mask(mask)
        assert filled.dtype == np.uint16


# ---------------------------------------------------------------------------
# remove_edge_objects
# ---------------------------------------------------------------------------

class TestRemoveEdgeObjects:
    def _mask_with_edge_and_center_object(self):
        mask = _make_5d(y=20, x=20)
        mask[0, 0, 0, 0:5, 0:5] = 1     # touches top-left edge
        mask[0, 0, 0, 8:13, 8:13] = 2   # fully interior
        return mask

    def test_edge_object_removed(self):
        mask = self._mask_with_edge_and_center_object()
        cleaned = sts.remove_edge_objects(mask, remove_xy=True)
        assert 1 not in np.unique(cleaned)

    def test_interior_object_preserved(self):
        mask = self._mask_with_edge_and_center_object()
        cleaned = sts.remove_edge_objects(mask, remove_xy=True)
        assert 2 in np.unique(cleaned)

    def test_remove_xy_false_preserves_all(self):
        mask = self._mask_with_edge_and_center_object()
        cleaned = sts.remove_edge_objects(mask, remove_xy=False, remove_z=False)
        np.testing.assert_array_equal(cleaned, mask)

    def test_empty_mask_unchanged(self):
        mask = _make_5d()
        cleaned = sts.remove_edge_objects(mask, remove_xy=True)
        np.testing.assert_array_equal(cleaned, mask)

    def test_output_shape_preserved(self):
        mask = self._mask_with_edge_and_center_object()
        cleaned = sts.remove_edge_objects(mask)
        assert cleaned.shape == mask.shape


# ---------------------------------------------------------------------------
# remove_small_objects_from_mask / remove_large_objects_from_mask
# ---------------------------------------------------------------------------

class TestRemoveObjectsBySize:
    def test_small_object_removed(self):
        mask = _labeled_5d_with_two_objects()
        # Object 1 has 16 px, object 2 has 25 px; remove those < 20
        cleaned = sts.remove_small_objects_from_mask(mask, min_size=20)
        assert 1 not in np.unique(cleaned)
        assert 2 in np.unique(cleaned)

    def test_large_object_removed(self):
        mask = _labeled_5d_with_two_objects()
        cleaned = sts.remove_large_objects_from_mask(mask, max_size=20)
        assert 2 not in np.unique(cleaned)
        assert 1 in np.unique(cleaned)

    def test_min_size_zero_returns_unchanged(self):
        mask = _labeled_5d_with_two_objects()
        cleaned = sts.remove_small_objects_from_mask(mask, min_size=0)
        np.testing.assert_array_equal(cleaned, mask)

    def test_max_size_inf_returns_unchanged(self):
        mask = _labeled_5d_with_two_objects()
        cleaned = sts.remove_large_objects_from_mask(mask, max_size=float("inf"))
        np.testing.assert_array_equal(cleaned, mask)

    def test_min_size_larger_than_all_removes_all(self):
        mask = _labeled_5d_with_two_objects()
        cleaned = sts.remove_small_objects_from_mask(mask, min_size=10000)
        assert np.all(cleaned == 0)

    def test_output_shape_preserved(self):
        mask = _labeled_5d_with_two_objects()
        cleaned = sts.remove_small_objects_from_mask(mask, 10)
        assert cleaned.shape == mask.shape


# ---------------------------------------------------------------------------
# _filter_labels_by_size (internal helper)
# ---------------------------------------------------------------------------

class TestFilterLabelsBySize:
    def test_passthrough_when_no_limits(self):
        mask = _labeled_5d_with_two_objects()
        result = sts._filter_labels_by_size(mask)
        np.testing.assert_array_equal(result, mask)

    def test_both_limits_applied(self):
        mask = _labeled_5d_with_two_objects()
        # object 1: 16 px, object 2: 25 px — keep only 15..20 px range
        result = sts._filter_labels_by_size(mask, keep_min=15, keep_max=20)
        assert 1 in np.unique(result)
        assert 2 not in np.unique(result)
