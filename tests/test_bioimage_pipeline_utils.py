"""
Tests for bioimage_pipeline_utils.py utility functions.

Covers: get_default_maxcores, resolve_maxcores, collapse_filename,
uncollapse_filename, get_files_to_process2, get_grouped_files_to_process,
split_comma_separated_strstring, split_comma_separated_intstring.
"""

import os
import pytest
import warnings
import bioimage_pipeline_utils as rp


# ---------------------------------------------------------------------------
# get_default_maxcores
# ---------------------------------------------------------------------------

class TestGetDefaultMaxcores:
    def test_returns_positive_integer(self):
        result = rp.get_default_maxcores()
        assert isinstance(result, int)
        assert result >= 1

    def test_is_at_most_cpu_count_minus_one(self):
        cpu_count = os.cpu_count() or 1
        result = rp.get_default_maxcores()
        # Single-core machines clamp to 1; otherwise cpu_count - 1
        assert result == max(1, cpu_count - 1)


# ---------------------------------------------------------------------------
# resolve_maxcores
# ---------------------------------------------------------------------------

class TestResolveMaxcores:
    def test_none_uses_default(self):
        result = rp.resolve_maxcores(None)
        assert result == rp.get_default_maxcores()

    def test_explicit_value_respected(self):
        assert rp.resolve_maxcores(4) == 4

    def test_task_count_limits_result(self):
        # When task_count is smaller than requested cores, result is capped.
        assert rp.resolve_maxcores(100, task_count=3) == 3

    def test_task_count_larger_than_cores_uses_cores(self):
        cores = rp.get_default_maxcores()
        result = rp.resolve_maxcores(cores, task_count=cores + 100)
        assert result == cores

    def test_minimum_one(self):
        assert rp.resolve_maxcores(1) == 1

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            rp.resolve_maxcores(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            rp.resolve_maxcores(-2)


# ---------------------------------------------------------------------------
# collapse_filename / uncollapse_filename
# ---------------------------------------------------------------------------

class TestCollapseFilename:
    def test_basic_collapse(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "subdir", "image.tif")
        collapsed = rp.collapse_filename(file_path, base)
        assert "__" in collapsed
        assert "subdir" in collapsed
        assert "image.tif" in collapsed

    def test_custom_delimiter(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "a", "b.tif")
        collapsed = rp.collapse_filename(file_path, base, delimiter="--")
        assert "--" in collapsed
        assert "__" not in collapsed

    def test_no_subdir_gives_just_filename(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "image.tif")
        collapsed = rp.collapse_filename(file_path, base)
        assert collapsed == "image.tif"

    def test_multiple_subdirs(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "a", "b", "c.tif")
        collapsed = rp.collapse_filename(file_path, base)
        assert collapsed.count("__") == 2


class TestUncollapseFilename:
    def test_roundtrip(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "subdir", "image.tif")
        collapsed = rp.collapse_filename(file_path, base)
        recovered = rp.uncollapse_filename(collapsed, base)
        # Normalise separators for comparison
        assert os.path.normpath(recovered) == os.path.normpath(file_path)

    def test_custom_delimiter_roundtrip(self, tmp_path):
        base = str(tmp_path)
        file_path = os.path.join(base, "x", "y.tif")
        collapsed = rp.collapse_filename(file_path, base, delimiter="||")
        recovered = rp.uncollapse_filename(collapsed, base, delimiter="||")
        assert os.path.normpath(recovered) == os.path.normpath(file_path)


# ---------------------------------------------------------------------------
# get_files_to_process2
# ---------------------------------------------------------------------------

class TestGetFilesToProcess2:
    def test_finds_files_with_pattern(self, tmp_path):
        (tmp_path / "a.tif").write_bytes(b"x")
        (tmp_path / "b.tif").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        pattern = str(tmp_path / "*.tif")
        result = rp.get_files_to_process2(pattern, search_subfolders=False)
        assert len(result) == 2
        assert all(f.endswith(".tif") for f in result)

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.tif").write_bytes(b"x")
        (sub / "b.tif").write_bytes(b"x")
        pattern = str(tmp_path / "*.tif")
        result = rp.get_files_to_process2(pattern, search_subfolders=True)
        assert len(result) == 2

    def test_no_match_returns_empty(self, tmp_path):
        pattern = str(tmp_path / "*.nd2")
        result = rp.get_files_to_process2(pattern, search_subfolders=False)
        assert result == []

    def test_result_is_sorted(self, tmp_path):
        for name in ["c.tif", "a.tif", "b.tif"]:
            (tmp_path / name).write_bytes(b"x")
        pattern = str(tmp_path / "*.tif")
        result = rp.get_files_to_process2(pattern, search_subfolders=False)
        assert result == sorted(result)

    def test_uses_forward_slashes(self, tmp_path):
        (tmp_path / "img.tif").write_bytes(b"x")
        pattern = str(tmp_path / "*.tif")
        result = rp.get_files_to_process2(pattern, search_subfolders=False)
        for f in result:
            assert "\\" not in f


# ---------------------------------------------------------------------------
# get_grouped_files_to_process
# ---------------------------------------------------------------------------

class TestGetGroupedFilesToProcess:
    def _create_test_files(self, tmp_path):
        """Create paired image / mask files."""
        for name in ["img001", "img002", "img003"]:
            (tmp_path / f"{name}.tif").write_bytes(b"x")
            (tmp_path / f"{name}_mask.tif").write_bytes(b"x")

    def test_groups_paired_files(self, tmp_path):
        self._create_test_files(tmp_path)
        patterns = {
            "image": str(tmp_path / "*.tif"),
            "mask": str(tmp_path / "*_mask.tif"),
        }
        groups = rp.get_grouped_files_to_process(patterns, search_subfolders=False)
        # mask files extracted with anchor "_mask.tif"
        mask_groups = {k: v for k, v in groups.items() if "mask" in v}
        assert len(mask_groups) == 3

    def test_empty_patterns_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            rp.get_grouped_files_to_process({}, search_subfolders=False)

    def test_pattern_without_wildcard_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must contain a '\\*' wildcard"):
            rp.get_grouped_files_to_process(
                {"bad": str(tmp_path / "no_wildcard.tif")},
                search_subfolders=False,
            )

    def test_result_is_sorted(self, tmp_path):
        self._create_test_files(tmp_path)
        patterns = {"image": str(tmp_path / "*.tif")}
        groups = rp.get_grouped_files_to_process(patterns, search_subfolders=False)
        keys = list(groups.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# split_comma_separated_strstring / split_comma_separated_intstring
# ---------------------------------------------------------------------------

class TestSplitCommaSeparated:
    def test_str_split(self):
        assert rp.split_comma_separated_strstring("a,b,c") == ["a", "b", "c"]

    def test_str_single(self):
        assert rp.split_comma_separated_strstring("hello") == ["hello"]

    def test_int_split(self):
        assert rp.split_comma_separated_intstring("1,2,3") == [1, 2, 3]

    def test_int_single(self):
        assert rp.split_comma_separated_intstring("7") == [7]

    def test_int_invalid_raises(self):
        with pytest.raises(ValueError):
            rp.split_comma_separated_intstring("a,b")


# ---------------------------------------------------------------------------
# get_files_to_process (deprecated)
# ---------------------------------------------------------------------------

class TestGetFilesToProcessDeprecated:
    def test_deprecation_warning(self, tmp_path):
        (tmp_path / "img.tif").write_bytes(b"x")
        with pytest.warns(FutureWarning):
            rp.get_files_to_process(str(tmp_path), ".tif", search_subfolders=False)

    def test_finds_correct_files(self, tmp_path):
        (tmp_path / "a.tif").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rp.get_files_to_process(str(tmp_path), ".tif", search_subfolders=False)
        assert len(result) == 1
        assert result[0].endswith(".tif")
