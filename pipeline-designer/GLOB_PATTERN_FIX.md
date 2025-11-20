# Glob Pattern Fix - Recursive Search Support

## Problem

The pipeline-designer file counter was using Go's built-in `filepath.Glob()`, which **does not support** the `**` recursive pattern that Python's `glob.glob()` supports.

### Before:
- Pattern: `folder/**/*.tif` would only match files in `folder/*/*.tif` (one level deep)
- Go's `filepath.Glob` treats `**` as a literal directory name
- Inconsistent with Python's behavior in `bioimage_pipeline_utils.py`

### After:
- Pattern: `folder/**/*.tif` now matches files at **any depth** under `folder/`
- Uses `github.com/bmatcuk/doublestar/v4` library for proper `**` support
- Consistent with Python's `glob.glob(pattern, recursive=True)` behavior

## Changes Made

### 1. Updated `go.mod`
Added dependency:
```go
github.com/bmatcuk/doublestar/v4 v4.7.1
```

### 2. Updated `app.go`
- Added import: `"github.com/bmatcuk/doublestar/v4"`
- Modified `CountFilesMatchingPattern()` to use `doublestar.FilepathGlob()`
- Modified `resolveOutputGlobPatterns()` to use `doublestar.FilepathGlob()`

## Example Patterns Now Supported

| Pattern | Matches |
|---------|---------|
| `data/*.tif` | All `.tif` files in `data/` folder |
| `data/**/*.tif` | All `.tif` files in `data/` and any subfolders (recursive) |
| `%YAML%/**/*_mask.tif` | All mask files anywhere under YAML directory |
| `E:/experiment/**/results/*.csv` | All CSV files in any `results/` folder under `experiment/` |

## Python Equivalent

The Go implementation now matches Python's behavior:

```python
# Python (bioimage_pipeline_utils.py)
glob.glob('folder/**/*.tif', recursive=True)

# Go (pipeline-designer app.go)
doublestar.FilepathGlob('folder/**/*.tif')
```

Both will now find files at **any depth** under the `folder/` directory.

## Testing

Build and test:
```powershell
cd pipeline-designer
go build -o pipeline-designer.exe
```

The file counter in the UI should now correctly count files in deeply nested folders when using `**` patterns.

---

**Date:** 2025-01-20  
**Author:** BIPHUB - Bioimage Informatics Hub  
**Issue:** Glob pattern inconsistency between Go and Python implementations
