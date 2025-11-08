# ImageJ Integration Guide

This repository supports **two approaches** for running ImageJ macros and scripts:

## Approach 1: Direct ImageJ Command-Line (Recommended for Windows)

**Status:** ✅ Working  
**File:** `pipeline_configs/segment_imagej_threshold.yaml`  
**Environment:** `imagej` (direct ImageJ/Fiji executable)

### Configuration Example:
```yaml
run:
  - name: Segment images using ImageJ threshold
    environment: imagej
    commands:
      - '%REPO%/standard_code/imagej/simple_treshold.ijm'
      - input: '%SARAH_DATA%/cellprofiler_input'
      - suffix: '.tif'
      - minsize: 35000
      - maxsize: 120000
```

### How It Works:
1. The Go orchestrator detects `environment: imagej`
2. Automatically uses ImageJ executable from `.env` file (`IMAGEJ_PATH`)
3. Detects file extension:
   - `.ijm` files → Uses `-macro` flag
   - `.py`/`.js`/`.groovy` files → Uses `--ij2 --run` flags
4. Converts YAML map syntax to ImageJ parameter format
5. Handles path conversion (backslashes → forward slashes)

### Advantages:
- No Python dependencies
- Direct ImageJ execution
- Works with existing ImageJ/Fiji installation
- Fast startup
- Proven stable on Windows

### Limitations:
- Some ImageJ plugins have limited headless mode support
- Dialog-based operations may fail
- Error messages can be cryptic

---

## Approach 2: PyImageJ (Python Wrapper)

**Status:** ⚠️ Experimental (headless mode issues on Windows)  
**File:** `pipeline_configs/segment_imagej_threshold_pyimagej.yaml`  
**Environment:** `uv@3.11:imagej` (Python with PyImageJ)

### Configuration Example:
```yaml
run:
  - name: Segment images using ImageJ (via PyImageJ)
    environment: uv@3.11:imagej
    commands:
      - python
      - '%REPO%/standard_code/python/run_imagej_script.py'
      - --script: '%REPO%/standard_code/imagej/simple_treshold.ijm'
      - --input: '%SARAH_DATA%/cellprofiler_input'
      - --suffix: '.tif'
      - --minsize: 35000
      - --maxsize: 120000
```

### How It Works:
1. Uses Python wrapper script (`run_imagej_script.py`)
2. PyImageJ initializes ImageJ JVM
3. Runs macros/scripts through Python API
4. Flexible argument passing via argparse

### Advantages:
- Better error handling
- Python integration
- Can manipulate ImageJ data structures directly
- Cross-platform (theoretically)

### Limitations:
- **JVM crashes in headless mode on Windows** (current blocker)
- Slow first-time startup (downloads ImageJ if needed)
- Requires Java 8/11
- Complex dependency chain

### Installation:
```bash
# Install PyImageJ dependencies
uv sync --group imagej

# Test
.\run_pipeline.exe pipeline_configs/segment_imagej_threshold_pyimagej.yaml
```

---

## Choosing an Approach

| Feature | Direct ImageJ | PyImageJ |
|---------|---------------|----------|
| Windows Stability | ✅ Excellent | ⚠️ Issues |
| Setup Complexity | ✅ Simple | ❌ Complex |
| Startup Time | ✅ Fast | ❌ Slow |
| Python Integration | ❌ None | ✅ Full |
| Error Handling | ⚠️ Basic | ✅ Good |
| Headless Mode | ⚠️ Limited | ⚠️ Problematic |

**Recommendation:** Use **Direct ImageJ** (Approach 1) for production pipelines on Windows.

---

## Writing ImageJ Macros for Headless Mode

### Best Practices:

1. **Avoid GUI-dependent functions:**
   ```javascript
   // ❌ Bad - requires GUI
   waitForUser("Click OK to continue");
   
   // ✅ Good - works headless
   print("Processing...");
   ```

2. **Use batch mode:**
   ```javascript
   setBatchMode(true);  // Suppresses image display
   // Your processing code
   setBatchMode(false);
   ```

3. **Accept arguments via getArgument():**
   ```javascript
   args = getArgument();
   // Parse: "input=/path,suffix=.tif,minsize=35000"
   ```

4. **Test manually first:**
   ```powershell
   # Test macro in headless mode
   "C:\Fiji.app\ImageJ-win64.exe" --headless --console -macro test.ijm "input=/data"
   ```

---

## Troubleshooting

### Direct ImageJ Issues:

**Problem:** "Cannot instantiate headless dialog"  
**Solution:** Remove interactive commands from macro, use batch mode

**Problem:** Bio-Formats errors  
**Solution:** Use `IJ.openImage()` instead of `run("Bio-Formats")`

### PyImageJ Issues:

**Problem:** JVM crash in headless mode  
**Solution:** Currently no fix on Windows - use Direct ImageJ instead

**Problem:** "Can't find org.jpype.jar"  
**Solution:** Use `jpype1==1.5.0` (already configured in pyproject.toml)

---

## File Locations

- **ImageJ macros:** `standard_code/imagej/*.ijm`
- **ImageJ2 scripts:** `standard_code/imagej/*.py`
- **PyImageJ wrapper:** `standard_code/python/run_imagej_script.py`
- **Pipeline configs:** `pipeline_configs/segment_imagej_*.yaml`
- **Go orchestrator:** `run_pipeline.go` (handles ImageJ command building)

---

## Future Work

- [ ] Fix PyImageJ headless mode on Windows
- [ ] Add ImageJ2 script examples (.py, .js, .groovy)
- [ ] Create hybrid approach (Python pre/post-processing + ImageJ core)
- [ ] Document all available ImageJ environment variables
- [ ] Add automatic ImageJ download/setup script
