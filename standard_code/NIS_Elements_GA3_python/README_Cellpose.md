# NIS-Elements GA3 Python Scripts for Cellpose

This folder contains scripts to run Cellpose segmentation from within NIS-Elements using the GA3 (General Analysis 3) Python interface.

## Available Scripts

### 1. `ga3_run_cellpose_NIS-Python.py`
**Uses**: NIS-Elements' built-in Python environment  
**Pros**: No external dependencies, simplest setup  
**Cons**: Version conflicts with NIS-Elements' numpy (1.23 vs 2.0+ required by latest cellpose)  
**Status**: ⚠️ Currently has numpy version incompatibility

### 2. `ga3_run_cellpose_UV-Python.py`
**Uses**: UV-managed virtual environment  
**Pros**: Fast, modern Python package management  
**Cons**: Missing Visual C++ redistributables for PyTorch CUDA on Windows  
**Status**: ⚠️ CPU-only works, GPU requires additional system dependencies

### 3. `ga3_run_cellpose_Conda-Python.py` ⭐ **RECOMMENDED**
**Uses**: Conda environment with full GPU support  
**Pros**: Complete dependency management including system libraries, proven working setup  
**Cons**: Larger download size, conda required  
**Status**: ✅ Fully working with GPU acceleration

## Setup Instructions

### Option 1: Conda Environment (Recommended for GPU)

1. **Create the environment** (one-time setup):
   ```powershell
   D:\biphub\Program_Files\Anaconda\Scripts\conda.exe env create -f d:\biphub\git\run_pipeline\conda_envs\cellpose3.yml -p D:\biphub\NIS-Python-envs\conda_cellpose3
   ```

2. **Load script in NIS-Elements**:
   - Open NIS-Elements
   - Go to GA3 interface
   - Load `ga3_run_cellpose_Conda-Python.py`
   - Set `USE_GPU = True` for GPU acceleration

3. **Configure parameters** in the script:
   - `SELECTED_MODEL`: Choose from available Cellpose models
   - `DIAMETER`: Cell diameter in pixels (most important!)
   - `USE_GPU`: Enable/disable GPU acceleration
   - Other segmentation parameters as needed

### Option 2: UV Environment (CPU-only, fast setup)

1. **Environment already created at**: `D:\biphub\NIS-Python-envs\venv_cellpose3`

2. **Use script**: `ga3_run_cellpose_UV-Python.py`

3. **Limitation**: CPU-only due to missing CUDA dependencies

## How It Works

All external environment scripts (`*UV-Python.py`, `*Conda-Python.py`) use the same approach:

1. **Subprocess Communication**: NIS-Elements GA3 Python spawns a separate Python process
2. **Binary Protocol**: Image data sent via stdin/stdout pipes (no file I/O overhead)
3. **Persistent Process**: Cellpose model loaded once, stays in memory between frames
4. **Performance**: ~100x faster than file-based approaches for batch processing

## Environment Locations

- **UV environments**: `D:\biphub\NIS-Python-envs\venv_*`
- **Conda environments**: `D:\biphub\NIS-Python-envs\conda_*`

## GPU Support

Your system:
- **GPU**: NVIDIA T1000 8GB
- **CUDA**: 12.7
- **Compatible**: PyTorch with CUDA 12.4 (forward compatible)

The conda environment installs PyTorch with `pytorch-cuda=12.4`, which is compatible with your CUDA 12.7 driver.

## Troubleshooting

### "Module not found" errors
- Verify environment path in script matches created environment
- Check that Python executable exists at the specified path

### Slow performance
- Enable `USE_GPU = True` if you have compatible GPU
- Check that CUDA is available: model will print status on first run

### Memory issues
- Reduce batch size or image dimensions
- Disable `RESAMPLE = False` for faster (but less accurate) results

## Model Selection

Available models (configure with `SELECTED_MODEL`):
- `cyto3` - Latest generalist cytoplasm model (recommended)
- `nuclei` - Specialized for nuclear segmentation
- `tissuenet` - For tissue samples
- `livecell` - For live cell imaging
- See full list in script comments

## Version Information

- **Cellpose**: 3.0.7 (conda) / 3.1.1.2 (UV)
- **PyTorch**: 2.4.0+cu124
- **Python**: 3.10 (conda) / 3.11 (UV)
