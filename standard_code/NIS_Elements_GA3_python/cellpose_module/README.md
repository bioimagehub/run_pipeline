# Cellpose Module for NIS-Elements GA3

UV-managed Cellpose integration for GA3 using isolated Python environments.

## Overview

This module demonstrates the **UV-based external module pattern** for integrating complex Python packages into GA3 without dependency conflicts.

### Architecture

```
GA3 (Built-in Python 3.12)
    ↓
ga3_cellpose_node.py (Coordinator)
    ↓ [numpy .npy files]
    ↓
UV-Managed Environment (.venv)
    ↓
cellpose_worker.py (Worker)
    ↓
Cellpose Library
```

## Quick Start

### 1. Installation

The environment is **auto-created on first use** using the bundled UV package manager (located in `external/UV/uv.exe`).

**No manual installation required!** The system will automatically:
- Use the bundled UV executable (no PATH setup needed)
- Create an isolated `.venv` directory
- Install Cellpose and dependencies

### 2. Testing (Recommended)

Before using in GA3, test the module:

```powershell
cd standard_code/NIS_Elements_GA3_python/cellpose_module
python test_cellpose_module.py
```

This will verify the bundled UV works and create the environment.

### 3. Usage in GA3

1. Open NIS-Elements and go to GA3 editor
2. Insert a Python node: `ND Processing & Conversions > Python Scripting > Python`
3. Configure node:
   - Add 1 **Channel input** (your microscopy image)
   - Add 1 **Binary output** (segmentation masks)
4. Open node settings (`...` button)
5. Paste contents of `ga3_cellpose_node.py`
6. **Important:** Enable "Run out of process"
7. Connect to your workflow and run!

On first run, the node will:
- Detect no environment exists
- Use UV to create `.venv/`
- Install Cellpose (1-2 minutes)
- Run segmentation

Subsequent runs use the cached environment and start instantly!

### 3. Configuration

Edit these parameters in `ga3_cellpose_node.py`:

```python
CELLPOSE_MODEL = "cyto3"  # cyto, cyto2, cyto3, nuclei
CELL_DIAMETER = None      # None for auto, or specify pixels
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
```

## File Structure

```
cellpose_module/
├── pyproject.toml              # UV dependencies
├── uv.lock                     # Locked versions (generated)
├── .venv/                      # Virtual environment (auto-created)
├── cellpose_worker.py          # Worker script (runs in .venv)
├── ga3_cellpose_node.py        # GA3 node (runs in NIS Python)
└── README.md                   # This file
```

## How It Works

### Data Flow

1. **GA3 Input**: User connects microscopy image to node
2. **Serialization**: `ga3_cellpose_node.py` saves image to temp `.npy` file
3. **Subprocess Call**: Spawns `.venv/Scripts/python.exe cellpose_worker.py`
4. **Worker Process**: Loads image, runs Cellpose, saves masks
5. **Deserialization**: `ga3_cellpose_node.py` loads masks back
6. **GA3 Output**: Masks appear in workflow for measurement/analysis

### Why This Pattern?

**Problem:** Cellpose has dependencies (PyTorch, etc.) that conflict with NIS-Elements' built-in Python.

**Solution:** Run Cellpose in isolated environment, exchange data via files.

**Benefits:**
- ✅ No dependency conflicts
- ✅ Reproducible (UV lock file)
- ✅ Fast environment creation
- ✅ Easy to debug (inspect temp files)
- ✅ Works with any external Python library

## Advanced Usage

### Manual Environment Setup

```powershell
# Navigate to module directory
cd standard_code/python/cellpose_module

# Create environment
uv venv

# Install dependencies
uv pip install -e .

# Test worker directly
.venv\Scripts\python.exe cellpose_worker.py --help
```

### Testing Worker Standalone

```powershell
# Create test image
python -c "import numpy as np; np.save('test_img.npy', np.random.rand(512, 512).astype('float32'))"

# Run segmentation
.venv\Scripts\python.exe cellpose_worker.py `
    --input test_img.npy `
    --output test_masks.npy `
    --model cyto3 `
    --diameter 30

# Check result
python -c "import numpy as np; masks = np.load('test_masks.npy'); print(f'Found {masks.max()} cells')"
```

### Adding Custom Models

To use custom-trained Cellpose models:

1. Place model file (`.npy` or `.pth`) in `cellpose_module/models/`
2. Modify `cellpose_worker.py`:
```python
from cellpose.models import CellposeModel

model = CellposeModel(
    gpu=use_gpu,
    pretrained_model='path/to/your/model'
)
```

### GPU Acceleration

The module uses GPU by default if CUDA is available.

**To force CPU-only:**
```python
# In ga3_cellpose_node.py, modify call_cellpose_worker():
cmd.append("--no-gpu")
```

**To check GPU status:**
```powershell
.venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Environment Creation Fails

**Error:** `UV not found`
```powershell
# Install UV
pip install uv
```

**Error:** `Failed to download packages`
- Check internet connection
- Try: `uv pip install cellpose` manually in the module directory

### Segmentation Fails

**Check logs:** GA3 shows logs in Python stdout/stderr

**Common issues:**
- Image too large → Resize before passing to Cellpose
- Wrong model type → Try different models (cyto vs nuclei)
- Out of memory → Reduce image size or disable GPU

**Debug manually:**
```powershell
# Save failing image from GA3, then:
.venv\Scripts\python.exe cellpose_worker.py `
    --input failing_image.npy `
    --output debug_masks.npy `
    --model cyto3
```

### Performance

**Slow first run:** Environment creation takes 1-2 minutes (one-time)

**Slow subsequent runs:** 
- Most time is Cellpose processing (~2-10s per image)
- Subprocess overhead is minimal (~100ms)

**Optimization:**
- Use GPU (10-20x faster than CPU)
- Batch process multiple images
- Consider server mode (keep process alive)

## Extending to Other Modules

This pattern works for any Python package! To create a new module:

1. Copy this directory structure:
```
new_module/
├── pyproject.toml           # Change dependencies
├── worker_script.py         # Change processing logic
└── ga3_node_script.py       # Change GA3 interface
```

2. Update `pyproject.toml`:
```toml
[project]
name = "ga3-your-module"
dependencies = [
    "your-package>=1.0",
]
```

3. Follow the same data exchange pattern (numpy files + subprocess)

## References

- [Cellpose Documentation](https://cellpose.readthedocs.io/)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [GA3 Python Scripting Docs](https://nis-elements.github.io/)
- [BIPHUB Pipeline Manager](https://github.com/bioimagehub/run_pipeline)

## License

MIT License - See repository root for full license text.
