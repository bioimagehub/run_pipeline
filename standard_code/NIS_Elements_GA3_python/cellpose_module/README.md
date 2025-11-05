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

### 1. Test the Module (Recommended)

Before using in GA3, verify everything works:

```powershell
cd c:\git\run_pipeline\standard_code\NIS_Elements_GA3_python\cellpose_module
python test_cellpose_module.py
```

This will:
- ✅ Initialize the GA3Adapter
- ✅ Auto-create the UV environment (first time only, ~1-2 minutes)
- ✅ Run a test segmentation
- ✅ Verify everything is ready for GA3

### 2. Use in NIS-Elements GA3

**Step 1:** Open GA3 editor and add Python node
- `ND Processing & Conversions > Python Scripting > Python`

**Step 2:** Configure inputs/outputs
- Add 1 **Channel input** (your microscopy image)
- Add 1 **Binary output** (segmentation masks)

**Step 3:** Copy the code
- Open `ga3_cellpose_node.py`
- Copy **entire contents** (Ctrl+A, Ctrl+C)
- Paste into GA3 Python node

**Step 4:** Update the path
```python
ADAPTER_PATH = r'C:\git\run_pipeline\standard_code\NIS_Elements_GA3_python'
```
Change to your actual installation location.

**Step 5:** Adjust parameters (optional)
```python
MODEL = "cyto3"          # cyto, cyto2, cyto3, nuclei
DIAMETER = None          # None for auto, or specify pixels
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
```

**Step 6:** Enable "Run out of process"
- Check this option in the node settings

**Step 7:** Run!
- Connect your workflow and execute
- First run may take 1-2 minutes (environment setup)
- Subsequent runs are instant

## File Structure

```
NIS_Elements_GA3_python/
├── ga3adapter.py                    # Universal adapter (reusable)
└── cellpose_module/
    ├── pyproject.toml               # UV dependencies
    ├── .venv/                       # Virtual environment (auto-created)
    ├── cellpose_worker.py           # Cellpose-specific logic
    ├── ga3_cellpose_node.py         # Code to paste into GA3 (~30 lines)
    ├── test_cellpose_module.py      # Test script
    └── README.md                    # This file
```

## How It Works

### The Pattern

```python
# User pastes this simple code into GA3:
from ga3adapter import GA3Adapter

adapter = GA3Adapter(
    module_dir="cellpose_module",
    worker_script="cellpose_worker.py"
)

def run(inp, out, ctx):
    image = inp[0].data[0, :, :, 0]
    masks = adapter.process(image, model="cyto3", diameter=None)
    out[0].data[0, :, :, 0] = masks.astype('int32')
```

### Behind the Scenes

1. **GA3Adapter** handles all orchestration:
   - Environment management (UV, .venv)
   - Subprocess coordination
   - Temp file handling
   - Error logging

2. **Worker script** does the actual work:
   - Loads Cellpose in isolated environment
   - Runs segmentation
   - Returns results

3. **Data exchange** via temporary numpy files:
   - Clean, simple interface
   - Easy to debug
   - Language-agnostic

### Why This Pattern?

**Problem:** NIS-Elements Python can't install Cellpose (numpy compatibility issues)

**Solution:** Run in isolated UV environment, coordinate via subprocess

**Benefits:**
- ✅ **Simple GA3 code** (~30 lines vs ~200 lines)
- ✅ **Reusable adapter** for any external Python package
- ✅ **No dependency conflicts** (isolated environment)
- ✅ **Reproducible** (UV lock files)
- ✅ **Easy to maintain** (code in git, not copy-pasted)

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

The **GA3Adapter is universal** - reuse it for any external Python package!

### Example: Adding StarDist

1. Create new module directory:
```
stardist_module/
├── pyproject.toml           # StarDist dependencies
├── stardist_worker.py       # StarDist segmentation logic
└── ga3_stardist_node.py     # ~30 lines for GA3
```

2. Update `pyproject.toml`:
```toml
[project]
name = "ga3-stardist"
dependencies = ["stardist>=0.8.0", "tensorflow>=2.0"]
```

3. GA3 code stays simple:
```python
from ga3adapter import GA3Adapter

adapter = GA3Adapter(
    module_dir="stardist_module",
    worker_script="stardist_worker.py"
)

def run(inp, out, ctx):
    image = inp[0].data[0, :, :, 0]
    masks = adapter.process(image, model="2D_versatile_fluo")
    out[0].data[0, :, :, 0] = masks.astype('int32')
```

**That's it!** The adapter handles everything else.

## References

- [Cellpose Documentation](https://cellpose.readthedocs.io/)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [GA3 Python Scripting Docs](https://nis-elements.github.io/)
- [BIPHUB Pipeline Manager](https://github.com/bioimagehub/run_pipeline)

## License

MIT License - See repository root for full license text.
