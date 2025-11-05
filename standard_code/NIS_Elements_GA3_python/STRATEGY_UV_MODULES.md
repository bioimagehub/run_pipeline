# Strategy: UV-Managed External Modules for NIS-Elements GA3

## Executive Summary

Integrate external Python modules (like Cellpose) into GA3 using isolated UV-managed environments, avoiding dependency conflicts with NIS-Elements' built-in Python while maintaining clean data exchange patterns.

**Note:** UV package manager is bundled in this repository at `external/UV/uv.exe` - no external installation required!

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  NIS-Elements GA3 (Built-in Python 3.12)                    │
│  ├─ GA3 Node (Coordinator)                                   │
│  │  └─ Uses: limnode, numpy, subprocess                     │
│  └─ Handles: UI, workflow, data routing                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ [Image Data Serialization]
                 │ • numpy array → .npy temp file
                 │ • OR shared memory for speed
                 │ • Parameters via JSON/CLI args
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  UV-Managed External Environment (.venv per module)         │
│  ├─ pyproject.toml (isolated dependencies)                  │
│  ├─ uv.lock (reproducible versions)                         │
│  └─ worker_script.py                                         │
│     ├─ Loads image from temp file                           │
│     ├─ Runs Cellpose/other module                           │
│     └─ Writes results back                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. **One pyproject.toml Per Module**
Each external tool gets its own isolated environment:
```
standard_code/python/
├── cellpose_module/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .venv/
│   └── cellpose_worker.py
├── stardist_module/
│   ├── pyproject.toml
│   └── stardist_worker.py
└── other_module/
```

**Benefits:**
- ✅ No dependency conflicts between modules
- ✅ Reproducible via uv.lock
- ✅ Easy to version control
- ✅ Fast environment creation with UV
- ✅ Can specify exact versions per module

### 2. **Data Exchange Patterns**

#### **Option A: File-Based (Recommended for GA3)**
```python
# GA3 Coordinator (built-in Python)
def run(inp, out, ctx):
    # Save input
    np.save("temp/input.npy", inp[0].data[0, :, :, 0])
    
    # Call UV environment
    subprocess.run([
        ".venv/Scripts/python.exe", 
        "cellpose_worker.py",
        "--input", "temp/input.npy",
        "--output", "temp/output.npy",
        "--diameter", "30"
    ])
    
    # Load result
    out[0].data[0, :, :, 0] = np.load("temp/output.npy")
```

**Why File-Based?**
- Simple to implement
- Works across process boundaries
- Easy to debug (inspect intermediate files)
- Handles large images efficiently with .npy format
- Can use Zarr for multi-dimensional data

#### **Option B: Shared Memory (Advanced)**
For high-frequency calls, use `multiprocessing.shared_memory`:
```python
# Share memory handle + metadata
shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
# External process accesses same memory
```

### 3. **Auto-Bootstrap & Environment Detection**

GA3 node can automatically set up the UV environment on first use:

```python
def ensure_cellpose_env():
    """Ensure cellpose UV environment exists, create if missing."""
    venv_path = Path(__file__).parent / "cellpose_module" / ".venv"
    
    if not venv_path.exists():
        logging.info("Setting up Cellpose environment (one-time)...")
        subprocess.run(["uv", "venv", str(venv_path)])
        subprocess.run([
            "uv", "pip", "install",
            "-e", str(venv_path.parent)
        ])
        logging.info("Cellpose environment ready!")
    
    return venv_path / "Scripts" / "python.exe"
```

## Implementation Strategy

### Phase 1: Proof of Concept (Cellpose)

Create minimal working example:

1. **Module Structure:**
```
cellpose_module/
├── pyproject.toml          # Dependencies
├── cellpose_worker.py      # Worker script
└── ga3_cellpose_node.py    # GA3 coordinator
```

2. **GA3 Node Template:**
```python
# ga3_cellpose_node.py
import limnode
import numpy as np
import subprocess
import tempfile
from pathlib import Path

CELLPOSE_PYTHON = ensure_cellpose_env()

def output(inp, out):
    out[0].makeNew("cellpose_masks", (0, 255, 255)).makeInt32()

def run(inp, out, ctx):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.npy"
        output_path = Path(tmpdir) / "output.npy"
        
        # Save input
        np.save(input_path, inp[0].data[0, :, :, 0])
        
        # Call external worker
        result = subprocess.run([
            str(CELLPOSE_PYTHON),
            "cellpose_worker.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", "cyto3",
            "--diameter", "30"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Cellpose failed: {result.stderr}")
        
        # Load result
        masks = np.load(output_path)
        out[0].data[0, :, :, 0] = masks.astype(np.int32)
```

3. **Worker Script:**
```python
# cellpose_worker.py
import argparse
import numpy as np
from cellpose import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="cyto3")
    parser.add_argument("--diameter", type=float, default=30)
    args = parser.parse_args()
    
    # Load image
    image = np.load(args.input)
    
    # Run Cellpose
    model = models.Cellpose(model_type=args.model)
    masks, flows, styles, diams = model.eval(image, diameter=args.diameter)
    
    # Save result
    np.save(args.output, masks)

if __name__ == "__main__":
    main()
```

### Phase 2: Generalize Pattern

Create reusable base classes:

```python
# standard_code/python/external_module_base.py
class ExternalModuleNode:
    """Base class for UV-managed external modules in GA3."""
    
    def __init__(self, module_name: str, venv_path: Path):
        self.module_name = module_name
        self.venv_path = venv_path
        self.python_exe = self._ensure_environment()
    
    def _ensure_environment(self) -> Path:
        """Auto-bootstrap UV environment if needed."""
        # Implementation from above
        pass
    
    def call_worker(self, worker_script: str, **kwargs) -> dict:
        """Call external worker with standardized interface."""
        # Handles temp files, subprocess, error handling
        pass
```

### Phase 3: Integration with run_pipeline

Your existing YAML pipeline configs can orchestrate these:

```yaml
# pipeline_configs/segment_cellpose_uv.yaml
pipeline_name: "Cellpose Segmentation (UV)"
description: "Segment cells using isolated Cellpose environment"

steps:
  - name: "Load Images"
    module: "bioimage_pipeline_utils"
    function: "load_tczyx_image"
    inputs:
      path: "%INPUT_DIR%/*.tif"
    outputs:
      - name: "images"
  
  - name: "Run Cellpose (External)"
    module: "ga3_cellpose_node"  # Uses UV-managed Cellpose
    inputs:
      images: "steps.Load Images.outputs.images"
      model_type: "cyto3"
      diameter: 30
    outputs:
      - name: "masks"
  
  - name: "Save Results"
    module: "bioimage_pipeline_utils"
    function: "save_tczyx_image"
    inputs:
      image: "steps.Run Cellpose.outputs.masks"
      path: "%OUTPUT_DIR%/masks.tif"
```

## Addressing Your Concerns

### "Is the idea sound?"

**Yes, with refinements:**

✅ **Good aspects of your idea:**
- Isolation prevents dependency hell
- UV is fast and modern
- Reproducible environments
- Scales to multiple modules

⚠️ **Key refinements needed:**
- Use file-based communication (not trying to share Python objects)
- GA3's built-in Python acts as coordinator
- External venv does the heavy lifting
- Keep interface simple (numpy arrays in/out)

### "How do we communicate images?"

**Best approach for GA3:**
1. **Serialize to .npy files** (fast, native numpy)
2. **Use temp directory** (automatic cleanup)
3. **Pass file paths + parameters** via CLI args
4. **Load results back** into GA3 node

**For larger/complex data:**
- Use **Zarr** for multi-dimensional chunked arrays
- Use **Memory-mapped files** for zero-copy access
- Use **Named pipes** for streaming (advanced)

## Performance Considerations

### File I/O Overhead
```python
# Typical timings for 2048x2048 image:
np.save()       # ~10ms
Subprocess call # ~100-200ms (Python startup)
Cellpose        # ~2-10s (GPU/CPU)
np.load()       # ~5ms
```

For most bioimage workflows, subprocess overhead is negligible compared to analysis time.

### Optimization Strategies
1. **Keep worker process alive** (server mode)
2. **Batch multiple images** in one call
3. **Use shared memory** for high-frequency calls
4. **Pre-warm environments** at GA3 startup

## Migration Path

### Immediate (Now)
1. Create proof-of-concept with Cellpose
2. Test with real GA3 workflows
3. Document the pattern

### Short-term (Next Month)
1. Generalize to base class
2. Add 2-3 more modules (StarDist, DeepImageJ, etc.)
3. Add auto-bootstrap logic

### Long-term (Future)
1. Create GA3 node generator tool
2. Build library of UV-managed modules
3. Integrate with run_pipeline orchestrator
4. Consider server/daemon mode for hot-reload

## Alternative: Hybrid Approach

For some modules, you might want a **hybrid** where:
- Simple modules → Install in NIS-Elements Python (current way)
- Complex/conflicting modules → UV-managed external process (new way)

This gives you flexibility based on each tool's needs.

## Conclusion

Your idea is **sound and elegant**! The key insight is:

> **Don't try to make NIS-Elements Python import from external venv.**  
> **Instead, use subprocess + file exchange as a clean boundary.**

This architecture:
- ✅ Works with GA3's existing infrastructure
- ✅ Provides complete dependency isolation
- ✅ Is reproducible and portable
- ✅ Scales to many modules
- ✅ Leverages UV's speed advantages
- ✅ Maintains debugging ability

**Next Steps:**
1. I can create a working proof-of-concept for you
2. We can test it in your GA3 environment
3. Then generalize the pattern for other modules

Would you like me to implement the Cellpose proof-of-concept now?
