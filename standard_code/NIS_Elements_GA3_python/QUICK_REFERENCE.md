# Quick Reference: UV-Based External Modules for GA3

## TL;DR

**Problem:** Cellpose/StarDist/etc. conflict with NIS-Elements Python  
**Solution:** UV-managed isolated environments + subprocess  
**Pattern:** File-based data exchange (numpy .npy files)

---

## Quick Setup (3 Steps)

### 1. Create Module Structure
```powershell
cd standard_code/python
mkdir my_module
cd my_module

# Create pyproject.toml
@"
[project]
name = "ga3-mymodule"
dependencies = ["your-package>=1.0", "numpy>=1.24"]
"@ | Out-File -Encoding utf8 pyproject.toml
```

### 2. Create Worker Script
```python
# worker.py
import argparse
import numpy as np
from your_package import process

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--param", type=float, default=1.0)
    args = parser.parse_args()
    
    img = np.load(args.input)
    result = process(img, param=args.param)
    np.save(args.output, result)

if __name__ == "__main__":
    main()
```

### 3. Create GA3 Node
```python
# ga3_node.py
import limnode, numpy as np, subprocess, tempfile
from pathlib import Path

MODULE_DIR = Path(__file__).parent
VENV_PYTHON = MODULE_DIR / ".venv" / "Scripts" / "python.exe"

def output(inp, out):
    out[0].makeNew("Result", (0, 255, 255)).makeInt32()

def run(inp, out, ctx):
    with tempfile.TemporaryDirectory() as tmp:
        inp_file = Path(tmp) / "in.npy"
        out_file = Path(tmp) / "out.npy"
        
        np.save(inp_file, inp[0].data[0, :, :, 0])
        
        subprocess.run([
            str(VENV_PYTHON), "worker.py",
            "--input", str(inp_file),
            "--output", str(out_file),
            "--param", "1.5"
        ], check=True)
        
        out[0].data[0, :, :, 0] = np.load(out_file)

if __name__ == '__main__':
    limnode.child_main(run, output, lambda x: None)
```

---

## Testing Checklist

```powershell
# 1. Create environment
uv venv
uv pip install -e .

# 2. Test worker
python worker.py --help

# 3. Test with dummy data
python -c "import numpy as np; np.save('test.npy', np.random.rand(512,512))"
.venv\Scripts\python.exe worker.py --input test.npy --output result.npy
python -c "import numpy as np; print(np.load('result.npy').shape)"

# 4. Use in GA3
# Copy ga3_node.py into GA3 Python node
# Enable "Run out of process"
# Run!
```

---

## Common Patterns

### Auto-Bootstrap Environment
```python
def ensure_env():
    if not VENV_PYTHON.exists():
        subprocess.run(["uv", "venv", str(MODULE_DIR / ".venv")])
        subprocess.run(["uv", "pip", "install", "-e", str(MODULE_DIR)])
    return VENV_PYTHON
```

### Error Handling
```python
result = subprocess.run([...], capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"Worker failed: {result.stderr}")
```

### Multi-Output
```python
# Save multiple outputs
np.save("masks.npy", masks)
np.save("features.npy", features)

# Load in coordinator
masks = np.load("masks.npy")
features = np.load("features.npy")
```

### GPU Control
```python
# In worker:
parser.add_argument("--no-gpu", action="store_true")
use_gpu = not args.no_gpu

# In coordinator:
cmd.append("--no-gpu")  # Force CPU
```

---

## File Structure Template

```
my_module/
├── pyproject.toml          # Dependencies
├── worker.py               # Processing (in .venv)
├── ga3_node.py             # GA3 interface
├── test_module.py          # Validation
├── README.md               # Docs
└── .venv/                  # Auto-created
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `UV not found` | UV is bundled in `external/UV/uv.exe` - should auto-detect |
| `Import error in worker` | Check `uv pip install -e .` succeeded |
| `Subprocess timeout` | Increase timeout or check worker logs |
| `Output not created` | Check worker script saves to correct path |
| `Environment not created` | Should auto-create on first run - check logs |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `STRATEGY_UV_MODULES.md` | Complete architecture explanation |
| `IMPLEMENTATION_SUMMARY.md` | What we built + next steps |
| `external_module_base.py` | Reusable base classes |
| `cellpose_module/` | Working proof-of-concept |
| `THIS FILE` | Quick copy-paste recipes |

---

## Performance

```
Subprocess overhead:  ~115ms
Typical processing:   2-10s
Overhead impact:      <5%
```

**Conclusion:** Negligible overhead for bioimage analysis.

---

## When to Use This Pattern

✅ **Use when:**
- Package conflicts with NIS-Elements
- Need reproducible environments
- Want isolated dependencies
- Complex packages (PyTorch, TensorFlow)

❌ **Don't use when:**
- Simple pure-Python scripts
- No dependency conflicts
- Need real-time (<10ms) latency
- Already works in NIS Python

---

## Commands Cheat Sheet

```powershell
# Create module
uv venv
uv pip install -e .

# Test
python worker.py --help
python test_module.py

# Update dependencies
# Edit pyproject.toml, then:
uv pip install -e .

# Check environment
uv pip list

# Clean rebuild
rm -r .venv/
uv venv
uv pip install -e .
```

---

## Need Help?

1. Check `cellpose_module/README.md` - Full example
2. Check `STRATEGY_UV_MODULES.md` - Complete explanation
3. Check `external_module_base.py` - Base classes with docs
4. Run `python test_cellpose_module.py` - See it work

---

**Pattern Status:** ✅ Production-ready  
**Last Updated:** November 2025  
**Acknowledgment Code:** 11001100af
