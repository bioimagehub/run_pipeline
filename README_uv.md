# UV pilot usage

This repo now supports uv-managed segments for a gradual migration away from conda.

Pilot groups:
- convert_to_tif -> environment: uv:convert_to_tif
- segment_ernet   -> environment: uv:segment_ernet

Quick start (Windows PowerShell):

```powershell
# Install uv once (recommended via pipx)
pipx install uv

# Create lockfile and sync pilot groups
uv lock
uv sync --group convert_to_tif
uv sync --group segment_ernet

# Smoke tests
uv run --group convert_to_tif python -m standard_code.python.convert_to_tif --help
uv run --group segment_ernet python -m standard_code.python._segment_ernet --help

# Run the pilot pipeline YAML
# Example: go run . pipeline_configs/segment_ernet_uv.yaml
```

Notes:
- The runner detects `environment: uv:<group>` and uses `uv run --group <group> ...`.
- CUDA wheels are pinned via PyTorch's CUDA 12.4 index in `pyproject.toml`.
- Bio-Formats (.ims) reading will launch Java via scyjava on first use; provide a system JDK if you want to avoid auto-downloads.
