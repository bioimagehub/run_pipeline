# Bio-Formats Support - Conda vs UV

## TL;DR

- **Most formats work with UV**: TIFF, ND2, LIF, CZI, DV (pure Python readers)
- **Exotic formats need Conda**: Use `conda_envs/convert_to_tif.yml` for Bio-Formats support

## Why Conda for Bio-Formats?

Bio-Formats is a Java library that requires:
1. Java Runtime Environment (JRE/JDK)
2. `jpype` (Python-Java bridge)
3. `scyjava` (SciJava Python wrapper)
4. `bioio-bioformats` (BioIO reader plugin)

**The problem with UV**: When you install these packages via UV/pip, they don't include Java. You need to install Java separately, and jpype may crash with incompatible Java versions (e.g., Java 17 causes access violations with jpype 1.6.0).

**Why Conda works**: When you install `scyjava` via conda-forge, it automatically bundles a compatible Java runtime and configures everything correctly.

## Supported Formats

### ✅ Work with UV (no Java needed)
- **TIFF** (.tif, .tiff) - via `bioio-tifffile` and `bioio-ome-tiff`
- **ND2** (.nd2) - via `bioio-nd2`
- **LIF** (.lif) - via `bioio-lif`
- **CZI** (.czi) - via `bioio-czi`
- **DV** (.dv) - via `bioio-dv`
- **Imaris** (.ims) - via custom `bioio_imaris` reader (fallback to Bio-Formats)

### ⚠️ Require Conda (need Bio-Formats/Java)
- **OIB/OIF** (.oib, .oif) - Olympus formats
- **VSI** (.vsi) - Olympus slide scanner
- **SCN** (.scn) - Leica slide scanner  
- **MRXS** (.mrxs) - 3DHISTECH slide scanner
- **Exotic/Legacy formats** - Various proprietary formats

## Usage

### Option 1: UV Environment (Fast, most formats)

```yaml
# In your pipeline_configs/your_pipeline.yaml
run:
  - name: Convert images
    environment: uv@3.11:convert-to-tif  # UV environment
    commands:
      - python
      - '%REPO%/standard_code/python/convert_to_tif.py'
      - --input-search-pattern: 'data/*.nd2'
      - --output-folder: 'output'
```

### Option 2: Conda Environment (Full format support)

```bash
# Create Conda environment (one-time setup)
conda env create -f conda_envs/convert_to_tif.yml
```

```yaml
# In your pipeline_configs/your_pipeline.yaml
run:
  - name: Convert images with Bio-Formats
    environment: convert_to_tif  # Conda environment
    commands:
      - python
      - '%REPO%/standard_code/python/convert_to_tif.py'
      - --input-search-pattern: 'data/*.oib'
      - --output-folder: 'output'
```

## Error Handling

If you try to open an exotic format with UV, you'll see:

```
======================================================================
ERROR: Failed to load your_file.oib
======================================================================

This file format requires Bio-Formats (Java), which failed to initialize.
Original error: [jpype error details]

SOLUTION: Use Conda environment instead of UV for Bio-Formats support:

1. Create Conda environment from: conda_envs/convert_to_tif.yml
   conda env create -f conda_envs/convert_to_tif.yml

2. Run your pipeline with the Conda environment:
   run_pipeline.exe pipeline_configs/your_config.yaml
   (use 'environment: convert_to_tif' in your YAML config)

NOTE: Most formats (ND2, LIF, CZI, DV, TIFF) work without Bio-Formats.
      Only exotic formats require the Conda environment.
======================================================================
```

## Technical Details

### Why jpype crashes with UV + Java 17

- jpype 1.6.0 (latest on PyPI) has compatibility issues with Java 17
- When jpype tries to load `jvm.dll`, it causes an access violation (0xC0000005)
- This happens during import, before Python can catch the exception
- Conda's scyjava bundles Java 11, which is fully compatible with jpype

### Alternative (not recommended)

You could manually install Java 11 and configure JAVA_HOME, but:
- Conda is easier and more reproducible
- Java version mismatches cause hard-to-debug crashes
- Conda guarantees all dependencies are compatible

##  When in doubt, use Conda!
