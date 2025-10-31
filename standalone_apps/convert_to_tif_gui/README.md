# BIPHUB Image Converter GUI

Standalone GUI application for converting microscopy images to OME-TIFF format.

## Features

- Supports all BioFormats-compatible file formats (CZI, LIF, ND2, LSM, etc.)
- User-friendly GUI powered by Gooey
- Preserves metadata and physical pixel dimensions
- Batch processing support

## Usage

Run the executable or:
```bash
python convert_to_tif_gui.py
```

## Building

To build the standalone executable:
```bash
pyinstaller --onefile --windowed convert_to_tif_gui.py
```
