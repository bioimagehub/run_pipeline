# Pipeline Designer - BIPHUB Visual Pipeline Editor

A graphical user interface for designing and editing YAML pipeline configurations for the BIPHUB Pipeline Manager. This tool provides a node-based visual editor for ANY command-line program.

## Architecture

- **Backend**: Go (Wails v2) for YAML parsing, subprocess execution, and CLI definition management
- **Frontend**: React + TypeScript + React Flow for the visual node editor
- **State Management**: Zustand for efficient React state management
- **Styling**: Custom dark theme CSS inspired by VS Code

## Project Status

✅ **Phase 1 MVP - COMPLETED AND RUNNING**

**Build successful!** The application compiles and launches successfully.

**Executable location**: `build/bin/pipeline-designer.exe` (~15-20MB)

All core components have been implemented and tested:

### Backend (Go)
- ✅ `models.go` - Data structures for nodes, sockets, pipelines
- ✅ `yaml_parser.go` - YAML ↔ Pipeline conversion (compatible with run_pipeline.go)
- ✅ `cli_definitions.go` - CLI definition loader/manager
- ✅ `app.go` - Wails bindings for frontend-backend communication
- ✅ CLI definitions created:
  - `convert_to_tif.json`
  - `segment_ernet.json`
  - `mask_measure.json`

### Frontend (React + TypeScript)
- ✅ `App.tsx` - Main 3-panel layout (Command Explorer | Canvas | Properties)
- ✅ `Canvas.tsx` - React Flow canvas with pan/zoom/minimap
- ✅ `CommandExplorer.tsx` - Left sidebar with CLI tool library
- ✅ `PropertiesPanel.tsx` - Right sidebar for editing node sockets
- ✅ `CLINode.tsx` - Custom node component with input/output sockets
- ✅ `pipelineStore.ts` - Zustand state management
- ✅ `globals.css` - Complete dark theme
- ✅ `types.ts` - TypeScript type definitions

## Prerequisites

✅ **All prerequisites satisfied!**

### 1. Go (v1.24.0)
Already installed ✅

### 2. Node.js (v22.21.0) + npm (v10.9.4)
Already installed ✅

### 3. Wails CLI (v2.10.2)
Already installed ✅

### 4. TypeScript (v5.7.3)
Already installed ✅

## Installation

✅ **Already complete!** All dependencies installed and application built successfully.

To rebuild:

```bash
# Navigate to the project
cd pipeline-designer

# Install frontend dependencies
cd frontend
npm install reactflow zustand lucide-react

# Return to project root
cd ..

# Verify installation
wails doctor
```

## Development

### Quick Start - Run the Application
```powershell
# From pipeline-designer directory
.\build\bin\pipeline-designer.exe
```

The application should open in a native window with WebView2!

### Development Mode (with hot-reload)
```powershell
# From pipeline-designer directory
wails dev
```

⚠️ **Note**: `wails dev` may occasionally fail with exit code 0xc000013a (Windows access violation). This is a known Wails issue. If this happens:
- Use `wails build` and run the .exe directly
- Or restart your terminal and try again

This will:
1. Start the Go backend
2. Start the React frontend with hot-reload
3. Open the application in your default browser

### Build for Production
```bash
# From pipeline-designer directory
wails build
```

Output will be in `build/bin/`:
- Windows: `pipeline-designer.exe`
- Linux: `pipeline-designer`
- macOS: `pipeline-designer.app`

## Usage

### Launching the Designer

**Option 1**: Direct launch
```bash
./pipeline-designer.exe
```

**Option 2**: From run_pipeline (future integration)
```bash
./run_pipeline.exe --design pipeline_configs/segment_ernet_uv.yaml
```

### Creating a Pipeline

1. **Browse Commands** (Left Panel)
   - Expand categories (Segmentation, Image Processing, etc.)
   - Click on a command to add it to the canvas

2. **Arrange Nodes** (Center Canvas)
   - Drag nodes to position them
   - Zoom with Ctrl+Scroll
   - Pan with Space+Drag or Middle-mouse drag

3. **Connect Nodes** (Center Canvas)
   - Drag from an output socket (right side) to an input socket (left side)
   - Connections represent data flow

4. **Edit Properties** (Right Panel)
   - Click a node to select it
   - Edit socket values (file paths, parameters, etc.)
   - Connected sockets show "⚡ Connected" and are read-only

5. **Save Pipeline**
   - File → Save (future menu implementation)
   - Pipeline is saved as YAML compatible with run_pipeline.go

## Project Structure

```
pipeline-designer/
├── main.go                     # Wails app entry point
├── app.go                      # App struct with Wails methods
├── models.go                   # Go data structures
├── yaml_parser.go              # YAML conversion logic
├── cli_definitions.go          # CLI definition loader
├── cli_definitions/            # CLI tool definitions
│   ├── convert_to_tif.json
│   ├── segment_ernet.json
│   └── mask_measure.json
└── frontend/                   # React + TypeScript
    ├── src/
    │   ├── App.tsx             # Main 3-panel layout
    │   ├── components/         # React components
    │   ├── stores/             # Zustand state
    │   └── styles/             # Dark theme CSS
    └── package.json
```

## Next Steps

See `FRONTEND_SETUP.md` for detailed instructions once Node.js is installed.

## License

MIT License - Copyright (c) 2024 BIPHUB, University of Oslo
