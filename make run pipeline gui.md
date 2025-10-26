# Run Pipeline GUI Design Document

## 10-Second Elevator Pitch 🚀
**"A visual node-based editor for designing command-line workflows - like Unreal Engine's Blueprints, but for ANY command-line program. Design pipelines visually with socket-based connections, test individual nodes with sample files, then export to YAML for production runs with run_pipeline.exe."**

## Overview
A graphical user interface (GUI) for designing and editing YAML pipeline configurations for the BIPHUB Pipeline Manager. This GUI provides a **universal node-based visual editor** for ANY command-line program, treating CLI arguments as input/output sockets that can be connected between nodes.

**Architecture**: Go backend (YAML parsing, subprocess execution) + Web frontend (React Flow for node editor) connected via Wails framework. This leverages battle-tested UI components while keeping robust Go execution logic.

## Core Innovation: CLI Arguments as Sockets
The breakthrough concept: **CLI programs don't naturally have "outputs" like traditional node systems, but we can treat arguments containing "output" patterns as output sockets**. This creates a visual programming paradigm where:
- **Input arguments** (--input-folder, --source, etc.) appear as **left-side sockets**
- **Output arguments** (--output-folder, --destination, etc.) appear as **right-side sockets**  
- **Connections** between nodes represent data flow (output folder → input folder)
- **Text values** can be typed directly into unconnected sockets
- **Type validation** ensures compatible connections (path→path, string→string)

This makes the system **100% generic** - any CLI tool can become a visual node!

## Project Goals
1. **Universal CLI Orchestration**: Work with ANY command-line program (Python, ImageJ, ffmpeg, etc.)
2. **Robust Implementation**: Use Go with a proven GUI framework for cross-platform compatibility
3. **Modern Dark UI**: Implement a dark theme consistent with modern development tools (Unreal, Houdini, Blender)
4. **Visual Pipeline Design**: Provide an infinite canvas for visual pipeline construction (similar to OneNote/node editors). This program is inspired by KNIME, OneNote, NIS Elements GA3, and Unreal Engine Blueprints.
5. **Intelligent Socket System**: Auto-classify arguments as inputs/outputs with user override capability
6. **CLI Definition Library**: Reusable templates for CLI tools with argument specifications
7. **Interactive Testing**: Test individual nodes with selected files before full pipeline execution
8. **Seamless Integration**: Launch from run_pipeline.exe with `-d` or `--design` flag
9. **File Path Management**: Integrate the glob pattern logic from `get_files_to_process2` for intuitive file selection

## Architecture

### Technology Stack
**Chosen: Wails v2 + React Flow** ✅

**Why This Stack:**
- **Wails v2**: Go backend + web frontend in single executable
- **React Flow**: Battle-tested node editor (50k+ GitHub stars)
  - Built-in pan/zoom, minimap, connection validation
  - Auto-layout algorithms included
  - Accessibility and keyboard navigation
  - Undo/redo system built-in
- **Go Backend**: YAML parsing, subprocess execution, file watching
- **React + TypeScript Frontend**: Modern, type-safe UI development
- **No Electron**: Wails uses native webview (smaller binary, faster startup)

**Key Benefits:**
- 80% of node editor features already implemented
- Professional UI out of the box
- Rapid development (3 weeks instead of 7)
- Easy to extend with React ecosystem (charts, image viewers, etc.)
- Cross-platform with native performance

### Project Structure
```
pipeline-designer/
├── main.go                       # Wails app entry point
├── app.go                        # Application struct with methods
├── wails.json                    # Wails configuration
├── go.mod / go.sum               # Go dependencies
│
├── backend/                      # Go backend
│   ├── yaml_parser.go            # YAML ↔ JSON conversion
│   ├── cli_definitions.go        # CLI definition loader/saver
│   ├── subprocess.go             # Test execution (single-file mode)
│   ├── file_watcher.go           # Monitor output files (fsnotify)
│   ├── validation.go             # Socket type validation, security checks
│   ├── help_parser.go            # Parse --help output
│   └── models.go                 # Shared data structures (Pipeline, Node, Socket)
│
├── frontend/                     # React + TypeScript
│   ├── src/
│   │   ├── App.tsx               # Main application component
│   │   ├── main.tsx              # React entry point
│   │   ├── wailsjs/              # Auto-generated Wails bindings
│   │   │   ├── go/               # Go struct types
│   │   │   └── runtime/          # Wails runtime functions
│   │   ├── components/
│   │   │   ├── Canvas.tsx        # React Flow canvas wrapper
│   │   │   ├── CommandExplorer.tsx   # Left panel: CLI library
│   │   │   ├── FileSelector.tsx  # Left panel: file browser
│   │   │   ├── NodeProperties.tsx    # Right panel: socket editor
│   │   │   ├── TestConsole.tsx   # Test output display
│   │   │   └── CustomNode.tsx    # CLI node component
│   │   ├── nodes/
│   │   │   └── CLINode.tsx       # Node renderer with sockets
│   │   ├── hooks/
│   │   │   ├── useUndo.ts        # Undo/redo hook
│   │   │   └── useFileWatch.ts   # File watching hook
│   │   ├── stores/
│   │   │   └── pipelineStore.ts  # Zustand state management
│   │   └── types/
│   │       └── pipeline.ts       # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
├── cli_definitions/              # CLI tool definition templates
│   ├── convert_to_tif.json
│   ├── segment_ernet.json
│   ├── mask_measure.json
│   └── custom/                   # User-created definitions
│
└── build/                        # Wails build output
    ├── bin/                      # Compiled executables
    └── windows/linux/darwin/     # Platform-specific resources
```

### Integration with run_pipeline.go
Add new command-line flag handling:
```go
// In main() function
if hasFlag("-d", "--design") {
    // Launch Wails GUI
    LaunchDesignerGUI(yamlPath)
    return
}

func LaunchDesignerGUI(yamlPath string) {
    // Wails will be in separate binary: pipeline-designer.exe
    cmd := exec.Command("pipeline-designer.exe", "--file", yamlPath)
    cmd.Start()
}
```

**Note**: Wails apps are best as separate executables due to frontend bundling. `run_pipeline.exe` can launch `pipeline-designer.exe` with the YAML path as argument.

## UI Layout Specification

### Window Configuration
- **Title**: "BIPHUB Pipeline Designer"
- **Default Size**: 1400x900 pixels
- **Minimum Size**: 1024x768 pixels
- **Theme**: Dark mode (custom dark theme)

### Layout Structure (3-Column Split)

#### Left Panel: Command Explorer (20% width, ~280px)
**Purpose**: Library of available pipeline commands/modules

**Components**:
- Search/filter bar at top
- Categorized tree view of commands:
  - **Segmentation**
    - segment_ernet
    - segment_threshold
    - segment_ilastik
    - segment_nellie
  - **Image Processing**
    - convert_to_tif
    - merge_channels
    - drift_correction
  - **Measurement**
    - mask_measure
    - mask_measure_hierarchical
  - **Tracking**
    - track_indexed_mask
  - **Visualization**
    - mask2imageJROI
    - plot_mask
  - **Utilities**
    - extract_metadata
    - copy_files_with_extension

**Interaction**:
- Drag-and-drop commands onto canvas
- Double-click to add to canvas at center
- Right-click for command documentation

#### Center Panel: Infinite Canvas (60% width, ~840px)
**Purpose**: Visual pipeline construction workspace with node-based programming

**Features**:
- **Infinite Scrollable Canvas**: Pan with middle-mouse drag or space+drag
- **Zoom**: Ctrl+Scroll or pinch gesture (0.25x to 4x range)
- **Grid Background**: Subtle dot grid for alignment (toggleable)
- **Node Representation**: Each CLI tool as a visual node with sockets
  - **Structure:**
    ```
         Input Sockets (Left)         Node Body          Output Sockets (Right)
    ┌────────────────────┬──────────────────────────────┬────────────────────┐
    │                    │  🔄 Convert to TIF      [≡]  │                    │
    │ ◄──● input-pattern ├──────────────────────────────┤  output-folder ●──►│
    │     [./input/*.ims]│  uv@3.11:convert-to-tif      │  [./output/tifs]   │
    │                    │  python convert_to_tif.py    │                    │
    │ ◄──● compression   │                              │                    │
    │     [6]            │                              │                    │
    └────────────────────┴──────────────────────────────┴────────────────────┘
    ```
  - Rounded rectangle with drop shadow
  - Color-coded by category (see color palette below)
  - Input sockets on left side (◄●), output sockets on right side (●►)
  - Inline text fields for unconnected sockets
  - Title bar with icon, name, and settings menu [≡]
  - Collapsible to show only title (for large pipelines)
  
- **Socket Rendering**:
  - **Connected socket**: Filled circle (●) with glowing effect
  - **Unconnected socket**: Empty circle (○) with text field
  - **Required but empty**: Red outline warning
  - **Type indicators**: Color-coded by argument type
    - Path: Blue
    - GlobPattern: Cyan
    - String: Gray
    - Int/Float: Green
    - Bool: Orange
  
- **Connection Lines**: Bezier curves connecting output→input sockets
  - Smooth curves that avoid overlapping nodes
  - Color matches socket type
  - Animated flow indicator (optional, subtle dots moving along line)
  - Hover highlighting
  - Click to select/delete
  - Invalid connections shown in red with tooltip explaining incompatibility
  
- **Interaction**:
  - **Drag socket to socket**: Create connection (output→input only)
  - **Click node**: Select (shows properties in right panel)
  - **Ctrl+Click**: Multi-select nodes
  - **Drag node**: Move on canvas
  - **Double-click node**: Toggle expand/collapse
  - **Right-click node**: Context menu (duplicate, delete, edit definition)
  - **Right-click connection**: Delete connection
  - **Space+Drag** or **Middle-mouse drag**: Pan canvas
  - **Ctrl+Scroll**: Zoom in/out
  
- **Canvas Controls**:
  - Mini-map in bottom-right corner (shows all nodes, current viewport)
  - Zoom level indicator (25%, 50%, 100%, 200%, 400%)
  - Reset view button (fit all nodes to viewport)
  - Grid toggle button
  - Auto-layout button (arrange nodes in logical flow)

**Node States**:
- Normal: Dark gray background (#2d2d2d)
- Selected: Blue outline (#007acc), thicker border
- Hover: Lighter background (#3e3e3e)
- Error: Red outline (#f48771) with error icon
- Running: Pulsing blue border animation
- Completed: Green checkmark badge in top-right corner (from status file)
- Disabled: Grayed out with 50% opacity

#### Right Panel: Properties Panel (20% width, ~280px)
**Purpose**: Edit socket values, node configuration, and test with sample files

**Components**:

**Section 1: File Selection Widget** (Always visible at top)
Dynamic file browser showing files matching the current pipeline's input pattern:

```
┌─────────────────────────────────────────────────────┐
│ FILE SELECTION                                      │
├─────────────────────────────────────────────────────┤
│ Pattern: [./input/*.tif                        ]    │
│ 📁 Browse  🔍 Search Subfolders [✓]                │
├─────────────────────────────────────────────────────┤
│ Matched Files: (234 found)                         │
│ ┌─────────────────────────────────────────────────┐ │
│ │ ● sample_001.tif              (2.3 MB)         │ │  ← Selected for testing
│ │ ○ sample_002.tif              (2.1 MB)         │ │
│ │ ○ sample_003.tif              (2.4 MB)         │ │
│ │ ○ experiment_A_01.tif         (3.1 MB)         │ │
│ │ ○ experiment_A_02.tif         (2.9 MB)         │ │
│ │ ... (229 more)                                  │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Selected: sample_001.tif                            │
└─────────────────────────────────────────────────────┘
```

**Purpose**:
- Browse files matching the pipeline's input glob pattern
- Select ONE file for testing individual nodes
- Show file count and basic info (size, date)
- Selected file is highlighted and used by "Run Node" buttons

**Section 2: Socket Value Editor** (Visible when node selected)
Dynamic form showing all sockets for the selected node:

```
┌─────────────────────────────────────────────────────┐
│ SOCKET CONFIGURATION: Convert to TIF                │
├─────────────────────────────────────────────────────┤
│ INPUT SOCKETS:                                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │ ◄ --input-search-pattern                        │ │
│ │   [./input/*.ims                            ]   │ │
│ │   Type: GlobPattern  ✓ Required                │ │
│ │   📁 Browse  🔍 Preview (234 files found)      │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ ◄ --compression                                 │ │
│ │   [6                                        ]   │ │
│ │   Type: Int  Range: 0-9                        │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ OUTPUT SOCKETS:                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │ --output-folder ►                               │ │
│ │   [./output/tifs                            ]   │ │
│ │   Type: Path  ✓ Required                       │ │
│ │   📁 Browse  ⚡ Connected (1 node)             │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │          ▶️ RUN NODE (Test Mode)                │ │
│ │                                                  │ │
│ │  Will execute with: sample_001.tif               │ │
│ │  Environment: uv@3.11:convert-to-tif            │ │
│ │                                                  │ │
│ │  [▶️ Run with Selected File]                    │ │
│ │  [📋 View Generated Command]                    │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**New "Run Node" Section Features:**
- **▶️ Run Button**: Execute THIS node only with selected file
- **File Preview**: Shows which file will be used
- **Command Preview**: View exact command that will run
- **Output Console**: Show stdout/stderr from execution
- **Status Indicator**: Running/Success/Failed with execution time

**Features:**
- Text fields for unconnected sockets (editable)
- **Connection indicator** for connected sockets (shows "⚡ Connected to [NodeName]")
- **Type-specific widgets**:
  - Path/GlobPattern: Browse button + file preview
  - Int: Number spinner with validation
  - Bool: Checkbox
  - String: Text field
- **Inline validation**: Red border + error message for invalid values
- **Real-time preview**: For glob patterns, show matched files count
- **Path resolution indicators**: Show %REPO%, %YAML% tokens with resolved preview

**Section 3: Node Properties** (Visible when node selected)
```
┌─────────────────────────────────────────────────────┐
│ NODE PROPERTIES                                     │
├─────────────────────────────────────────────────────┤
│ Name: [Convert to TIF                          ]    │
│ Category: [Image Processing               ▼]       │
│ Icon: [🔄]  Color: [#569cd6] [Pick]                │
├─────────────────────────────────────────────────────┤
│ EXECUTION CONFIGURATION:                            │
│ Environment: [uv@3.11:convert-to-tif       ▼]      │
│   Options: Base, UV environments, Conda envs        │
│ Executable: [python                        ▼]      │
│ Script: [📁 Browse] convert_to_tif.py              │
├─────────────────────────────────────────────────────┤
│ [Edit CLI Definition...]                            │
│ [Discover Arguments from --help]                    │
└─────────────────────────────────────────────────────┘
```

**Section 3: CLI Definition Editor** (Opens in dialog)
When user clicks "Edit CLI Definition" or adds a new tool:
```
┌─────────────────────────────────────────────────────┐
│ CLI TOOL DEFINITION: Convert to TIF                 │
├─────────────────────────────────────────────────────┤
│ Name: [Convert to TIF              ]                │
│ Category: [Image Processing    ▼]                   │
│ Icon: [🔄]  Color: [#569cd6]                        │
├─────────────────────────────────────────────────────┤
│ Executable: [python            ▼]                   │
│ Script: [Browse...] convert_to_tif.py               │
│ Environment: [uv@3.11:convert-to-tif]               │
│ Help Command: [python script.py --help]             │
├─────────────────────────────────────────────────────┤
│ ARGUMENTS:                                          │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Flag: --input-search-pattern                    │ │
│ │ Type: [glob_pattern ▼]                          │ │
│ │ Socket: [✓ ◄ Input] [   ► Output] [  ◄► Both] │ │
│ │ Required: [✓]  Default: [./input/*.ims]        │ │
│ │ Description: [Glob pattern for input files]     │ │
│ │ Validation: [must_exist               ▼]       │ │
│ │ [X Remove] [↑ Move Up] [↓ Move Down]            │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ Flag: --output-folder                           │ │
│ │ Type: [path ▼]                                  │ │
│ │ Socket: [   ◄ Input] [✓ ► Output] [  ◄► Both] │ │
│ │ Required: [✓]  Default: [./output]             │ │
│ │ Description: [Output folder for TIF files]      │ │
│ │ Validation: [create_if_missing    ▼]           │ │
│ │ [X Remove] [↑ Move Up] [↓ Move Down]            │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ + Add Argument Manually                         │ │
│ │ 🔍 Discover from --help Output                  │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ [💾 Save Definition] [Cancel] [⚠️ Delete Definition]│
└─────────────────────────────────────────────────────┘
```

**Key Features:**
- **Socket Classification Radio Buttons**: User can override auto-classification
- **Argument Reordering**: Move up/down to control display order
- **Type Dropdown**: path, glob_pattern, string, int, float, bool
- **Validation Rules**: must_exist, create_if_missing, range:min-max, regex:pattern
- **Help Parser Button**: Auto-discover arguments from `--help` output

## Data Model

### CLI Node with Socket System + Test Execution
```go
type CLINode struct {
    ID              string              // Unique node instance ID (UUID)
    DefinitionID    string              // Reference to CLI definition template
    Name            string              // Display name (user can override)
    Position        Point               // Canvas coordinates (X, Y)
    Size            Size                // Node dimensions (Width, Height)
    
    // Execution configuration
    Environment     string              // "uv@3.11:convert-to-tif", "base", "conda:env_name"
    Executable      string              // "python", "imagej", etc.
    Script          string              // Path to script file
    
    // Socket system
    InputSockets    []Socket            // Left-side sockets
    OutputSockets   []Socket            // Right-side sockets
    
    // Visual state
    IsSelected      bool
    IsCollapsed     bool
    Category        string              // For color coding
    
    // Test execution state (NEW)
    TestStatus      TestStatus          // NotRun, Running, Success, Failed
    LastTestFile    string              // Last file used for testing
    LastTestOutput  string              // Captured stdout/stderr
    LastTestTime    time.Duration       // Execution duration
    TestError       error               // Error if test failed
}

type TestStatus int
const (
    TestNotRun  TestStatus = iota  // Haven't run this node yet
    TestRunning                     // Currently executing
    TestSuccess                     // Last test passed
    TestFailed                      // Last test failed
)

type Socket struct {
    ID              string              // Unique socket ID (UUID)
    NodeID          string              // Parent node ID
    ArgumentFlag    string              // "--input-folder", "--output-path", etc.
    Type            ArgumentType        // Path, GlobPattern, String, Int, Bool, Float
    SocketSide      SocketSide          // Input (left) or Output (right)
    
    // Value handling
    Value           string              // Direct text value (if not connected)
    ConnectedTo     *SocketConnection   // Connection to another socket (nil if unconnected)
    
    // Metadata
    IsRequired      bool
    DefaultValue    string
    Description     string
    Validation      string              // "must_exist", "range:0-9", "create_if_missing"
}

type SocketSide int
const (
    SocketInput  SocketSide = iota  // Left side of node
    SocketOutput                     // Right side of node
)

type ArgumentType int
const (
    TypePath         ArgumentType = iota  // File or directory path
    TypeGlobPattern                       // Glob pattern like "*.tif"
    TypeString                            // Generic string
    TypeInt                               // Integer number
    TypeFloat                             // Floating point number
    TypeBool                              // Boolean flag
)

type SocketConnection struct {
    ID              string              // Unique connection ID
    FromNodeID      string              // Source node
    FromSocketID    string              // Source socket (must be Output)
    ToNodeID        string              // Destination node
    ToSocketID      string              // Destination socket (must be Input)
    
    // Visual properties
    Color           color.Color         // Connection line color
    IsValid         bool                // Type compatibility check result
}

type NodeStatus int
const (
    StatusPending  NodeStatus = iota
    StatusRunning
    StatusSuccess
    StatusFailed
)
```

## Interactive Node Testing System (NEW!)

### Purpose
Allow developers to test individual nodes during pipeline design without running the full pipeline. This is a **development/debugging** feature, not production execution.

### Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT WORKFLOW                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Design Node                                              │
│     ├─ Drag "Convert to TIF" onto canvas                     │
│     ├─ Configure arguments                                   │
│     └─ Set input/output folders                              │
│                                                              │
│  2. Select Test File                                         │
│     ├─ Browse file list in right panel                       │
│     ├─ Click on "sample_001.tif"                             │
│     └─ File is highlighted                                   │
│                                                              │
│  3. Run Node Test                                            │
│     ├─ Click ▶️ "Run with Selected File" button             │
│     ├─ GUI generates command with SINGLE file                │
│     ├─ Executes command, captures output                     │
│     └─ Shows result in console pane                          │
│                                                              │
│  4. Verify & Iterate                                         │
│     ├─ Check output files created correctly                  │
│     ├─ Read stdout/stderr for errors                         │
│     ├─ Adjust arguments if needed                            │
│     └─ Re-test with same or different file                   │
│                                                              │
│  5. Save to YAML                                             │
│     ├─ Once satisfied, save pipeline                         │
│     └─ Use run_pipeline.exe for full batch processing        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Command Generation Logic

```go
// GenerateTestCommand creates a command for single-file testing
func (node *CLINode) GenerateTestCommand(selectedFile string) (*exec.Cmd, error) {
    // Build command parts
    var cmdParts []string
    
    // 1. Environment wrapper (if needed)
    if strings.HasPrefix(node.Environment, "uv@") {
        // Parse: "uv@3.11:convert-to-tif" → Python 3.11, group convert-to-tif
        parts := strings.Split(node.Environment, ":")
        uvGroup := parts[1]
        cmdParts = append(cmdParts, "uv", "run", "--group", uvGroup, "--")
    }
    
    // 2. Executable + script
    cmdParts = append(cmdParts, node.Executable, node.Script)
    
    // 3. Arguments with SINGLE FILE substitution
    for _, socket := range node.InputSockets {
        value := socket.GetResolvedValue()
        
        // Special handling for input patterns - replace with selected file
        if socket.Type == TypeGlobPattern || socket.Type == TypePath {
            if strings.Contains(socket.ArgumentFlag, "input") {
                // Override glob pattern with single file path
                value = selectedFile
            }
        }
        
        cmdParts = append(cmdParts, socket.ArgumentFlag+": "+value)
    }
    
    for _, socket := range node.OutputSockets {
        value := socket.GetResolvedValue()
        cmdParts = append(cmdParts, socket.ArgumentFlag+": "+value)
    }
    
    // Build exec.Cmd
    cmd := exec.Command(cmdParts[0], cmdParts[1:]...)
    cmd.Dir = filepath.Dir(node.Script)  // Set working directory
    
    return cmd, nil
}

// RunTest executes the node with the selected file
func (node *CLINode) RunTest(selectedFile string) error {
    node.TestStatus = TestRunning
    node.LastTestFile = selectedFile
    
    cmd, err := node.GenerateTestCommand(selectedFile)
    if err != nil {
        node.TestStatus = TestFailed
        node.TestError = err
        return err
    }
    
    // Capture stdout/stderr
    var stdout, stderr bytes.Buffer
    cmd.Stdout = &stdout
    cmd.Stderr = &stderr
    
    // Run command
    startTime := time.Now()
    err = cmd.Run()
    node.LastTestTime = time.Since(startTime)
    
    // Store output
    node.LastTestOutput = stdout.String() + "\n--- STDERR ---\n" + stderr.String()
    
    if err != nil {
        node.TestStatus = TestFailed
        node.TestError = err
        return err
    }
    
    node.TestStatus = TestSuccess
    return nil
}
```

### UI Components for Testing

**1. Run Button on Node (Canvas)**
```
┌──────────────────────────────────────────────────────┐
│ 🔄 Convert to TIF                    [▶️] [≡]       │  ← Run button in title bar
├──────────────────────────────────────────────────────┤
│ ◄──● input-pattern                output-folder ●──►│
│     [./input/*.ims]               [./output/tifs]   │
└──────────────────────────────────────────────────────┘
        ↓ Click ▶️ triggers test with selected file
```

**2. Test Output Console (Right Panel, expandable)**
```
┌─────────────────────────────────────────────────────┐
│ TEST OUTPUT                                [Clear]  │
├─────────────────────────────────────────────────────┤
│ Running: python convert_to_tif.py                   │
│   --input-search-pattern: sample_001.tif            │
│   --output-folder: ./output/tifs                    │
│                                                     │
│ --- STDOUT ---                                      │
│ Processing: sample_001.tif                          │
│ Output: ./output/tifs/sample_001.tif                │
│ ✓ Conversion successful                             │
│                                                     │
│ --- STDERR ---                                      │
│ (no errors)                                         │
│                                                     │
│ ✅ Completed in 2.34s                               │
└─────────────────────────────────────────────────────┘
```

**3. Status Indicator on Node**
```
Node States After Testing:

✅ Success (green badge):
┌──────────────────────────────────────────────────────┐
│ 🔄 Convert to TIF              [✅ 2.34s] [▶️] [≡]  │
│ ...                                                  │

❌ Failed (red badge):
┌──────────────────────────────────────────────────────┐
│ 🔄 Convert to TIF              [❌ Error] [▶️] [≡]  │
│ ...                                                  │

⏳ Running (pulsing animation):
┌──────────────────────────────────────────────────────┐
│ 🔄 Convert to TIF              [⏳ Running] [⏹] [≡] │
│ ...                                                  │
```

### Key Differences: Test Mode vs Production Mode

| Feature | Test Mode (GUI) | Production Mode (run_pipeline.exe) |
|---------|-----------------|-------------------------------------|
| **Input Files** | SINGLE selected file | ALL files matching glob pattern |
| **Purpose** | Verify node works correctly | Process entire dataset |
| **Execution** | Interactive, per-node | Batch, entire pipeline |
| **Output** | Real-time console feedback | Log files + status YAML |
| **Environment** | GUI spawns subprocess | run_pipeline orchestrates |
| **Error Handling** | Show immediately in GUI | Continue/stop based on config |

### Example Test Workflow

**Scenario**: Testing a segmentation node

```
1. User selects "sample_001.tif" in right panel file list

2. User clicks ▶️ on "Segment ERnet" node

3. GUI generates command:
   ```bash
   uv run --group segment-ernet -- python segment_ernet.py \
     --input-folder: sample_001.tif \
     --output-folder: ./output/masks \
     --model-path: weights/ernet.pth
   ```

4. GUI executes command, shows live output:
   ```
   Loading model: weights/ernet.pth
   Processing: sample_001.tif (512x512x10 voxels)
   Segmenting...
   Saving: ./output/masks/sample_001_mask.tif
   ✓ Done in 3.45s
   ```

5. User verifies output file was created correctly

6. User adjusts threshold parameter, clicks ▶️ again to re-test

7. Once satisfied, saves YAML for production run
```

### Security & Safety

**Important Constraints**:
- Test mode does NOT modify pipeline YAML status files
- Test mode does NOT skip segments marked as completed
- Test mode runs in ISOLATED subprocess (no state pollution)
- Output files from test runs are REAL (not temporary) - user should use test output folders
- Long-running tests can be cancelled with ⏹ (Stop) button

**Security Validation (NEW)**:
```go
type SecurityPolicy struct {
    AllowedExecutables []string  // Whitelist: python, imagej, etc.
    MaxExecutionTime   time.Duration  // Default: 5 minutes
    RequireConfirmation bool     // First-run confirmation dialog
}

func ValidateCommand(node *CLINode, policy SecurityPolicy) error {
    // 1. Check executable whitelist
    if !contains(policy.AllowedExecutables, node.Executable) {
        return fmt.Errorf("executable not in whitelist: %s", node.Executable)
    }
    
    // 2. Check for shell injection patterns
    dangerousPatterns := []string{";", "&&", "|", "`", "$(", ">", "<"}
    for _, socket := range node.InputSockets {
        for _, pattern := range dangerousPatterns {
            if strings.Contains(socket.Value, pattern) {
                return fmt.Errorf("potentially dangerous pattern in argument: %s", pattern)
            }
        }
    }
    
    // 3. Validate paths exist (for required inputs)
    for _, socket := range node.InputSockets {
        if socket.IsRequired && socket.Type == TypePath {
            if _, err := os.Stat(socket.Value); os.IsNotExist(err) {
                return fmt.Errorf("required input path does not exist: %s", socket.Value)
            }
        }
    }
    
    return nil
}

func ExecuteWithTimeout(cmd *exec.Cmd, timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    
    return cmd.Run(ctx)
}
```

### File Watching for Live Reload (NEW)
**Feature**: Automatically detect when test execution creates/modifies output files.

```go
import "github.com/fsnotify/fsnotify"

type FileWatcher struct {
    watcher *fsnotify.Watcher
    callbacks map[string]func(string)  // path → callback
}

func (fw *FileWatcher) WatchOutputFolder(outputPath string, callback func(string)) error {
    // Add directory to watch list
    if err := fw.watcher.Add(outputPath); err != nil {
        return err
    }
    
    fw.callbacks[outputPath] = callback
    
    // Listen for events
    go func() {
        for {
            select {
            case event := <-fw.watcher.Events:
                if event.Op&fsnotify.Create == fsnotify.Create {
                    // File was created - trigger callback
                    if cb, ok := fw.callbacks[filepath.Dir(event.Name)]; ok {
                        cb(event.Name)
                    }
                }
            case err := <-fw.watcher.Errors:
                log.Printf("file watcher error: %v", err)
            }
        }
    }()
    
    return nil
}
```

**UI Integration**:
- When test completes, watch output folder
- Show notification: "Output file created: sample_001_mask.tif"
- Auto-refresh image preview if viewer is open
- Highlight new files in file browser

### Benefits of This Approach

✅ **Rapid Iteration**: Test changes immediately without full pipeline  
✅ **Debugging**: See stdout/stderr in real-time  
✅ **Safe Development**: Test with single file before processing entire dataset  
✅ **Pipeline Design**: Verify each step works before connecting nodes  
✅ **Learning Tool**: Understand what each CLI command actually does  
✅ **Production Ready**: Once tested, full pipeline with run_pipeline.exe  



### CLI Definition Template
CLI definitions are stored as JSON files that describe how to use a command-line tool:

```go
type CLIDefinition struct {
    // Identity
    ID              string              // Unique identifier (filename without .json)
    Name            string              // Display name
    Category        string              // "Segmentation", "Image Processing", etc.
    Description     string              // Brief description
    
    // Visual properties
    Color           string              // Hex color for node (#569cd6)
    Icon            string              // Emoji or icon character (🔄)
    
    // Execution
    Executable      string              // "python", "imagej", etc.
    Script          string              // Relative path to script
    Environment     string              // Default environment
    
    // Arguments
    Arguments       []ArgumentDefinition
    
    // Discovery
    HelpCommand     string              // Command to run for --help parsing
    LastParsed      time.Time           // Last time --help was parsed
}

type ArgumentDefinition struct {
    Flag            string              // "--input-folder"
    Type            string              // "path", "glob_pattern", "string", "int", "bool", "float"
    SocketSide      string              // "input", "output", "both"
    IsRequired      bool
    DefaultValue    string
    Description     string
    Validation      string              // Validation rule
    
    // Auto-classification override
    UserOverride    bool                // True if user manually set SocketSide
}
```

### Example CLI Definition: `convert_to_tif.json`
```json
{
  "id": "convert_to_tif",
  "name": "Convert to TIF",
  "category": "Image Processing",
  "executable": "python",
  "script": "./standard_code/python/convert_to_tif.py",
  "environment": "uv@3.11:convert-to-tif",
  "description": "Convert various image formats to TIFF",
  "color": "#569cd6",
  "icon": "🔄",
  "arguments": [
    {
      "flag": "--input-search-pattern",
      "type": "glob_pattern",
      "socket_side": "input",
      "required": true,
      "default": "./input/*.ims",
      "description": "Glob pattern for input files",
      "validation": "must_exist",
      "user_override": false
    },
    {
      "flag": "--output-folder",
      "type": "path",
      "socket_side": "output",
      "required": true,
      "default": "./output",
      "description": "Folder to save converted TIF files",
      "validation": "create_if_missing",
      "user_override": false
    },
    {
      "flag": "--compression",
      "type": "int",
      "socket_side": "input",
      "required": false,
      "default": "6",
      "description": "Compression level (0-9)",
      "validation": "range:0-9",
      "user_override": false
    }
  ],
  "help_command": "python ./standard_code/python/convert_to_tif.py --help",
  "last_parsed": "2025-01-25T10:30:00Z"
}
```

### Pipeline Structure
```go
type Pipeline struct {
    Nodes           []CLINode
    Connections     []SocketConnection
    Metadata        PipelineMetadata
}

// Undo/Redo System (NEW)
type Command interface {
    Execute() error
    Undo() error
    Description() string
}

type CommandHistory struct {
    commands    []Command
    currentIdx  int
    maxHistory  int  // Default: 100
}

func (h *CommandHistory) ExecuteCommand(cmd Command) error {
    // Truncate future if we're in the middle of history
    h.commands = h.commands[:h.currentIdx+1]
    
    // Execute command
    if err := cmd.Execute(); err != nil {
        return err
    }
    
    // Add to history
    h.commands = append(h.commands, cmd)
    h.currentIdx++
    
    // Limit history size
    if len(h.commands) > h.maxHistory {
        h.commands = h.commands[1:]
        h.currentIdx--
    }
    
    return nil
}

func (h *CommandHistory) Undo() error {
    if h.currentIdx < 0 {
        return fmt.Errorf("nothing to undo")
    }
    err := h.commands[h.currentIdx].Undo()
    if err == nil {
        h.currentIdx--
    }
    return err
}

func (h *CommandHistory) Redo() error {
    if h.currentIdx >= len(h.commands)-1 {
        return fmt.Errorf("nothing to redo")
    }
    h.currentIdx++
    return h.commands[h.currentIdx].Execute()
}

// Example command implementations
type AddNodeCommand struct {
    pipeline *Pipeline
    node     CLINode
}

func (c *AddNodeCommand) Execute() error {
    c.pipeline.Nodes = append(c.pipeline.Nodes, c.node)
    return nil
}

func (c *AddNodeCommand) Undo() error {
    // Remove last node
    c.pipeline.Nodes = c.pipeline.Nodes[:len(c.pipeline.Nodes)-1]
    return nil
}

func (c *AddNodeCommand) Description() string {
    return fmt.Sprintf("Add node: %s", c.node.Name)
}

type PipelineMetadata struct {
    Name            string
    Description     string
    Version         string
    Author          string
    Created         time.Time
    Modified        time.Time
}
```

### YAML Serialization
Convert between visual node representation and YAML format:

**Visual Nodes:**
```
┌────────────────────────────────┐          ┌────────────────────────────────┐
│ 🔄 Convert to TIF              │          │ 🔬 Segment with ERnet          │
├────────────────────────────────┤          ├────────────────────────────────┤
│ input-pattern              ●   │          │ input-folder              ●────►
│ [./input/*.ims]                │          │                                │
│                                │          │                                │
│ output-folder              ●───┼──────────┼──●                             │
│ [./output/tifs]                │          │                                │
└────────────────────────────────┘          │ output-folder             ●────►
                                            │ [./output/masks]               │
                                            └────────────────────────────────┘
```

**Resulting YAML:**
```yaml
run:
  - name: "Convert to TIF"
    environment: "uv@3.11:convert-to-tif"
    commands:
      - python
      - ./standard_code/python/convert_to_tif.py
      - --input-search-pattern: './input/*.ims'
      - --output-folder: './output/tifs'
  
  - name: "Segment with ERnet"
    environment: "uv@3.11:segment-ernet"
    commands:
      - python
      - ./standard_code/python/segment_ernet.py
      - --input-folder: './output/tifs'        # ← Connected from previous node's output
      - --output-folder: './output/masks'
```

**Key Points:**
- Connected sockets: value is resolved from source node's output socket
- Unconnected sockets: value is taken from text field
- Execution order: determined by node connections (topological sort)
- Type validation: ensures output→input type compatibility

## Argument Classification System

### Explicit Classification (User-Defined)

**Design Decision**: NO automatic classification heuristics. Users explicitly mark sockets as input or output.

**Why**: 
- Heuristics are fragile (`--log-file` could be input OR output)
- Edge cases like `--input-output-dir` are ambiguous
- Explicit > Implicit (Zen of Python principle)
- Reduces false positives that frustrate users

```go
type SocketSide int
const (
    SocketInput  SocketSide = iota  // Left side (default)
    SocketOutput                     // Right side (user-marked)
)
```

**Default Behavior**:
- All arguments start as **Input sockets** (conservative)
- User clicks checkbox "Mark as Output" in CLI Definition Editor
- Choice is saved in JSON definition file
- No guessing, no surprises

### Classification Examples

| Argument Flag | Auto-Classification | Reasoning |
|--------------|-------------------|-----------|
| `--input-folder` | Input (◄) | Contains "input" |
| `--output-folder` | Output (►) | Contains "output" |
| `--source-path` | Input (◄) | Contains "source" |
| `--destination-dir` | Output (►) | Contains "destination" |
| `--save-results` | Output (►) | Contains "save" |
| `--load-config` | Input (◄) | Contains "load" |
| `--compression` | Input (◄) | Ambiguous → defaults to input |
| `--export-format` | Output (►) | Contains "export" |
| `--threshold` | Input (◄) | Default for parameters |

### User Override

Users can override auto-classification in the CLI Definition Editor:
```
Socket: [  ◄ Input] [✓ ► Output] [  ◄► Both]
        ^Click to override classification^
```

When `user_override` is set to `true` in the definition JSON, the system respects the user's choice and doesn't re-classify on reload.

## Connection Validation System

### Type Compatibility Rules

```go
// CanConnect checks if two sockets can be connected
func CanConnect(from, to Socket) (bool, string) {
    // 1. Direction check: must be output → input
    if from.SocketSide != SocketOutput {
        return false, "Source must be an output socket"
    }
    if to.SocketSide != SocketInput {
        return false, "Destination must be an input socket"
    }
    
    // 2. No self-connections
    if from.NodeID == to.NodeID {
        return false, "Cannot connect node to itself"
    }
    
    // 3. Input sockets can only have one connection
    if to.ConnectedTo != nil {
        return false, "Input socket already connected"
    }
    
    // 4. Type compatibility check
    if !TypesCompatible(from.Type, to.Type) {
        return false, fmt.Sprintf("Incompatible types: %s → %s", from.Type, to.Type)
    }
    
    // 5. Cycle detection (prevents infinite loops)
    if WouldCreateCycle(from.NodeID, to.NodeID) {
        return false, "Connection would create a cycle"
    }
    
    return true, ""
}

// TypesCompatible checks if output type can feed into input type
func TypesCompatible(fromType, toType ArgumentType) bool {
    // Exact type match always works
    if fromType == toType {
        return true
    }
    
    // Path can feed GlobPattern (directory → pattern)
    if fromType == TypePath && toType == TypeGlobPattern {
        return true
    }
    
    // GlobPattern can feed Path (matched files → input)
    if fromType == TypeGlobPattern && toType == TypePath {
        return true
    }
    
    // String is universal receiver (everything can convert to string)
    if toType == TypeString {
        return true
    }
    
    // No implicit conversions for numbers/bools
    return false
}
```

### Visual Feedback for Connections

```
Valid Connection (Green):
    [Node A: output-folder] ●──────────● [Node B: input-folder]
                           Green line, both sockets glow

Invalid Type Mismatch (Red):
    [Node A: output-count (int)] ●╌╌╌╌╌╌X╌╌╌● [Node B: input-path (path)]
                                Red dashed line, X at midpoint
                                Tooltip: "Incompatible types: int → path"

Invalid Direction (Red):
    [Node A: input-folder] ●╌╌╌╌╌╌X╌╌╌● [Node B: output-folder]
                          Red dashed line, X at midpoint
                          Tooltip: "Cannot connect input to output"

Cycle Detection (Red):
    [Node A] → [Node B] → [Node C]
       ↑                      ↓
       └──────────[Attempting]←─────── Red line with warning icon
                  Tooltip: "Connection would create a cycle"
```

### Connection Interaction

1. **Creating Connections**:
   - Click and drag from output socket (●►)
   - Hover over potential input sockets
   - Valid targets highlight green, invalid targets highlight red
   - Release to create connection
   - Invalid connections show error tooltip

2. **Connection States**:
   - **Normal**: Subtle colored line matching socket type
   - **Hover**: Line thickens, both sockets highlight
   - **Selected**: Line becomes dashed, delete button appears
   - **Invalid**: Red color, warning icon, tooltip with reason

3. **Deleting Connections**:
   - Click connection line → shows delete button
   - Right-click connection → "Delete Connection" menu
   - Select connection → press Delete key
   - Delete either connected node → auto-removes connections

## Help Parser System

### Automatic Argument Discovery

The GUI can parse `--help` output to auto-discover CLI arguments:

```go
// ParseHelpOutput extracts arguments from --help text
func ParseHelpOutput(helpText string) []ArgumentDefinition {
    var args []ArgumentDefinition
    
    // Common patterns in --help output:
    // --input-folder PATH        Input directory for files
    // --output-folder PATH       Output directory (default: ./output)
    // --compression INT          Compression level (range: 0-9)
    // --verbose, -v              Enable verbose logging
    
    lines := strings.Split(helpText, "\n")
    for _, line := range lines {
        // Skip non-argument lines
        if !strings.HasPrefix(strings.TrimSpace(line), "--") {
            continue
        }
        
        // Extract flag, type, description
        parts := regexp.MustCompile(`\s+`).Split(line, -1)
        flag := parts[0]
        
        argDef := ArgumentDefinition{
            Flag:         flag,
            Type:         inferTypeFromHelp(parts),
            SocketSide:   ClassifyArgument(flag),
            IsRequired:   !strings.Contains(line, "optional"),
            DefaultValue: extractDefault(line),
            Description:  extractDescription(line),
            UserOverride: false,
        }
        
        args = append(args, argDef)
    }
    
    return args
}
```

### Usage Flow

1. User adds new CLI tool to canvas
2. GUI prompts: "Discover arguments automatically?"
3. User provides help command: `python script.py --help`
4. System runs command, captures output
5. Parses output into argument definitions
6. Shows preview in CLI Definition Editor
7. User reviews/modifies before saving

## File Pattern Widget Logic

### Implementation of `get_files_to_process2` in Go
Based on Python implementation from `bioimage_pipeline_utils.py`:

```go
// GetFilesToProcess2 mimics the Python function
func GetFilesToProcess2(searchPattern string, searchSubfolders bool) ([]string, error) {
    // If recursive search and no ** in pattern, inject it
    if searchSubfolders && !strings.Contains(searchPattern, "**") {
        dir, file := filepath.Split(searchPattern)
        searchPattern = filepath.Join(dir, "**", file)
    }
    
    // Use filepath.Glob with Match for pattern matching
    matches, err := filepath.Glob(searchPattern)
    if err != nil {
        return nil, err
    }
    
    // Convert to forward slashes and sort
    for i := range matches {
        matches[i] = filepath.ToSlash(matches[i])
    }
    sort.Strings(matches)
    
    return matches, nil
}
```

### Widget Behavior
1. **On Text Change**: Debounce 500ms, then validate and preview
2. **On Browse Click**: Open native directory picker, update pattern to `selected_dir/*.tif`
3. **On Checkbox Toggle**: Re-evaluate matches with/without recursion
4. **Validation**: 
   - Check if pattern is valid glob syntax
   - Check if any files match
   - Display count and sample results

## Feature Roadmap

### Phase 0: Setup & Prototype (Week 1)
**Goal**: Get Wails environment working with React Flow

- [ ] **Wails Setup**
  - [ ] Install Wails v2: `go install github.com/wailsapp/wails/v2/cmd/wails@latest`
  - [ ] Create new project: `wails init -n pipeline-designer -t react-ts`
  - [ ] Test build: `wails dev` (hot reload) and `wails build`
  - [ ] Verify cross-platform builds (Windows primary)

- [ ] **React Flow Integration**
  - [ ] Install React Flow: `npm install reactflow`
  - [ ] Create basic canvas with 2 example nodes
  - [ ] Test pan/zoom/minimap functionality
  - [ ] Verify Wails ↔ React communication (ping/pong test)

### Phase 1: Core Framework (MVP) - Week 2
**Goal**: Basic visual editor that can load/save YAML with socket-based nodes

- [ ] **Project Setup**
  - [ ] Set up 3-panel layout (Tailwind CSS grid)
  - [ ] Add dark theme (Tailwind dark mode)
  - [ ] Integrate with `run_pipeline.exe` (launch with `--file` arg)

- [ ] **CLI Definition System**
  - [ ] Define `CLIDefinition` Go struct
  - [ ] Implement JSON loader/saver in Go backend
  - [ ] Create `cli_definitions/` directory with convert_to_tif.json
  - [ ] Wails method: `GetCLIDefinitions() []CLIDefinition`

- [ ] **Node Rendering (React Flow)**
  - [ ] Create CustomNode.tsx component
  - [ ] Render sockets as React Flow handles (left/right)
  - [ ] Style nodes with Tailwind (dark theme)
  - [ ] Add inline text fields for unconnected sockets
  - [ ] Node dragging works out-of-the-box (React Flow feature)

- [ ] **YAML Integration**
  - [ ] Load `convert_to_tif_uv.yaml` → create visual nodes
  - [ ] Parse YAML segments into `CLINode` instances
  - [ ] Save visual nodes → regenerate YAML format
  - [ ] Validate round-trip: Load → Modify → Save → Reload

- [ ] **Minimal Interaction**
  - [ ] Click to select node
  - [ ] Edit text in socket fields (right panel)
  - [ ] Save button (generates YAML file)

**Success Criteria**: Load `convert_to_tif_uv.yaml`, see one node on canvas with editable sockets, save back to identical YAML.

### Phase 2: Socket Connection System - Week 3
**Goal**: Enable visual connections between nodes + basic test execution

- [ ] **Connection System (React Flow Built-In)**
  - [ ] Define `SocketConnection` Go struct (backend storage)
  - [ ] Connection validation hook in React (type checking)
  - [ ] Custom edge styling (color by type)
  - [ ] Bezier curves rendered automatically by React Flow ✅
  - [ ] Connection storage synced to backend via Wails

- [ ] **Test Execution System (NEW)**
  - [ ] File selection widget in right panel
  - [ ] "Run Node" button on each node
  - [ ] Command generation with single file substitution
  - [ ] Subprocess execution with stdout/stderr capture
  - [ ] Test output console pane
  - [ ] Test status indicators (success/failed/running)
  - [ ] Stop button for long-running tests

- [ ] **Visual Feedback**
  - [ ] Socket hover highlighting
  - [ ] Valid/invalid connection preview (green/red)
  - [ ] Connection selection and deletion
  - [ ] Socket state indicators (connected/unconnected)

- [ ] **Canvas Enhancement**
  - [ ] Zoom in/out (Ctrl+Scroll)
  - [ ] Mini-map in corner
  - [ ] Grid background (toggleable)
  - [ ] Multi-select nodes (Ctrl+Click)

- [ ] **YAML Connection Resolution**
  - [ ] Resolve connected socket values during YAML export
  - [ ] Generate proper argument values from upstream nodes
  - [ ] Handle disconnected nodes gracefully

**Success Criteria**: Create two nodes, connect output→input, see value flow in YAML output. Test individual node with selected file.

### Phase 3: Command Explorer & Undo/Redo - Week 4
**Goal**: Build library of reusable CLI definitions + history system

- [ ] **Command Explorer (Left Panel)**
  - [ ] Load all CLI definitions from `cli_definitions/`
  - [ ] Category tree view (Segmentation, Processing, etc.)
  - [ ] Search/filter functionality
  - [ ] Drag-and-drop to canvas (creates new node instance)
  - [ ] Double-click to add to canvas center

- [ ] **Undo/Redo System (NEW)**
  - [ ] Implement Command pattern in Go backend
  - [ ] CommandHistory struct with Execute/Undo/Redo
  - [ ] React hook: `useUndo()` connected to Wails methods
  - [ ] Keyboard shortcuts: Ctrl+Z/Ctrl+Y
  - [ ] Visual indicator: "Undo: Add Node" in status bar

- [ ] **CLI Definition Editor**
  - [ ] Dialog for creating new CLI definitions
  - [ ] Argument list editor (add/remove/reorder)
  - [ ] Socket side override (input/output/both radio buttons)
  - [ ] Type dropdown (path, glob, string, int, bool, float)
  - [ ] Save custom definitions to `cli_definitions/custom/`

- [ ] **Node Properties Panel**
  - [ ] Socket value editor (text fields + browse buttons)
  - [ ] Environment dropdown (base, uv:*, conda:*)
  - [ ] Node name/color customization
  - [ ] "Edit Definition" button

- [ ] **Help Parser (Basic)**
  - [ ] Run `--help` command and capture output
  - [ ] Parse common patterns (--flag TYPE description)
  - [ ] Populate argument list in definition editor

**Success Criteria**: Add 5+ CLI definitions, drag onto canvas, connect them, customize in property panel.

### Phase 4: Security & Advanced Features - Week 5
**Goal**: Production-ready features with security and polish

- [ ] **Security Validation (NEW)**
  - [ ] Command whitelist enforcement
  - [ ] Shell injection pattern detection
  - [ ] Execution timeout (default 5 minutes)
  - [ ] First-run confirmation dialog for new CLIs
  - [ ] Path validation (required inputs must exist)

- [ ] **File Watching (NEW)**
  - [ ] Implement fsnotify watcher in Go backend
  - [ ] Watch output folders during test execution
  - [ ] Show notification when files created
  - [ ] Auto-refresh image preview if viewer open
  - [ ] Highlight new files in file browser

- [ ] **Validation & Error Handling**
  - [ ] Real-time socket value validation
  - [ ] Required socket warnings (red outline)
  - [ ] Type mismatch error messages
  - [ ] Cycle detection in connections
  - [ ] Path existence checking

- [ ] **Test Execution Enhancements**
  - [ ] Test history (last 5 runs per node)
  - [ ] Diff viewer (compare test outputs)
  - [ ] Test output file preview (image viewer for TIFs)
  - [ ] Batch test mode (run all nodes with selected file)
  - [ ] Export test command as shell script

- [ ] **Environment Integration**
  - [ ] Auto-detect conda environments from `conda_envs/`
  - [ ] Auto-detect UV groups from `pyproject.toml`
  - [ ] Environment dropdown populated dynamically

- [ ] **Status File Integration**
  - [ ] Load `*_status.yaml` alongside pipeline
  - [ ] Show completed nodes (green checkmark badge)
  - [ ] Display execution time in tooltip
  - [ ] Gray out skipped segments

- [ ] **Keyboard Shortcuts**
  - [ ] Ctrl+S: Save pipeline
  - [ ] Ctrl+N: New pipeline
  - [ ] Ctrl+O: Open pipeline
  - [ ] Delete: Remove selected nodes/connections
  - [ ] Ctrl+D: Duplicate selected nodes
  - [ ] Ctrl+Z/Y: Undo/redo

- [ ] **Canvas Features**
  - [ ] Auto-layout algorithm (organize nodes in flow)
  - [ ] Snap to grid
  - [ ] Alignment guides
  - [ ] Node grouping/comments
  - [ ] Export canvas as PNG/SVG

**Success Criteria**: Full pipeline editor with validation, status integration, test execution, and polished UX.

### Phase 5: Polish & Documentation - Week 6
**Goal**: Release-ready application

**Timeline Summary**: 6 weeks total (vs 7 weeks with Fyne from scratch)

- [ ] **UI Polish**
  - [ ] Node animations (collapse/expand)
  - [ ] Connection flow animation (optional)
  - [ ] Loading spinners for --help parsing
  - [ ] Toast notifications for save/load
  - [ ] Confirmation dialogs for destructive actions

- [ ] **Testing**
  - [ ] Test with all existing YAML configs in `pipeline_configs/`
  - [ ] Cross-platform testing (Windows, Linux, macOS)
  - [ ] Large pipeline performance (50+ nodes)
  - [ ] Edge cases (empty YAML, malformed definitions)

- [ ] **Documentation**
  - [ ] User guide (launching, creating pipelines, shortcuts)
  - [ ] CLI definition guide (creating custom tools)
  - [ ] Video tutorial (3-5 minutes)
  - [ ] README with screenshots

- [ ] **Build & Distribution**
  - [ ] Single executable with embedded resources
  - [ ] Windows installer (optional)
  - [ ] GitHub release with binaries

**Success Criteria**: Stable, documented, cross-platform GUI that enhances run_pipeline workflow.

## Integration Plan

### run_pipeline.go Modifications
```go
// Add flag parsing in main()
designMode := false
for _, arg := range os.Args[1:] {
    if arg == "-d" || arg == "--design" {
        designMode = true
    }
    // ... existing flag handling
}

if designMode {
    // Import the GUI package
    gui.LaunchDesigner(yamlPath, mainProgramDir)
    return
}
// ... continue with existing pipeline execution
```

### Build Configuration
```bash
# Development mode (hot reload)
cd pipeline-designer
wails dev

# Production build
wails build
# Output: build/bin/pipeline-designer.exe (Windows)
#         build/bin/pipeline-designer (Linux/macOS)

# Build for multiple platforms
wails build -platform windows/amd64
wails build -platform linux/amd64
wails build -platform darwin/universal

# Integration with run_pipeline.exe
go build -o run_pipeline.exe run_pipeline.go
# Place pipeline-designer.exe in same directory
```

**Binary Sizes**:
- Wails app: ~15-20MB (includes webview bundle)
- run_pipeline.exe: ~5MB
- Total distribution: ~25MB (vs 30-40MB for Electron)

## Dependencies

### Backend (Go)
```bash
go get github.com/wailsapp/wails/v2@latest
go get gopkg.in/yaml.v3
go get github.com/fsnotify/fsnotify  # File watching
go get github.com/google/uuid        # Node IDs
```

### Frontend (Node.js)
```bash
npm install
npm install reactflow          # Node editor
npm install zustand             # State management
npm install @tanstack/react-query  # Data fetching
npm install lucide-react        # Icons
npm install tailwindcss         # Styling
```

### go.mod
```go
require (
    github.com/wailsapp/wails/v2 v2.8.0
    gopkg.in/yaml.v3 v3.0.1
    github.com/fsnotify/fsnotify v1.7.0
    github.com/google/uuid v1.6.0
)
```

## Dark Theme Specification

### Color Palette
```go
const (
    BackgroundDark     = "#1e1e1e"  // Main background
    BackgroundMedium   = "#2d2d2d"  // Panel backgrounds
    BackgroundLight    = "#3e3e3e"  // Hover states
    BorderDark         = "#404040"  // Subtle borders
    AccentBlue         = "#007acc"  // Primary actions
    AccentGreen        = "#4ec9b0"  // Success/valid
    AccentRed          = "#f48771"  // Error/invalid
    AccentYellow       = "#dcdcaa"  // Warning
    TextPrimary        = "#d4d4d4"  // Main text
    TextSecondary      = "#858585"  // Muted text
    TextDisabled       = "#5a5a5a"  // Disabled text
)
```

### Node Category Colors
- Segmentation: Purple (#c586c0)
- Image Processing: Blue (#569cd6)
- Measurement: Green (#4ec9b0)
- Tracking: Orange (#ce9178)
- Visualization: Pink (#f48771)
- Utilities: Gray (#858585)

## Testing Strategy

### Manual Testing Checklist
- [ ] Launch with `-d` flag from command line
- [ ] Create new pipeline from scratch
- [ ] Load existing YAML files (all examples in pipeline_configs/)
- [ ] Save and verify YAML structure
- [ ] Drag nodes and connections
- [ ] Zoom and pan canvas
- [ ] Edit node properties
- [ ] File pattern matching with various glob patterns
- [ ] Cross-platform testing (Windows primary, Linux/macOS secondary)

### Edge Cases
- Empty YAML file
- Malformed YAML
- Invalid glob patterns
- Missing conda/UV environments
- Very large pipelines (100+ nodes)
- Circular dependencies in pipeline

## Documentation

### User Guide Topics
1. Launching the designer
2. Creating your first pipeline
3. Adding and connecting nodes
4. Configuring environments
5. File pattern syntax
6. Saving and loading pipelines
7. Keyboard shortcuts reference

### Developer Documentation
1. Architecture overview
2. Adding new command types
3. Custom themes
4. Extending the canvas
5. Node plugin system (future)

## Future Enhancements (Post-MVP)

### Advanced Features
- **Live Preview**: Run individual nodes and preview outputs
- **Validation**: Real-time checking of file paths and parameters
- **Templates**: Save common pipeline patterns as templates
- **Collaboration**: Share pipeline designs as JSON/YAML
- **Version Control**: Git integration for pipeline history
- **Performance**: Lazy loading for large pipelines
- **Export**: Generate standalone scripts from visual pipelines
- **Node Groups**: Collapse multiple nodes into reusable subgraphs
- **Auto-complete**: Smart suggestions for file paths and parameters
- **Diff View**: Visual comparison of pipeline versions

### Platform-Specific Features
- **Windows**: Taskbar progress indicator during pipeline execution
- **macOS**: Native menu integration
- **Linux**: Desktop file for application launcher

## Success Criteria

A successful MVP should:
1. ✅ Launch from run_pipeline with `-d` flag
2. ✅ Load any existing YAML from pipeline_configs/
3. ✅ Display pipeline as visual nodes
4. ✅ Allow dragging nodes to rearrange
5. ✅ Edit basic node properties
6. ✅ Save back to valid YAML format
7. ✅ File pattern widget shows matched files
8. ✅ Dark theme applied throughout
9. ✅ Responsive UI (no freezing during operations)
10. ✅ Executable size < 20MB

## Implementation Timeline

### Week 1: Wails Setup
- Install Wails, create project from template
- Integrate React Flow with example nodes
- Set up 3-panel layout with Tailwind
- Test Wails ↔ React communication

### Week 2: YAML & Nodes
- Go backend: YAML parser (load/save)
- CLI definitions JSON loader
- Custom React Flow nodes with sockets
- Connect frontend to backend via Wails methods

### Week 3: Connections & Testing
- React Flow connections (automatic Bezier curves)
- Connection validation (type checking)
- Test execution with subprocess
- File selection and test output console

### Week 4: CLI Library & Undo
- Command explorer with drag-and-drop
- CLI definition editor dialog
- Undo/redo system (Command pattern)
- Keyboard shortcuts

### Week 5: Security & Polish
- Security validation (whitelist, timeouts)
- File watching with fsnotify
- Error handling and validation
- Auto-layout and UI refinements

### Week 6: Documentation & Release
- User guide and video tutorial
- Cross-platform testing
- Build distribution packages
- GitHub release

## Questions for User

Before implementation, please confirm:

1. **GUI Framework**: Wails + React Flow confirmed ✅. Provides 80% of features built-in, 3-week timeline instead of 7.
   
2. **File Location**: Confirm `go/gui/` as the package location?

3. **Standalone vs Integrated**: Should this be integrated into `run_pipeline.exe` with `-d` flag (recommended), or a separate `pipeline_designer.exe`?

4. **Feature Priority**: Start with Phase 1 MVP (basic nodes + YAML round-trip)?

5. **Socket Classification**: Is the auto-classification logic acceptable (output/input/dest/source patterns)? Should we add more patterns?

6. **CLI Definition Storage**: Store definitions in `go/gui/cli_definitions/` or separate `cli_definitions/` at project root?

7. **Initial CLI Definitions**: Which tools should have pre-built definitions?
   - convert_to_tif ✓
   - segment_ernet ✓
   - segment_threshold
   - mask_measure
   - track_indexed_mask
   - merge_channels
   - drift_correction
   - extract_metadata

8. **Help Parser**: Should the --help parser be Phase 1 or Phase 3? (Recommend Phase 3 for MVP speed)

9. **Execution Order**: Should node execution order be:
   - (A) Determined purely by connections (topological sort) - recommended
   - (B) Manual order numbers on each node
   - (C) Top-to-bottom visual position on canvas

10. **Multi-output handling**: Should one output socket connect to multiple input sockets (1-to-many)? This is useful for reusing results. (Recommend YES)

## Key Design Decisions

### Decision 0: Technology Stack (REVISED)
**Chosen**: Wails v2 + React Flow (instead of Fyne from scratch)
**Reason**: 
- 80% of node editor already built (React Flow has 50k+ stars)
- Professional UI out-of-the-box
- 3-week timeline instead of 7 weeks
- Active ecosystem (React, npm packages)
**Alternative Rejected**: Fyne custom node editor (too much work)

### Decision 1: Sockets vs Ports
**Chosen**: Sockets (left/right) for CLI arguments  
**Reason**: More intuitive for CLI tools where "output" is just an argument  
**Alternative**: Traditional top/bottom ports (rejected - less suitable for CLI)

### Decision 2: Argument Classification (REVISED)
**Chosen**: Explicit user-defined classification (NO heuristics)
**Reason**: 
- Heuristics are fragile (e.g., `--log-file` ambiguous)
- Explicit > Implicit (reduces errors)
- Users mark outputs with checkbox in CLI editor
**Alternative Rejected**: Auto-classification with pattern matching (too many edge cases)

### Decision 3: Definition Storage
**Chosen**: JSON files in `cli_definitions/` directory  
**Reason**: Human-readable, version-controllable, easy to share  
**Alternative**: SQLite database (rejected - overkill for this use case)

### Decision 4: Connection Direction
**Chosen**: Output (right) → Input (left) only  
**Reason**: Matches data flow semantics (producer → consumer)  
**Alternative**: Bidirectional connections (rejected - ambiguous semantics)

### Decision 5: YAML Compatibility
**Chosen**: 100% compatible with existing run_pipeline.go parser  
**Reason**: GUI must not break existing workflows  
**Alternative**: New YAML format (rejected - would require run_pipeline rewrite)

### Decision 6: Node Layout
**Chosen**: Infinite canvas with manual positioning  
**Reason**: Flexibility for complex pipelines, similar to Unreal/Houdini  
**Alternative**: Auto-layout tree (rejected - too rigid for complex flows)

### Decision 7: Socket Type System
**Chosen**: Typed sockets with validation (Path, GlobPattern, String, Int, Bool, Float)  
**Reason**: Prevents invalid connections, better UX  
**Alternative**: Untyped strings (rejected - error-prone)

### Decision 8: CLI Definition Scope
**Chosen**: Per-tool definitions stored in JSON library  
**Reason**: Reusable templates reduce duplication  
**Alternative**: Per-node argument definitions (rejected - too much manual work)

### Decision 9: Test Execution Architecture (NEW)
**Chosen**: GUI spawns subprocesses for single-node testing, run_pipeline.exe for production  
**Reason**: Separates development workflow from batch processing workflow  
**Benefits**:
- Faster iteration during development
- No risk of corrupting pipeline state
- Real-time feedback for debugging
- Production execution remains unchanged

**Alternative Rejected**: GUI calls run_pipeline.exe with special flags  
**Why Rejected**: Would require modifying run_pipeline to support single-file mode, adds complexity

### Decision 10: File Selection Scope (NEW)
**Chosen**: Left panel shows ALL files matching glob pattern, user selects ONE for testing  
**Reason**: Visual feedback of what will be processed, explicit test file selection  
**Alternative Rejected**: Auto-select first file  
**Why Rejected**: User should intentionally choose test file (edge cases, problem files, etc.)

### Decision 11: Undo/Redo System (NEW)
**Chosen**: Command pattern with history stack (max 100 actions)
**Reason**: 
- Industry standard for editors (Blender, Photoshop, VS Code)
- Clean separation of concerns
- Easy to implement with React Flow's state management
**Implementation**: Go backend stores command history, React hooks trigger Execute/Undo/Redo

### Decision 12: Security Model (NEW)
**Chosen**: Whitelist executables + timeout + shell injection detection
**Reason**: 
- Arbitrary CLI execution is dangerous (malicious JSON definitions)
- Timeout prevents infinite loops
- Validation catches common attack vectors
**Trade-off**: Requires initial configuration, but safer for production

### Decision 13: File Watching (NEW)
**Chosen**: Use fsnotify to monitor output directories during test execution
**Reason**: 
- Provides immediate feedback when files are created
- Enables "live preview" workflow
- Low overhead (OS-level notifications)
**Use Case**: User runs segmentation, sees mask appear in real-time

## Notes

- The GUI should never modify `run_pipeline.go` core execution logic
- **Test mode is for development/debugging, production execution is run_pipeline.exe**
- **Test runs use REAL output files (not temporary) - use separate test output folders**
- All YAML generated must be 100% compatible with current run_pipeline.go parser
- Focus on Windows first (primary development platform), then test cross-platform
- Keep dependencies minimal for faster builds and smaller executables
- Design for future extensibility (custom node types, plugins)
- CLI definitions are the "building blocks" - users can create their own
- Socket connections represent **data flow**, not just execution order
- The system is **language-agnostic** - Python, ImageJ, R, shell scripts all work
- Think of this as "Visual Studio Code meets Unreal Blueprints for CLI tools"
- **Selected file in right panel is the test input, glob pattern is for production**

## Implementation Notes

### Code Organization
```
go/gui/
├── main entry (gui.go)
├── core models (node.go, socket.go, connection.go, cli_definition.go)
├── UI components (canvas.go, command_explorer.go, properties_panel.go)
├── business logic (yaml_parser.go, argument_classifier.go, help_parser.go)
├── visual styling (theme.go, rendering.go)
└── utilities (validation.go, file_utils.go)
```

### Critical Implementation Details

1. **Canvas Coordinate System**:
   - World coordinates: Nodes positioned in infinite 2D space
   - Screen coordinates: Viewport into world (with zoom/pan transforms)
   - Need coordinate conversion functions for mouse interaction

2. **Socket Positioning**:
   - Input sockets: Left edge, evenly spaced vertically
   - Output sockets: Right edge, evenly spaced vertically
   - Socket Y-offset calculation: `(nodeHeight / (socketCount + 1)) * socketIndex`

3. **Bezier Curve Connections**:
   - Start: Output socket center point
   - End: Input socket center point
   - Control points: Horizontal offset for smooth curve
   - Formula: `P(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃`

4. **YAML Serialization Strategy**:
   - On save: Topological sort of nodes by connections
   - Resolve socket values: connected → use upstream, unconnected → use text field
   - Preserve YAML structure: `run: [segments]` format
   - Handle special segment types: pause, stop, force

5. **Event Handling**:
   - Mouse down on socket: Start connection drag
   - Mouse move: Update connection preview line
   - Mouse up on socket: Validate and create connection
   - Mouse up on canvas: Cancel connection
   - Click on node: Select and show properties
   - Drag node: Update position in world coordinates

6. **Type System**:
   ```go
   TypePath:        color.RGBA{52, 152, 219, 255}   // Blue
   TypeGlobPattern: color.RGBA{26, 188, 156, 255}   // Cyan
   TypeString:      color.RGBA{149, 165, 166, 255}  // Gray
   TypeInt:         color.RGBA{46, 204, 113, 255}   // Green
   TypeFloat:       color.RGBA{46, 204, 113, 255}   // Green
   TypeBool:        color.RGBA{230, 126, 34, 255}   // Orange
   ```

7. **Performance Considerations**:
   - Lazy rendering: Only draw visible nodes (viewport culling)
   - Connection line decimation for zoom < 50%
   - Debounce text field updates (500ms)
   - Cache bezier curve points for redraw

8. **Validation Pipeline**:
   - On socket edit: Validate value format (path exists, int in range, etc.)
   - On connection: Check type compatibility
   - On save: Ensure all required sockets have values
   - Before execution: Run full validation pass

### Wails + React Flow Integration Tips

**Go Backend Methods** (auto-bound to frontend):
```go
// App struct with Wails methods
type App struct {
    ctx context.Context
}

// Wails will auto-generate TypeScript bindings for these:
func (a *App) LoadPipeline(path string) (*Pipeline, error) {
    // Parse YAML, return Pipeline struct
}

func (a *App) SavePipeline(pipeline *Pipeline, path string) error {
    // Convert to YAML, write file
}

func (a *App) GetCLIDefinitions() ([]CLIDefinition, error) {
    // Load all JSON files from cli_definitions/
}

func (a *App) ExecuteNode(node *CLINode, testFile string) (*TestResult, error) {
    // Run subprocess, capture output
}
```

**React Flow Custom Node**:
```tsx
import { Handle, Position } from 'reactflow';

function CLINode({ data }) {
  return (
    <div className="bg-gray-800 border-2 border-gray-600 rounded-lg p-4">
      {/* Input sockets on left */}
      {data.inputSockets.map((socket, i) => (
        <Handle
          key={socket.id}
          type="target"
          position={Position.Left}
          id={socket.id}
          style={{ top: `${(i + 1) * 30}px` }}
        />
      ))}
      
      {/* Node content */}
      <div className="text-white font-semibold">{data.name}</div>
      
      {/* Output sockets on right */}
      {data.outputSockets.map((socket, i) => (
        <Handle
          key={socket.id}
          type="source"
          position={Position.Right}
          id={socket.id}
          style={{ top: `${(i + 1) * 30}px` }}
        />
      ))}
    </div>
  );
}
```

**State Management with Zustand**:
```tsx
import create from 'zustand';
import { LoadPipeline, SavePipeline } from '../wailsjs/go/main/App';

const usePipelineStore = create((set) => ({
  nodes: [],
  edges: [],
  
  loadPipeline: async (path) => {
    const pipeline = await LoadPipeline(path);
    set({ nodes: pipeline.nodes, edges: pipeline.connections });
  },
  
  savePipeline: async (path) => {
    const { nodes, edges } = get();
    await SavePipeline({ nodes, connections: edges }, path);
  },
}));
```

---

**Document Version**: 3.0 (Wails + React Flow Architecture)  
**Date**: 2025-10-25  
**Status**: Architecture Revised - Ready for Implementation  
**Key Changes from v2.0**:
- Switched from Fyne to Wails + React Flow (3 weeks saved)
- Removed auto-classification heuristics (explicit user choice)
- Added Undo/Redo system (Command pattern)
- Added Security validation (whitelist + timeouts)
- Added File watching for live preview

**Next Steps**: 
1. Install Wails: `go install github.com/wailsapp/wails/v2/cmd/wails@latest`
2. Create project: `wails init -n pipeline-designer -t react-ts`
3. Test basic setup: `wails dev`
4. Begin Phase 1 implementation (YAML parser + CLI definitions)
