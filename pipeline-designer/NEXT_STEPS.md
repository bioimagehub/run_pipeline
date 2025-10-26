# Next Steps Checklist

## Quick Start Guide

Follow these steps in order to get the Pipeline Designer running:

## ☐ Step 1: Install Node.js (15 minutes)

1. Go to https://nodejs.org/
2. Click **"Download Node.js (LTS)"** - recommended version
3. Run the installer
4. Accept all defaults
5. Verify installation:
   ```powershell
   node --version
   npm --version
   ```
   You should see version numbers like `v18.x.x` and `9.x.x`

## ☐ Step 2: Install Frontend Dependencies (5 minutes)

```powershell
# Navigate to frontend directory
cd E:\Oyvind\OF_git\run_pipeline\pipeline-designer\frontend

# Install React Flow, Zustand, and Lucide React
npm install reactflow zustand lucide-react

# Verify package.json was updated
```

**Expected output**: Progress bars, then "added XXX packages"

## ☐ Step 3: Build Wails Bindings (2 minutes)

```powershell
# Navigate to pipeline-designer root
cd E:\Oyvind\OF_git\run_pipeline\pipeline-designer

# Check Wails setup
wails doctor

# This will show if everything is ready
```

## ☐ Step 4: Run the Application (1 minute)

```powershell
# From pipeline-designer directory
wails dev
```

**What happens**:
1. Go backend compiles
2. Wails generates TypeScript bindings
3. React frontend starts
4. Application window opens

**First run might take 2-3 minutes** (compilation + dependency installation)

## ☐ Step 5: Test the GUI (10 minutes)

Once the application opens:

### Basic Functionality Test
1. **Command Explorer (Left Panel)**
   - Verify you see categories: Segmentation, Image Processing, Measurement
   - Click "Image Processing" to expand
   - You should see "Convert to TIF" with 🔄 icon

2. **Canvas (Center Panel)**
   - Click "Convert to TIF" in the left panel
   - A node should appear on the canvas
   - Try dragging the node around
   - Use mouse wheel to zoom in/out
   - Try middle-click drag to pan

3. **Properties Panel (Right Panel)**
   - Click the node you just added
   - Right panel should show "Properties"
   - You should see:
     - Node name: "Convert to TIF"
     - Input sockets section
     - Output sockets section
     - Environment info

4. **Socket Editing**
   - Find `--input-search-pattern` socket
   - Click in the value field
   - Type a test path: `./test/*.tif`
   - Value should update

### Advanced Test (If Time Permits)
5. **Add Multiple Nodes**
   - Add "Segment with ERnet" from Segmentation category
   - Position it to the right of Convert to TIF
   - Try connecting sockets:
     - Drag from "Convert to TIF" output socket (right side)
     - To "Segment with ERnet" input socket (left side)
   - Connection line should appear

6. **Load Existing YAML** (Future feature)
   - File → Open menu not yet implemented
   - Will be added in Phase 2

## ☐ Step 6: Verify Build (Optional - 5 minutes)

```powershell
# Build production executable
wails build

# Check output
ls build\bin\

# Should see: pipeline-designer.exe
```

## Troubleshooting

### "npm: command not found"
- Node.js not installed correctly
- Restart PowerShell after installing Node.js
- Check PATH: `$env:Path` should include Node.js

### "Cannot find module 'reactflow'"
- npm install didn't complete
- Check for errors in npm install output
- Try: `npm install --force`

### "Wails bindings not found" errors in VS Code
- Normal before first run
- Run `wails dev` once
- Bindings auto-generate in `frontend/wailsjs/`

### Application won't start
- Check `wails doctor` output
- Ensure Go and Node.js are in PATH
- Check for port conflicts (frontend uses port 34115)

### White screen / blank canvas
- Open browser DevTools (F12)
- Check console for errors
- React might still be loading

## Success Criteria

You'll know it's working when:
- ✅ Application window opens
- ✅ Dark theme is visible
- ✅ Left panel shows CLI commands
- ✅ Center panel has infinite canvas
- ✅ Right panel says "Select a node to edit"
- ✅ Clicking a command adds a node
- ✅ Node can be dragged around
- ✅ Selecting node shows properties

## Getting Help

If you encounter issues:

1. **Check the logs**:
   - Wails logs appear in terminal
   - Frontend logs in browser DevTools (F12)

2. **Common fixes**:
   - Restart `wails dev`
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules`, run `npm install` again

3. **Documentation**:
   - `README.md` - Full project documentation
   - `FRONTEND_SETUP.md` - Detailed setup instructions
   - `IMPLEMENTATION_SUMMARY.md` - What was built

4. **Design Reference**:
   - `../make run pipeline gui.md` - Complete design specification

## What's Next (Phase 2)

Once Phase 1 is working:

- **File Menu**: Implement save/load dialogs
- **Connection Validation**: Type checking for socket connections
- **Test Execution**: Run individual nodes with sample files
- **Undo/Redo**: Command pattern implementation
- **More CLI Definitions**: Add remaining pipeline tools

## Quick Reference Commands

```powershell
# Check versions
node --version
npm --version
wails doctor

# Install dependencies (one-time)
cd pipeline-designer\frontend
npm install reactflow zustand lucide-react

# Run development mode
cd pipeline-designer
wails dev

# Build production
wails build

# Clean build
Remove-Item -Recurse -Force build, frontend\node_modules
```

---

**Total estimated time: 30-40 minutes**

Good luck! 🚀
