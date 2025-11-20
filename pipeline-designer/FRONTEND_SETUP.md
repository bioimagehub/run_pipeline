# Frontend Setup Instructions

## Prerequisites Required

Before continuing with frontend development, you need to install:

1. **Node.js** (v18 or later): https://nodejs.org/
   - This includes npm (Node Package Manager)
   - Required for installing React dependencies

## Once Node.js is Installed

Run these commands from the `pipeline-designer/frontend` directory:

```bash
# Install dependencies
npm install reactflow zustand lucide-react autoprefixer postcss tailwindcss

# Initialize Tailwind CSS
npx tailwindcss init -p
```

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx              # Main 3-panel layout
│   ├── components/
│   │   ├── Canvas.tsx       # React Flow canvas wrapper
│   │   ├── CommandExplorer.tsx  # Left panel: CLI library
│   │   ├── PropertiesPanel.tsx  # Right panel: socket editor
│   │   └── nodes/
│   │       └── CLINode.tsx  # Custom node with sockets
│   ├── stores/
│   │   └── pipelineStore.ts # Zustand state management
│   └── styles/
│       └── globals.css      # Dark theme styles
```

## Files Created

The following React components have been created and are ready to use once npm dependencies are installed:

1. ✅ **App.tsx** - Main 3-panel layout
2. ✅ **components/Canvas.tsx** - React Flow canvas
3. ✅ **components/CommandExplorer.tsx** - Left sidebar with CLI tools
4. ✅ **components/PropertiesPanel.tsx** - Right sidebar for editing
5. ✅ **components/nodes/CLINode.tsx** - Custom node component
6. ✅ **stores/pipelineStore.ts** - State management
7. ✅ **styles/globals.css** - Dark theme CSS

## Next Steps

1. Install Node.js
2. Run `npm install` commands
3. Run `wails dev` from the `pipeline-designer` directory
4. Test the GUI with existing YAML files
