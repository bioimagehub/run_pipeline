# Node Width Persistence Implementation

## Overview

This document describes the implementation of persistent node width in the Pipeline Designer. Users can now resize nodes horizontally by dragging the resize handle, and the width will be saved and restored when the pipeline is closed and reopened.

## Changes Made

### 1. TypeScript Interface Updates

**File: `frontend/src/stores/pipelineStore.ts`**

Added `width` property to the `CLINodeData` interface:
```typescript
interface CLINodeData {
  // ... existing properties
  width?: number; // Store node width for persistence
}
```

**File: `frontend/src/components/nodes/CLINode.tsx`**

Added `width` property to the local `CLINodeData` interface:
```typescript
interface CLINodeData {
  // ... existing properties
  width?: number; // Store node width for persistence
}
```

### 2. State Management

**File: `frontend/src/stores/pipelineStore.ts`**

Added new function `updateNodeWidth` to the pipeline store:
```typescript
updateNodeWidth: (nodeId, width) => {
  set((state) => {
    const updatedNodes = state.nodes.map((node) => {
      if (node.id === nodeId) {
        return {
          ...node,
          data: {
            ...node.data,
            width: width,
          },
        };
      }
      return node;
    });
    // ... update selectedNode and mark as modified
  });
}
```

### 3. Component Updates

**File: `frontend/src/components/nodes/CLINode.tsx`**

#### Initialize width from stored data:
```typescript
const [nodeWidth, setNodeWidth] = useState(data.width || 300);
```

#### Save width on resize complete:
```typescript
const handleMouseUp = () => {
  // ... cleanup event listeners
  // Save the final width to the store
  updateNodeWidth(id, nodeWidth);
};
```

#### Sync width when data changes (e.g., when loading from file):
```typescript
React.useEffect(() => {
  if (data.width !== undefined && data.width !== nodeWidth) {
    setNodeWidth(data.width);
  }
}, [data.width, nodeWidth]);
```

### 4. Persistence Layer

**File: `frontend/src/stores/pipelineStore.ts`**

#### When creating new nodes:
```typescript
data: {
  // ... other properties
  width: 300, // Default width for new nodes
}
```

#### When loading from YAML:
```typescript
data: {
  // ... other properties
  width: node.size?.width || 300, // Preserve width from YAML or default to 300
}
```

#### When saving to backend:
```typescript
size: { width: node.data.width || 300, height: 150 }, // Use stored width or default
```

## How It Works

1. **User Interaction**: User drags the resize handle on the right edge of a node
2. **Local State Update**: The `nodeWidth` state is updated in real-time for smooth UX
3. **Store Update**: On mouse up, `updateNodeWidth` is called to save to the global store
4. **Persistence**: The width is saved to both:
   - `.reactflow.json` file (full React Flow state including node data)
   - YAML file (via the backend's `Size` struct with `width` and `height`)
5. **Restoration**: When loading a pipeline:
   - If `.reactflow.json` exists, load complete state including width
   - If loading from YAML only, extract width from `node.size.width`
   - Default to 300px if no width is found

## Backend Compatibility

The Go backend already supports the `Size` struct with `Width` and `Height` fields in `models.go`:

```go
type Size struct {
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}
```

The YAML parser (`yaml_parser.go`) already loads and saves node sizes from/to YAML metadata, so no backend changes were required.

## Testing

To test the implementation:

1. Open the Pipeline Designer
2. Add a node to the canvas
3. Drag the resize handle on the right edge to change the width
4. Save the pipeline
5. Close and reopen the pipeline
6. Verify the node width is preserved

## Benefits

- **User Preference**: Users can customize node widths to fit their workflow
- **Readability**: Wider nodes can display longer parameter values without truncation
- **Persistence**: Width preferences are maintained across sessions
- **Backward Compatible**: Old pipelines without width data will use the default 300px

## Future Enhancements

Potential improvements for future versions:
- Height resizing (currently fixed at calculated height based on content)
- Bulk resize (resize all nodes at once)
- Width presets (small, medium, large)
- Auto-fit width based on content length
