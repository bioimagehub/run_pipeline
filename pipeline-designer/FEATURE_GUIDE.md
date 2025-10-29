# Pipeline Designer - Type Validation & Node Deletion Guide

## 🎨 Socket Type Validation

### Visual Color Coding

The connection dots (handles) on nodes now show the status of your connections:

```
⚫ Gray    - Not connected
🟢 Green   - Valid connection (types match)
🔴 Red     - Invalid connection (types don't match)
```

### Example Scenarios

#### ✅ Valid Connection
```
[Convert to TIF]                    [Merge Channels]
    TIF files (glob_pattern) -----> input-files (glob_pattern)
         🟢                              🟢
```
Both sockets are `glob_pattern` type → **Green handles** → All good!

#### ⚠️ Type Mismatch Warning  
```
[Convert to TIF]                    [Some Node]
    TIF files (glob_pattern) -----> input-path (string)
         🔴                              🔴
```
Connecting `glob_pattern` to `string` → **Red handles** → Type mismatch warning!

The connection will still work, but the red color alerts you that there might be issues.

### Connection Lines

- **Solid Blue Line**: Valid connection ━━━━━━
- **Dashed Red Line**: Invalid connection ╍╍╍╍╍╍ (animated)

### Compatible Types

The system allows some flexibility:

| Source Type    | Target Type    | Result |
|---------------|---------------|---------|
| glob_pattern  | glob_pattern  | 🟢 Valid |
| string        | string        | 🟢 Valid |
| path          | string        | 🟢 Valid |
| path          | glob_pattern  | 🟢 Valid |
| glob_pattern  | string        | 🔴 Warning (but allowed) |
| glob_pattern  | bool          | 🔴 Invalid |
| string        | number        | 🔴 Invalid |

### Checking Socket Types

Hover over any connection dot to see the socket type in the tooltip:

```
[Hover over dot]
  ↓
"Type: glob_pattern"
```

---

## 🗑️ Deleting Nodes

### Two Ways to Delete

#### Method 1: Keyboard Shortcut (Recommended)
1. **Click** on a node to select it (blue outline appears)
2. Press the **Delete** key
3. Node and all its connections are removed

#### Method 2: Delete Button
1. **Click** on a node to select it
2. Click the **🗑️ Delete** button in the node header
3. Node is removed

### Safety Features

✅ **Smart Detection**: Delete key only works when you're NOT typing
- Typing in input fields? Delete key removes characters (normal behavior)
- Not typing? Delete key removes the selected node

✅ **Visual Feedback**: Selected nodes have a blue outline so you know what will be deleted

✅ **Undo Support**: Deleted nodes can be restored with Ctrl+Z

### Examples

#### Safe Deletion
```
1. Click node (node selected, blue outline)
2. Press Delete → Node removed ✓
```

#### Safe Editing  
```
1. Click node (node selected)
2. Click in an input field
3. Type "myfile" then press Delete → "e" is deleted, node stays ✓
```

---

## 🎯 Best Practices

### When Building Pipelines

1. **Check Socket Colors**: Before running a pipeline, scan for red connection dots
2. **Verify Types**: Hover over sockets to confirm types match your intent
3. **Fix Mismatches**: If you see red connections:
   - Disconnect and reconnect to correct sockets
   - Or verify the pipeline will handle the type conversion

### When Deleting Nodes

1. **Select First**: Always click the node before pressing Delete
2. **Visual Confirm**: Check for the blue outline before deleting
3. **Undo Available**: Don't worry - Ctrl+Z brings it back!

---

## 🔧 Keyboard Shortcuts Summary

| Shortcut          | Action                    |
|-------------------|---------------------------|
| Delete            | Delete selected node      |
| Ctrl+S            | Save pipeline             |
| Ctrl+Shift+S      | Save As...               |
| Ctrl+O            | Open pipeline            |
| Ctrl+Z            | Undo                     |
| Ctrl+Y            | Redo                     |

---

## 🐛 Troubleshooting

### Q: Socket won't turn green after connecting
**A**: Check if the types are actually compatible. Hover over both sockets to see their types.

### Q: Delete key deletes text in input field
**A**: This is correct behavior! When focused on an input, Delete should remove text.

### Q: Can't see the socket colors
**A**: Try zooming in on the canvas. The colored dots are on the left/right edges of nodes.

### Q: Connection is red but should be valid
**A**: Check the type compatibility table above. Some conversions require explicit handling.

### Q: Accidentally deleted a node
**A**: Press Ctrl+Z to undo!

---

## 📝 Technical Notes

- Type validation is **non-blocking** - you can still create any connection
- Visual feedback is designed to **warn**, not prevent
- Socket colors update in **real-time** as you connect/disconnect
- Delete key uses **focus detection** to avoid interfering with text editing

---

**Happy Pipeline Building! 🚀**

For more information, see [SOCKET_TYPE_VALIDATION.md](SOCKET_TYPE_VALIDATION.md) for technical implementation details.
