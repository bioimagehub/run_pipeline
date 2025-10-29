# Visual Examples - Socket Type Validation

## Before and After

### BEFORE (No Type Validation)
```
┌─────────────────────┐         ┌─────────────────────┐
│   Convert to TIF    │         │   Merge Channels    │
│                     │         │                     │
│  TIF files ⚫────────┼────────▶⚫ input-files        │
│  (glob_pattern)     │         │ (glob_pattern)      │
│                     │         │                     │
└─────────────────────┘         └─────────────────────┘
```
**Problem**: Can't tell if connection is valid!

### AFTER (With Type Validation) ✅
```
┌─────────────────────┐         ┌─────────────────────┐
│   Convert to TIF    │         │   Merge Channels    │
│                     │         │                     │
│  TIF files 🟢───────┼────────▶🟢 input-files        │
│  (glob_pattern)     │         │ (glob_pattern)      │
│                     │         │                     │
└─────────────────────┘         └─────────────────────┘
```
**Solution**: Green dots = Valid connection!

---

## Type Mismatch Warning

### Valid Connection
```
┌─────────────────────┐         ┌─────────────────────┐
│  Node A             │         │  Node B             │
│                     │         │                     │
│  output 🟢──────────┼────────▶🟢 input             │
│  (glob_pattern)     │ Solid   │ (glob_pattern)      │
│                     │ Blue    │                     │
└─────────────────────┘         └─────────────────────┘
```

### Invalid Connection (Warning)
```
┌─────────────────────┐         ┌─────────────────────┐
│  Node A             │         │  Node B             │
│                     │         │                     │
│  output 🔴╍╍╍╍╍╍╍╍╍╍┼────────▶🔴 input             │
│  (glob_pattern)     │ Dashed  │ (string)            │
│                     │ Red     │                     │
└─────────────────────┘         └─────────────────────┘
```
**Warning**: Types don't match! (but connection still allowed)

---

## Real-World Example

### Image Processing Pipeline

```
┌────────────────────────┐
│   Input: Select Files  │
│                        │
│  Raw files 🟢          │
│  (glob_pattern)        │
└──────────┬─────────────┘
           │ Green
           ▼
┌────────────────────────┐
│   Convert to TIF       │
│                        │
│🟢 input-files          │
│  (glob_pattern)        │
│                        │
│  TIF files 🟢          │
│  (glob_pattern)        │
└──────────┬─────────────┘
           │ Green
           ▼
┌────────────────────────┐
│   Segment Cells        │
│                        │
│🟢 input-images         │
│  (glob_pattern)        │
│                        │
│  Masks 🟢              │
│  (glob_pattern)        │
└──────────┬─────────────┘
           │ Green
           ▼
┌────────────────────────┐
│   Measure Masks        │
│                        │
│🟢 mask-files           │
│  (glob_pattern)        │
│                        │
│  Results 🟢            │
│  (csv)                 │
└────────────────────────┘
```

**Result**: All green = pipeline is correctly wired! ✅

### Broken Pipeline (Type Mismatch)

```
┌────────────────────────┐
│   Convert to TIF       │
│                        │
│  TIF files 🔴          │
│  (glob_pattern)        │
└──────────┬─────────────┘
           │ Red (Dashed)
           ▼
┌────────────────────────┐
│   Custom Script        │
│                        │
│🔴 single-file          │
│  (path)                │ ⚠️ Warning!
│                        │
└────────────────────────┘
```

**Problem**: Connecting a glob pattern (many files) to a single file input!
**Action**: User sees red dots and can fix the pipeline

---

## Socket Color Legend

### Color States

```
⚫ Gray    - Not connected yet
🟢 Green   - Valid connection (types match)
🔴 Red     - Invalid connection (type mismatch)
```

### Connection Line Styles

```
━━━━━━━━  Solid Blue   - Valid connection
╍╍╍╍╍╍╍╍  Dashed Red   - Invalid connection (animated)
```

---

## Delete Node Feature

### Before Selection
```
┌─────────────────────┐
│   Convert to TIF    │  ← Not selected
│                     │
│  Input  ⚫          │
│  Output ⚫          │
└─────────────────────┘
```
**Action**: Click the node

### After Selection
```
┏━━━━━━━━━━━━━━━━━━━━━┓  ← Blue outline = selected
┃   Convert to TIF    ┃
┃  🔄 Update All      ┃  ← Action buttons appear
┃  ▶️ Run  🗑️ Delete   ┃
┃                     ┃
┃  Input  ⚫          ┃
┃  Output ⚫          ┃
┗━━━━━━━━━━━━━━━━━━━━━┛
```
**Action**: Press Delete key OR click 🗑️ button

### After Deletion
```
(node removed)

All connections automatically removed!
```

---

## Keyboard Shortcuts

```
╔════════════════════════════════════════════════════╗
║  Shortcut          │  Action                       ║
╠════════════════════════════════════════════════════╣
║  Delete            │  Delete selected node         ║
║  Ctrl+S            │  Save pipeline                ║
║  Ctrl+Shift+S      │  Save As...                   ║
║  Ctrl+O            │  Open pipeline                ║
║  Ctrl+Z            │  Undo                         ║
║  Ctrl+Y            │  Redo                         ║
╚════════════════════════════════════════════════════╝
```

---

## Smart Delete Detection

### Scenario 1: Typing in Input Field
```
┏━━━━━━━━━━━━━━━━━━━━━┓
┃   Node Name         ┃
┃                     ┃
┃  Input: [myfile█]   ┃  ← Cursor in input field
┃                     ┃
┗━━━━━━━━━━━━━━━━━━━━━┛

User types: "myfiles"
User presses Delete: "s" is removed
Result: Input shows "myfile"
```
**Smart Detection**: Delete key removes characters, NOT the node! ✅

### Scenario 2: Node Selected, Not Typing
```
┏━━━━━━━━━━━━━━━━━━━━━┓
┃   Node Name         ┃
┃                     ┃
┃  Input: [myfile]    ┃  ← No cursor
┃                     ┃
┗━━━━━━━━━━━━━━━━━━━━━┛

User presses Delete: Node is removed
Result: Node and connections deleted
```
**Smart Detection**: No input field focused, so Delete removes the node! ✅

---

## Type Compatibility Matrix

```
┌──────────────┬─────────┬────────┬──────┬─────────────┐
│ From \ To    │ string  │ path   │ bool │ glob_pattern│
├──────────────┼─────────┼────────┼──────┼─────────────┤
│ string       │   🟢    │   🟢   │  🔴  │     🔴      │
├──────────────┼─────────┼────────┼──────┼─────────────┤
│ path         │   🟢    │   🟢   │  🔴  │     🟢      │
├──────────────┼─────────┼────────┼──────┼─────────────┤
│ bool         │   🔴    │   🔴   │  🟢  │     🔴      │
├──────────────┼─────────┼────────┼──────┼─────────────┤
│ glob_pattern │   🔴    │   🟢   │  🔴  │     🟢      │
└──────────────┴─────────┴────────┴──────┴─────────────┘

Legend:
🟢 = Compatible (green sockets)
🔴 = Incompatible (red sockets)
```

---

## Tooltip Preview

When hovering over a socket:
```
     ┌──────────────────────┐
     │ Type: glob_pattern   │  ← Tooltip appears
     └──────────────────────┘
            ▲
          ⚫ Socket
```

This helps users verify types before connecting!

---

## Animation Effects

### Invalid Connection Animation
```
Frame 1:  🔴╍╍╍╍╍╍╍╍╍
Frame 2:  🔴 ╍╍╍╍╍╍╍╍
Frame 3:  🔴  ╍╍╍╍╍╍╍
...
(repeats - creates moving dashed line effect)
```

Draws attention to type mismatches!

---

**These visual examples show how the features enhance the pipeline designer UX!** 🎨
