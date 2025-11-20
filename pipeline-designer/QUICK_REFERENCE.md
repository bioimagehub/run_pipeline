# Quick Reference Card - New Features

## 🎨 Socket Colors

```
⚫ Gray  = Not connected
🟢 Green = Valid connection (types match!)
🔴 Red   = Type mismatch warning!
```

**What to do if you see red:**
1. Check the socket types (hover to see tooltip)
2. Verify if the connection makes sense
3. Either disconnect and fix, or proceed with caution

---

## 🗑️ Delete Nodes

### Two Ways:

**Method 1: Delete Key** (fastest)
```
1. Click node to select it
2. Press Delete
3. Done!
```

**Method 2: Delete Button**
```
1. Click node to select it
2. Click 🗑️ Delete in node header
3. Done!
```

### Safe Delete:
- Typing in an input? Delete key removes text
- Node selected? Delete key removes node
- Undo with Ctrl+Z!

---

## ⌨️ Keyboard Shortcuts

| Key           | Action        |
|---------------|---------------|
| Delete        | Delete node   |
| Ctrl+S        | Save          |
| Ctrl+Shift+S  | Save As       |
| Ctrl+O        | Open          |
| Ctrl+Z        | Undo          |
| Ctrl+Y        | Redo          |

---

## 📊 Type Compatibility

| Works Well | Needs Check |
|------------|-------------|
| same → same | glob → string |
| path → string | string → glob |
| path → glob | - |

**Tip**: When in doubt, hover over the socket to see its type!

---

## 💡 Pro Tips

1. **Color-code your workflow**: Scan for red dots before running
2. **Delete faster**: Select + Delete key = instant removal
3. **Undo mistakes**: Ctrl+Z restores deleted nodes
4. **Check types**: Hover over connection dots to verify
5. **Trust the colors**: Green = good, Red = review

---

## 🐛 Common Issues

**Q: Socket won't turn green**
- Check if types actually match (hover to see)

**Q: Delete removes text, not node**
- This is correct! Finish typing first, then press Delete

**Q: Can't see socket colors**
- Zoom in closer or check monitor brightness

**Q: Accidentally deleted node**
- Press Ctrl+Z immediately to undo

---

**Need more help?** See `FEATURE_GUIDE.md` for detailed instructions.
