# CLI Definition Format Reference

This document describes how to create new CLI definition files for the Pipeline Designer. Each definition file represents a command-line tool that can be added to visual pipelines.

## What is `create_if_missing`?

The validation option `"create_if_missing"` tells the pipeline runner to automatically create the output directory if it doesn't exist when the pipeline runs. This is commonly used for output paths to ensure the destination folder exists before writing files.

## File Structure

CLI definitions are JSON files stored in the `cli_definitions/` folder. Each file defines one command-line tool with its arguments and configuration.

Allthough output args does not really exist in cli this is added and the idea is that all files that are created by the cli should be listed here.
This way the next cli can use the files

## Top-Level Properties

```json
{
  "id": "unique_tool_id",
  "name": "Display Name",
  "category": "Category Name",
  "icon": "🔬",
  "color": "#569cd6",
  "description": "Brief description of what this tool does",
  "environment": "uv@3.11:env-name OR conda:env-name",
  "executable": "python",
  "script": "path/to/script.py",
  "helpCommand": "python path/to/script.py --help",
  "arguments": [ /* see below */ ],
  "version": "1.0.0",
  "author": "Your Name/Organization",
  "lastParsed": "2025-01-25T10:30:00Z"
}
```

### Property Details

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | ✅ | Unique identifier (use lowercase with underscores) |
| `name` | string | ✅ | Display name shown in the UI |
| `category` | string | ✅ | Category for grouping (see categories below) |
| `icon` | string | ✅ | Emoji icon displayed in the UI |
| `color` | string | ✅ | Hex color code for the node header |
| `description` | string | ✅ | Short description of the tool's purpose |
| `environment` | string | ✅ | Environment specification (see formats below) |
| `executable` | string | ✅ | Command to run (e.g., `python`, `bash`, `./tool`) |
| `script` | string | ✅ | Path to script relative to project root |
| `helpCommand` | string | ❌ | Command to display help (optional) |
| `arguments` | array | ✅ | List of command-line arguments (see below) |
| `version` | string | ❌ | Tool version |
| `author` | string | ❌ | Author name or organization |
| `lastParsed` | string | ❌ | ISO timestamp of last parsing |

### Environment Formats

The `environment` field specifies how to run the tool:

- **UV environments**: `uv@3.11:env-name` (fast, modern Python package manager)
- **Conda environments**: `conda:env-name` (traditional conda environments)

### Standard Categories

Use one of these categories for consistency:

- `Segmentation` - Cell/organelle segmentation tools
- `Image Processing` - Filtering, conversion, enhancement
- `Measurement` - Quantification and analysis
- `Tracking` - Time-series and tracking
- `Visualization` - Plotting and visualization
- `Input/Output` - File selection and I/O operations
- `Utilities` - General-purpose tools

### Standard Colors

Recommended colors by category:

- Segmentation: `#c586c0` (purple)
- Image Processing: `#569cd6` (blue)
- Measurement: `#4ec9b0` (teal)
- Tracking: `#ce9178` (orange)
- Visualization: `#f48771` (coral)
- Input/Output: `#4ec9b0` (teal)
- Utilities: `#858585` (gray)

## Argument Structure

Each argument in the `arguments` array has this structure:

```json
{
  "flag": "--argument-name",
  "type": "path",
  "socketSide": "input",
  "isRequired": true,
  "defaultValue": "./default/path",
  "description": "Description of what this argument does",
  "validation": "must_exist",
  "userOverride": false
}
```

### Argument Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `flag` | string | ✅ | Command-line flag (e.g., `--input-folder`) |
| `type` | string | ✅ | Data type (see types below) |
| `socketSide` | string | ✅ | `"input"` or `"output"` |
| `isRequired` | boolean | ✅ | Whether this argument is mandatory |
| `defaultValue` | string | ✅ | Default value (can use path tokens) |
| `description` | string | ✅ | Help text shown in UI |
| `validation` | string | ❌ | Validation rules (see below) |
| `userOverride` | boolean | ❌ | Currently unused (keep `false`) |

### Argument Types

| Type | Description | Example Value |
|------|-------------|---------------|
| `path` | File or folder path | `./data/output` |
| `glob_pattern` | File pattern with wildcards | `./input/*.tif` |
| `file_list` | List of file paths | `["file1.tif", "file2.tif"]` |
| `string` or `str` | Text string | `"any text"` |
| `int` | Integer number | `42` |
| `float` | Decimal number | `3.14` |
| `bool` | Boolean value | `true` or `false` |
| `choice` | Selection from options | `"gaussian"` |

### Socket Side

Determines whether the argument is an input or output:

- **`"input"`**: Left side of node (accepts data)
- **`"output"`**: Right side of node (produces data)

### Validation Rules

The `validation` field specifies how to validate argument values. Multiple rules can be combined.

#### Path Validation

| Rule | Meaning | Use Case |
|------|---------|----------|
| `must_exist` | Path must exist before running | Input files/folders |
| `create_if_missing` | Create path if it doesn't exist | Output folders |
| (empty) | No validation | Optional or dynamic paths |

#### Numeric Validation

| Rule | Example | Description |
|------|---------|-------------|
| `range:MIN-MAX` | `range:0-9` | Value must be between MIN and MAX |
| `range:MIN-MAX` | `range:0.0-1.0` | Works for both int and float |

#### Choice Validation

| Rule | Example | Description |
|------|---------|-------------|
| `enum:opt1,opt2,opt3` | `enum:gaussian,median,mean` | Value must be one of the listed options |

### Path Tokens

Use these special tokens in `defaultValue` for portable paths:

- `%REPO%` - Project root directory
- `%YAML%` - Directory containing the YAML pipeline file
- `%TEST_FOLDER%` - Custom environment variable from `.env`
- Any other `%VAR_NAME%` defined in `.env` (except `CONDA_PATH` and `IMAGEJ_PATH`)

Example:
```json
"defaultValue": "%YAML%/input/*.tif"
```

This becomes `E:\path\to\pipeline_configs\input/*.tif` at runtime.

## Complete Example

```json
{
  "id": "my_custom_tool",
  "name": "My Custom Tool",
  "category": "Image Processing",
  "icon": "✨",
  "color": "#569cd6",
  "description": "Does something amazing with images",
  "environment": "uv@3.11:image-processing",
  "executable": "python",
  "script": "standard_code/python/my_tool.py",
  "helpCommand": "python standard_code/python/my_tool.py --help",
  "arguments": [
    {
      "flag": "--input-folder",
      "type": "path",
      "socketSide": "input",
      "isRequired": true,
      "defaultValue": "%YAML%/input",
      "description": "Folder containing input images",
      "validation": "must_exist",
      "userOverride": false
    },
    {
      "flag": "--output-folder",
      "type": "path",
      "socketSide": "output",
      "isRequired": true,
      "defaultValue": "%YAML%/output",
      "description": "Folder for output results",
      "validation": "create_if_missing",
      "userOverride": false
    },
    {
      "flag": "--method",
      "type": "choice",
      "socketSide": "input",
      "isRequired": true,
      "defaultValue": "fast",
      "description": "Processing method to use",
      "validation": "enum:fast,accurate,balanced",
      "userOverride": false
    },
    {
      "flag": "--threshold",
      "type": "float",
      "socketSide": "input",
      "isRequired": false,
      "defaultValue": "0.5",
      "description": "Detection threshold (0.0-1.0)",
      "validation": "range:0.0-1.0",
      "userOverride": false
    },
    {
      "flag": "--enable-preview",
      "type": "bool",
      "socketSide": "input",
      "isRequired": false,
      "defaultValue": "false",
      "description": "Enable preview mode",
      "validation": "",
      "userOverride": false
    }
  ],
  "version": "1.0.0",
  "author": "BIPHUB",
  "lastParsed": "2025-10-27T10:00:00Z"
}
```

## Testing Your Definition

1. **Save** the JSON file in `pipeline-designer/cli_definitions/`
2. **Rebuild** the pipeline designer:
   ```bash
   cd pipeline-designer
   .\build.ps1
   ```
3. **Run** the pipeline designer:
   ```bash
   cd ..
   .\run_pipeline.exe -d
   ```
4. **Find** your tool in the Command Explorer under its category
5. **Drag** it onto the canvas to test

## Common Validation Patterns

### Input Files (Must Exist)
```json
{
  "flag": "--input-files",
  "type": "glob_pattern",
  "socketSide": "input",
  "isRequired": true,
  "defaultValue": "%YAML%/input/*.tif",
  "validation": "must_exist"
}
```

### Output Folder (Auto-Create)
```json
{
  "flag": "--output-folder",
  "type": "path",
  "socketSide": "output",
  "isRequired": true,
  "defaultValue": "%YAML%/output",
  "validation": "create_if_missing"
}
```

### Numeric Parameter with Range
```json
{
  "flag": "--iterations",
  "type": "int",
  "socketSide": "input",
  "isRequired": false,
  "defaultValue": "10",
  "validation": "range:1-100"
}
```

### Dropdown Selection
```json
{
  "flag": "--mode",
  "type": "choice",
  "socketSide": "input",
  "isRequired": true,
  "defaultValue": "auto",
  "validation": "enum:auto,manual,advanced"
}
```

### File List Output (from connections)
```json
{
  "flag": "--file-list",
  "type": "file_list",
  "socketSide": "output",
  "isRequired": false,
  "defaultValue": "",
  "validation": ""
}
```

## Tips and Best Practices

1. **Use descriptive flags**: `--input-folder` is better than `--in`
2. **Provide good defaults**: Use path tokens for portability
3. **Add validation**: Helps catch errors before running
4. **Write clear descriptions**: Users see these as tooltips
5. **Group related tools**: Use consistent categories
6. **Test with real data**: Ensure your tool works in the pipeline
7. **Document in script**: Add `--help` support in your Python/script
8. **Use standard patterns**: Follow existing examples for consistency

## Troubleshooting

### "Definition not found" error
- Check that the JSON file is in `cli_definitions/` folder
- Rebuild the pipeline designer after adding new files
- Verify the `id` matches what you're trying to use

### Node doesn't appear in UI
- Check JSON syntax (use a JSON validator)
- Ensure all required fields are present
- Look in the pipeline-designer.log for error messages

### Values not propagating between nodes
- Verify socket types match (output → input)
- Check that `socketSide` is correct (`input` vs `output`)
- Use the browser console to debug value updates

## Questions or Issues?

See the main project documentation or contact BIPHUB support.
