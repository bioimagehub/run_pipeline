# Legacy YAML Import Feature

## Overview

The Pipeline Designer now supports importing legacy YAML pipeline files that were created before the visual Designer existed. These files lack the `_designer_metadata` section but can still be converted to visual node graphs.

## What Gets Imported

### Detection
The system automatically detects legacy YAMLs by checking for the absence of `_designer_metadata`:

```yaml
# Legacy YAML (will trigger import)
pipeline_name: "My Pipeline"
run:
  - name: Step 1
    commands: [...]

# Modern YAML (loads normally)
pipeline_name: "My Pipeline"
_designer_metadata:
  nodes: [...]
  connections: [...]
```

### Import Process

1. **Script Extraction**: Parses `commands` array to find Python script paths
   - Example: `./standard_code/python/convert_to_tif.py` → `convert_to_tif.py`

2. **CLI Definition Matching**: Searches `cli_definitions/` for matching JSON
   - Matches based on `Script` field in CLI definition
   - Example: `convert_to_tif.py` → `cli_definitions/Input Output/convert_to_tif.json`

3. **Argument Mapping**: Extracts flags and values from commands array
   ```yaml
   commands:
     - python
     - ./standard_code/python/convert_to_tif.py
     - --input-search-pattern
     - '%YAML%/input/**/*.nd2'
     - --destination
     - '%YAML%/output/tif'
   ```
   Maps to node sockets:
   - `input_search_pattern` socket → `'%YAML%/input/**/*.nd2'`
   - `destination` socket → `'%YAML%/output/tif'`

4. **Node Layout**: Places nodes left-to-right with 450px spacing
   - Step 1: Position (100, 200)
   - Step 2: Position (550, 200)
   - Step 3: Position (1000, 200)

## Validation & Error Reporting

### NO SILENT ERRORS Policy

The import process prints a detailed report with:

#### ✅ Success Metrics
```
[LEGACY_IMPORT] Import complete: 2/2 steps successful
```

#### ⚠️ Warnings (Non-Critical)
- **Unmatched Arguments**: YAML has args not in CLI definition
  ```
  Step 1 ('Convert to tif'): 2 unmatched arguments from YAML: [--collapse-structure, --z-projection]
  ```
  Reason: CLI definition might be outdated or script accepts dynamic args

#### ❌ Errors (Critical)
- **No Script Found**: Commands array doesn't contain a .py file
  ```
  Step 3 ('Custom step'): No script path found in commands
  ```

- **No CLI Definition**: Script exists but no matching JSON found
  ```
  Step 2 ('Custom script'): No CLI definition found for script 'my_custom_script.py'
  ```

- **Missing Required Arguments**: CLI definition requires args not in YAML
  ```
  Step 1 ('Convert to tif'): 2 REQUIRED arguments missing in YAML: [--input-search-pattern, --destination]
  ```

### Report Structure

```go
type LegacyImportReport struct {
    TotalSteps       int                    // Total steps in YAML
    SuccessfulSteps  int                    // Steps imported without errors
    Warnings         []string               // Non-critical issues
    Errors           []string               // Critical failures
    UnmatchedArgs    map[string][]string    // NodeID -> unmatched args
    MissingArgs      map[string][]string    // NodeID -> missing required args
}
```

## Code Architecture

### Files Modified

1. **legacy_importer.go** (NEW - 306 lines)
   - Core import logic
   - Script extraction and matching
   - Argument validation
   - Report generation

2. **yaml_parser.go** (MODIFIED)
   - Legacy detection: `if yamlPipeline.DesignerMetadata == nil`
   - CLI definitions path resolution using `os.Executable()`
   - Report printing integration

3. **models.go** (NO CHANGES NEEDED)
   - Existing `Pipeline` and `CLINode` structs used directly

### Key Functions

```go
// Main entry point for legacy import
func ImportLegacyYAML(yamlPipeline *YAMLPipeline, definitionsManager *CLIDefinitionsManager) (*Pipeline, *LegacyImportReport, error)

// Extracts script path from commands array
func extractScriptPath(commands interface{}) string

// Finds CLI definition matching script filename
func findCLIDefinitionByScript(scriptFilename string, manager *CLIDefinitionsManager) *CLIDefinition

// Parses YAML commands into flag-value map
func extractArgumentsFromCommands(commands interface{}) map[string]string

// Maps YAML args to node sockets, returns unmatched/missing
func populateSocketsFromYAMLArgs(node *CLINode, yamlArgs map[string]string, definition *CLIDefinition) ([]string, []string)

// Creates basic node for steps without CLI definition
func createBasicNodeFromStep(step YAMLStep, index int, startX, startY, spacing float64) CLINode

// Prints detailed import report to console
func PrintLegacyImportReport(report *LegacyImportReport)
```

## Usage Examples

### Example 1: Simple Pipeline

**Input YAML** (`test_legacy_import.yaml`):
```yaml
pipeline_name: "Legacy Test Pipeline"
run:
  - name: Convert to tif
    environment: uv@3.11:convert-to-tif
    commands:
      - python
      - ./standard_code/python/convert_to_tif.py
      - --input-search-pattern
      - '%YAML%/input/**/*.nd2'
      - --destination
      - '%YAML%/output/tif'
```

**Expected Output**:
- 1 node created: "Convert to TIFF and Collapse Folder Structure"
- Position: (100, 200)
- Sockets populated:
  - `input_search_pattern`: `%YAML%/input/**/*.nd2`
  - `destination`: `%YAML%/output/tif`
- Report: `1/1 steps successful`

### Example 2: Multi-Step with Warnings

**Input YAML**:
```yaml
run:
  - name: Step 1
    commands:
      - python
      - ./standard_code/python/convert_to_tif.py
      - --input-search-pattern
      - '%YAML%/input/**/*.nd2'
      - --destination
      - '%YAML%/output/tif'
      - --experimental-flag  # Not in CLI definition
      - true
  
  - name: Step 2
    commands:
      - python
      - ./standard_code/python/merge_channels.py
      - --input-search-pattern
      - '%YAML%/output/tif/**/*.tif'
      # Missing --destination (required)
```

**Expected Report**:
```
[LEGACY_IMPORT] Import complete: 1/2 steps successful
Warnings: 1
  Step 1: 1 unmatched arguments from YAML: [--experimental-flag]
Errors: 1
  Step 2: 1 REQUIRED arguments missing in YAML: [--destination]
```

## Testing Checklist

- [ ] Build Designer: `cd pipeline-designer ; go build`
- [ ] Test with `test_legacy_import.yaml`
- [ ] Verify nodes appear on canvas
- [ ] Check console for import report
- [ ] Test with YAML missing required args
- [ ] Test with YAML having extra args
- [ ] Test with YAML having unknown script
- [ ] Test with YAML having no script in commands
- [ ] Verify node spacing: 450px horizontal
- [ ] Verify socket values populated correctly

## Future Enhancements

1. **Auto-Connection**: Connect nodes based on glob patterns
   - If Step 1 outputs `%YAML%/output/tif/**/*.tif`
   - And Step 2 inputs `%YAML%/output/tif/**/*.tif`
   - Auto-create connection between `__link_out__` and `__link_in__`

2. **Smart Matching**: Use fuzzy matching for CLI definitions
   - Example: `convert_tif.py` → suggest `convert_to_tif.json`

3. **Bulk Import**: Import entire directories of legacy YAMLs
   - Generate migration report showing success/failure stats

4. **Diff Viewer**: Show changes between legacy YAML and modern YAML
   - Highlight new fields added by Designer

5. **Export Warning**: When saving, warn if nodes have unmatched args
   - Prevent data loss during round-trip conversion

## Related Files

- `pipeline-designer/legacy_importer.go` - Core implementation
- `pipeline-designer/yaml_parser.go` - Integration point
- `test_legacy_import.yaml` - Test file
- `pipeline_configs/*.yaml` - Real legacy pipelines to test

## See Also

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Overall Designer architecture
- [NEXT_STEPS.md](NEXT_STEPS.md) - Future development plans
- [../TODO.txt](../TODO.txt) - Task tracking
