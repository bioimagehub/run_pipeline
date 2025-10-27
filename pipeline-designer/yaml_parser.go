package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

// LoadYAMLPipeline reads a YAML pipeline file and converts it to our visual Pipeline format
func LoadYAMLPipeline(filePath string, definitionsManager *CLIDefinitionsManager) (*Pipeline, error) {
	// Read the YAML file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read YAML file: %w", err)
	}

	// Parse YAML
	var yamlPipeline YAMLPipeline
	if err := yaml.Unmarshal(data, &yamlPipeline); err != nil {
		return nil, fmt.Errorf("failed to parse YAML: %w", err)
	}

	// Check if this is a legacy YAML (no _designer_metadata)
	if yamlPipeline.DesignerMetadata == nil {
		appLogger.Println("[YAML_LOAD] Legacy YAML detected (no _designer_metadata). Using legacy importer...")

		// Use legacy importer with the provided definitions manager
		pipeline, report, err := ImportLegacyYAML(&yamlPipeline, definitionsManager)
		if err != nil {
			return nil, fmt.Errorf("legacy import failed: %w", err)
		}

		// Print detailed report
		PrintLegacyImportReport(report)

		// Fail if there are critical errors
		if len(report.Errors) > 0 {
			appLogger.Println("[YAML_LOAD] ⚠ Legacy import completed with errors. Please review the report above.")
			appLogger.Println("[YAML_LOAD] The pipeline has been loaded, but you should verify all nodes and connections.")
		}

		return pipeline, nil
	}

	// Convert to visual Pipeline (modern YAML with designer metadata)
	pipeline := &Pipeline{
		Nodes:       make([]CLINode, 0),
		Connections: make([]SocketConnection, 0),
		Metadata: PipelineMetadata{
			Name:     yamlPipeline.PipelineName,
			Modified: time.Now(),
		},
	}

	// Create map of node IDs from designer metadata
	nodePositions := make(map[string]Point)
	nodeSizes := make(map[string]Size)
	if yamlPipeline.DesignerMetadata != nil {
		for _, nodeMeta := range yamlPipeline.DesignerMetadata.Nodes {
			nodePositions[nodeMeta.ID] = nodeMeta.Position
			nodeSizes[nodeMeta.ID] = nodeMeta.Size
		}
	}

	// Convert each YAML step to a visual node
	for i, step := range yamlPipeline.Run {
		nodeID := step.NodeID
		if nodeID == "" {
			nodeID = uuid.New().String()
		}

		// Get position and size from metadata, or use default
		position := Point{X: 100, Y: float64(i * 250)}
		if pos, ok := nodePositions[nodeID]; ok {
			position = pos
		}

		size := Size{Width: 300, Height: 150}
		if sz, ok := nodeSizes[nodeID]; ok {
			size = sz
		}

		node := CLINode{
			ID:            nodeID,
			Name:          step.Name,
			Environment:   step.Environment,
			Script:        step.Script,
			Position:      position,
			Size:          size,
			InputSockets:  make([]Socket, 0),
			OutputSockets: make([]Socket, 0),
			Category:      inferCategory(step.Name),
			TestStatus:    TestNotRun,
		}

		// Parse commands into sockets
		if step.Commands != nil {
			parseCommandsToSockets(&node, step.Commands)
		}

		// Also handle legacy args format
		for flag, value := range step.Args {
			socket := Socket{
				ID:           uuid.New().String(),
				NodeID:       node.ID,
				ArgumentFlag: flag,
				Value:        value,
				Type:         inferArgumentType(flag, value),
				SocketSide:   inferSocketSide(flag),
			}

			if socket.SocketSide == SocketInput {
				node.InputSockets = append(node.InputSockets, socket)
			} else {
				node.OutputSockets = append(node.OutputSockets, socket)
			}
		}

		// Note: No longer injecting __link_in__/__link_out__ sockets
		// Output sockets are now properly defined in CLI definition JSON files

		pipeline.Nodes = append(pipeline.Nodes, node)
	}

	// Restore connections from metadata
	if yamlPipeline.DesignerMetadata != nil {
		for _, connMeta := range yamlPipeline.DesignerMetadata.Connections {
			connection := SocketConnection{
				ID:           connMeta.ID,
				FromNodeID:   connMeta.FromNodeID,
				ToNodeID:     connMeta.ToNodeID,
				FromSocketID: connMeta.FromSocketID,
				ToSocketID:   connMeta.ToSocketID,
				IsValid:      true,
			}
			pipeline.Connections = append(pipeline.Connections, connection)
		}
	}

	return pipeline, nil
}

// SaveYAMLPipeline converts our visual Pipeline format back to YAML
func SaveYAMLPipeline(pipeline *Pipeline, filePath string) error {
	yamlPipeline := YAMLPipeline{
		PipelineName: pipeline.Metadata.Name,
		Run:          make([]YAMLStep, 0),
		DesignerMetadata: &DesignerMetadata{
			Nodes:       make([]VisualNodeMetadata, 0),
			Connections: make([]VisualConnectionMetadata, 0),
		},
	}

	// Save visual node metadata
	for _, node := range pipeline.Nodes {
		nodeMeta := VisualNodeMetadata{
			ID:       node.ID,
			Position: node.Position,
			Size:     node.Size,
		}
		yamlPipeline.DesignerMetadata.Nodes = append(yamlPipeline.DesignerMetadata.Nodes, nodeMeta)

		// Convert node to YAML step with commands and expected output files
		commands, expectedOutputFiles := convertSocketsToCommandsAndOutputs(node, pipeline)

		step := YAMLStep{
			Name:                node.Name,
			Environment:         node.Environment,
			Script:              node.Script,
			NodeID:              node.ID,
			Commands:            commands,
			ExpectedOutputFiles: expectedOutputFiles,
		}

		yamlPipeline.Run = append(yamlPipeline.Run, step)
	}

	// Save connections metadata
	for _, conn := range pipeline.Connections {
		connMeta := VisualConnectionMetadata{
			ID:           conn.ID,
			FromNodeID:   conn.FromNodeID,
			ToNodeID:     conn.ToNodeID,
			FromSocketID: conn.FromSocketID,
			ToSocketID:   conn.ToSocketID,
		}
		yamlPipeline.DesignerMetadata.Connections = append(yamlPipeline.DesignerMetadata.Connections, connMeta)
	}

	// Marshal to YAML
	data, err := yaml.Marshal(&yamlPipeline)
	if err != nil {
		return fmt.Errorf("failed to marshal YAML: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write YAML file: %w", err)
	}

	return nil
}

// parseCommandsToSockets converts YAML commands array to input/output sockets
func parseCommandsToSockets(node *CLINode, commands interface{}) {
	switch cmds := commands.(type) {
	case []interface{}:
		for _, cmd := range cmds {
			switch v := cmd.(type) {
			case string:
				// Plain string command (like "python" or script path)
				if v == "python" || v == "imagej" {
					// Skip interpreter commands
					continue
				}
				// Script path - store in node.Script
				node.Script = v
			case map[string]interface{}:
				// Flag-value pairs
				for flag, value := range v {
					strValue := fmt.Sprintf("%v", value)
					socket := Socket{
						ID:           uuid.New().String(),
						NodeID:       node.ID,
						ArgumentFlag: flag,
						Value:        strValue,
						Type:         inferArgumentType(flag, strValue),
						SocketSide:   inferSocketSide(flag),
					}

					if socket.SocketSide == SocketInput {
						node.InputSockets = append(node.InputSockets, socket)
					} else {
						node.OutputSockets = append(node.OutputSockets, socket)
					}
				}
			}
		}
	}
}

// convertSocketsToCommandsAndOutputs converts node sockets back to YAML commands format
// Returns commands (input sockets only) and expected output files (output sockets)
func convertSocketsToCommandsAndOutputs(node CLINode, pipeline *Pipeline) ([]interface{}, map[string]string) {
	commands := make([]interface{}, 0)
	expectedOutputFiles := make(map[string]string)

	// Add python interpreter if needed
	if node.Environment != "" && node.Environment != "imageJ" {
		commands = append(commands, "python")
	}

	// Add script path
	if node.Script != "" {
		commands = append(commands, node.Script)
	}

	// Add input socket arguments
	for _, socket := range node.InputSockets {
		if socket.SkipEmit {
			continue
		}

		// Use socket.Value if set, otherwise use defaultValue
		value := socket.Value
		if value == "" {
			value = socket.DefaultValue
		}

		// Skip empty values (typical for bool/flag arguments like --dry-run, --version, etc.)
		if value == "" {
			continue
		}

		// Add as map entry
		argMap := make(map[string]interface{})
		argMap[socket.ArgumentFlag] = value
		commands = append(commands, argMap)
	}

	// Add output socket values to expected_output_files instead of commands
	for _, socket := range node.OutputSockets {
		if socket.SkipEmit {
			continue
		}

		// Use socket.Value if set, otherwise use defaultValue (which should already be resolved)
		value := socket.Value
		if value == "" {
			value = socket.DefaultValue
		}

		// Skip empty values (though output files should typically have values)
		if value == "" {
			continue
		}

		// Add to expected output files map
		expectedOutputFiles[socket.ArgumentFlag] = value
	}

	return commands, expectedOutputFiles
}

// resolveOutputSocketPlaceholders resolves placeholders in output socket values
// based on the node's input socket values
func resolveOutputSocketPlaceholders(outputValue string, node CLINode) string {
	if outputValue == "" {
		return outputValue
	}

	resolved := outputValue

	// Iterate through all input sockets to find values to substitute
	for _, inputSocket := range node.InputSockets {
		// Get the input socket value (prefer value, fallback to defaultValue)
		inputValue := inputSocket.Value
		if inputValue == "" {
			inputValue = inputSocket.DefaultValue
		}

		// Convert flag to placeholder format: --output-folder -> <output_folder>
		flagName := inputSocket.ArgumentFlag
		flagName = strings.TrimPrefix(flagName, "--")
		flagName = strings.ReplaceAll(flagName, "-", "_")

		// Handle transformations like <input_search_pattern:dirname>
		transformPattern := fmt.Sprintf("<%s:([^>]+)>", flagName)
		re := regexp.MustCompile(transformPattern)
		resolved = re.ReplaceAllStringFunc(resolved, func(match string) string {
			submatch := re.FindStringSubmatch(match)
			if len(submatch) > 1 {
				transform := submatch[1]
				return applyTransform(inputValue, transform)
			}
			return match
		})

		// Simple placeholder replacement: <output_folder> -> actual value
		placeholder := fmt.Sprintf("<%s>", flagName)
		resolved = strings.ReplaceAll(resolved, placeholder, inputValue)
	}

	return resolved
}

// applyTransform applies a transformation to a value
func applyTransform(value, transform string) string {
	if value == "" {
		return ""
	}

	switch transform {
	case "dirname":
		// Extract directory name from glob pattern
		// e.g., "%YAML%/input/**/*.nd2" -> "%YAML%/input"
		dir := value

		// Find the position of glob patterns (*, ?, [)
		globChars := []rune{'*', '?', '['}
		globIndex := -1
		for i, ch := range dir {
			for _, globCh := range globChars {
				if ch == globCh {
					globIndex = i
					break
				}
			}
			if globIndex != -1 {
				break
			}
		}

		if globIndex != -1 {
			// Get everything before the glob pattern
			dir = dir[:globIndex]
			// Remove any trailing slashes or partial path segments
			dir = strings.TrimRight(dir, "/\\")
		}

		return dir

	case "basename":
		// Extract filename without extension
		parts := strings.Split(value, "/")
		filename := parts[len(parts)-1]
		extIndex := strings.LastIndex(filename, ".")
		if extIndex != -1 {
			filename = filename[:extIndex]
		}
		return filename

	default:
		return value
	}
}

// resolveSocketValue finds the value of a connected socket
func resolveSocketValue(socketID string, pipeline *Pipeline) (string, error) {
	// Find the connection
	var connection *SocketConnection
	for _, conn := range pipeline.Connections {
		if conn.ToSocketID == socketID {
			connection = &conn
			break
		}
	}

	if connection == nil {
		return "", fmt.Errorf("no connection found for socket %s", socketID)
	}

	// Find the source socket
	for _, node := range pipeline.Nodes {
		for _, socket := range node.OutputSockets {
			if socket.ID == connection.FromSocketID {
				return socket.Value, nil
			}
		}
	}

	return "", fmt.Errorf("source socket not found")
}

// inferCategory guesses the category based on the step name
func inferCategory(name string) string {
	nameLower := filepath.Base(name)

	if contains(nameLower, "segment") {
		return "Segmentation"
	} else if contains(nameLower, "convert") || contains(nameLower, "drift") || contains(nameLower, "merge") {
		return "Image Processing"
	} else if contains(nameLower, "measure") || contains(nameLower, "distance") {
		return "Measurement"
	} else if contains(nameLower, "track") {
		return "Tracking"
	} else if contains(nameLower, "plot") || contains(nameLower, "roi") {
		return "Visualization"
	}

	return "Utilities"
}

// inferArgumentType guesses the argument type based on flag name and value
func inferArgumentType(flag, value string) ArgumentType {
	flagLower := flag

	if contains(flagLower, "pattern") {
		return TypeGlobPattern
	} else if contains(flagLower, "folder") || contains(flagLower, "path") || contains(flagLower, "dir") || contains(flagLower, "file") {
		return TypePath
	} else if value == "true" || value == "false" {
		return TypeBool
	}

	// Try to parse as int or float
	var intVal int
	if _, err := fmt.Sscanf(value, "%d", &intVal); err == nil {
		return TypeInt
	}

	var floatVal float64
	if _, err := fmt.Sscanf(value, "%f", &floatVal); err == nil {
		return TypeFloat
	}

	return TypeString
}

// inferSocketSide determines if an argument should be an input or output socket
func inferSocketSide(flag string) SocketSide {
	flagLower := flag

	// Output patterns
	if contains(flagLower, "output") || contains(flagLower, "destination") || contains(flagLower, "save") || contains(flagLower, "export") {
		return SocketOutput
	}

	// Input patterns (default)
	return SocketInput
}

// contains checks if a string contains a substring (case-insensitive helper)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && s[:len(substr)] == substr || s[len(s)-len(substr):] == substr)
}

// ConvertNodeToYAMLStep converts a single node to a YAML step (without connections)
func ConvertNodeToYAMLStep(node *CLINode) *YAMLStep {
	commands := make([]interface{}, 0)

	// Add python interpreter if needed
	if node.Environment != "" && node.Environment != "imageJ" {
		commands = append(commands, "python")
	}

	// Add script path
	if node.Script != "" {
		commands = append(commands, node.Script)
	}

	// Add input socket arguments
	for _, socket := range node.InputSockets {
		if !socket.SkipEmit && socket.Value != "" {
			argMap := make(map[string]interface{})
			argMap[socket.ArgumentFlag] = socket.Value
			commands = append(commands, argMap)
		}
	}

	// Add output socket arguments
	for _, socket := range node.OutputSockets {
		if !socket.SkipEmit && socket.Value != "" {
			argMap := make(map[string]interface{})
			argMap[socket.ArgumentFlag] = socket.Value
			commands = append(commands, argMap)
		}
	}

	step := &YAMLStep{
		Name:        node.Name,
		Environment: node.Environment,
		Script:      node.Script,
		Commands:    commands,
	}

	return step
}

// WriteYAMLPipeline writes a YAML pipeline to a file
func WriteYAMLPipeline(yamlPipeline *YAMLPipeline, filePath string) error {
	// Marshal to YAML
	data, err := yaml.Marshal(yamlPipeline)
	if err != nil {
		return fmt.Errorf("failed to marshal YAML: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write YAML file: %w", err)
	}

	return nil
}
