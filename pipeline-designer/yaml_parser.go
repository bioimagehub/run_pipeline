package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

// LoadYAMLPipeline reads a YAML pipeline file and converts it to our visual Pipeline format
func LoadYAMLPipeline(filePath string) (*Pipeline, error) {
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

		// Find cli_definitions directory
		exePath, _ := os.Executable()
		exeDir := filepath.Dir(exePath)
		definitionsPath := filepath.Join(exeDir, "cli_definitions")

		// Load all CLI definitions
		definitionsManager := NewCLIDefinitionsManager(definitionsPath)
		if err := definitionsManager.LoadAllDefinitions(); err != nil {
			return nil, fmt.Errorf("failed to load CLI definitions for legacy import: %w", err)
		}

		// Use legacy importer
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

		// Inject default visual-only link sockets (ignored when generating YAML)
		linkIn := Socket{ID: uuid.New().String(), NodeID: node.ID, ArgumentFlag: "__link_in__", Type: TypeString, SocketSide: SocketInput, Value: "", IsRequired: false, DefaultValue: "", Description: "Visual link-only handle (ignored in YAML/command)", Validation: "", SkipEmit: true}
		linkOut := Socket{ID: uuid.New().String(), NodeID: node.ID, ArgumentFlag: "__link_out__", Type: TypeString, SocketSide: SocketOutput, Value: "", IsRequired: false, DefaultValue: "", Description: "Visual link-only handle (ignored in YAML/command)", Validation: "", SkipEmit: true}
		node.InputSockets = append(node.InputSockets, linkIn)
		node.OutputSockets = append(node.OutputSockets, linkOut)

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

		// Convert node to YAML step
		step := YAMLStep{
			Name:        node.Name,
			Environment: node.Environment,
			Script:      node.Script,
			NodeID:      node.ID,
			Commands:    convertSocketsToCommands(node, pipeline),
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

// convertSocketsToCommands converts node sockets back to YAML commands format
func convertSocketsToCommands(node CLINode, pipeline *Pipeline) []interface{} {
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
		if socket.SkipEmit {
			continue
		}
		value := socket.Value

		// Check if this socket is connected
		for _, conn := range pipeline.Connections {
			if conn.ToSocketID == socket.ID {
				// Find the source socket value
				for _, sourceNode := range pipeline.Nodes {
					for _, sourceSocket := range sourceNode.OutputSockets {
						if sourceSocket.ID == conn.FromSocketID {
							value = sourceSocket.Value
							break
						}
					}
				}
			}
		}

		// Add as map entry
		argMap := make(map[string]interface{})
		argMap[socket.ArgumentFlag] = value
		commands = append(commands, argMap)
	}

	// Add output socket arguments
	for _, socket := range node.OutputSockets {
		if socket.SkipEmit {
			continue
		}
		argMap := make(map[string]interface{})
		argMap[socket.ArgumentFlag] = socket.Value
		commands = append(commands, argMap)
	}

	return commands
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
