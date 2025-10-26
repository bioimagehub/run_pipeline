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

	// Convert to visual Pipeline
	pipeline := &Pipeline{
		Nodes:       make([]CLINode, 0),
		Connections: make([]SocketConnection, 0),
		Metadata: PipelineMetadata{
			Name:     yamlPipeline.PipelineName,
			Modified: time.Now(),
		},
	}

	// Convert each YAML step to a visual node
	for i, step := range yamlPipeline.Run {
		node := CLINode{
			ID:            uuid.New().String(),
			Name:          step.Name,
			Environment:   step.Environment,
			Script:        step.Script,
			Position:      Point{X: 100, Y: float64(i * 200)}, // Stack vertically
			Size:          Size{Width: 300, Height: 150},
			InputSockets:  make([]Socket, 0),
			OutputSockets: make([]Socket, 0),
			Category:      inferCategory(step.Name),
			TestStatus:    TestNotRun,
		}

		// Convert args to sockets
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

		pipeline.Nodes = append(pipeline.Nodes, node)
	}

	return pipeline, nil
}

// SaveYAMLPipeline converts our visual Pipeline format back to YAML
func SaveYAMLPipeline(pipeline *Pipeline, filePath string) error {
	yamlPipeline := YAMLPipeline{
		PipelineName: pipeline.Metadata.Name,
		Run:          make([]YAMLStep, 0),
	}

	// Convert each node to a YAML step
	for _, node := range pipeline.Nodes {
		step := YAMLStep{
			Name:        node.Name,
			Environment: node.Environment,
			Script:      node.Script,
			Args:        make(map[string]string),
		}

		// Convert sockets to args
		for _, socket := range node.InputSockets {
			value := socket.Value
			// If socket is connected, resolve the value from the connected socket
			if socket.ConnectedTo != nil {
				resolvedValue, err := resolveSocketValue(*socket.ConnectedTo, pipeline)
				if err == nil {
					value = resolvedValue
				}
			}
			step.Args[socket.ArgumentFlag] = value
		}

		for _, socket := range node.OutputSockets {
			value := socket.Value
			// If socket is connected, use the connected value
			if socket.ConnectedTo != nil {
				resolvedValue, err := resolveSocketValue(*socket.ConnectedTo, pipeline)
				if err == nil {
					value = resolvedValue
				}
			}
			step.Args[socket.ArgumentFlag] = value
		}

		yamlPipeline.Run = append(yamlPipeline.Run, step)
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
