package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

// CLIDefinitionsManager handles loading and saving CLI definitions
type CLIDefinitionsManager struct {
	definitionsPath string
	definitions     map[string]*CLIDefinition
}

// NewCLIDefinitionsManager creates a new CLI definitions manager
func NewCLIDefinitionsManager(definitionsPath string) *CLIDefinitionsManager {
	return &CLIDefinitionsManager{
		definitionsPath: definitionsPath,
		definitions:     make(map[string]*CLIDefinition),
	}
}

// LoadAllDefinitions loads all CLI definitions from the definitions directory
func (m *CLIDefinitionsManager) LoadAllDefinitions() error {
	// Ensure directory exists
	if _, err := os.Stat(m.definitionsPath); os.IsNotExist(err) {
		if err := os.MkdirAll(m.definitionsPath, 0755); err != nil {
			return fmt.Errorf("failed to create definitions directory: %w", err)
		}
		return nil // Empty directory
	}

	// Read all JSON files in the directory
	files, err := ioutil.ReadDir(m.definitionsPath)
	if err != nil {
		return fmt.Errorf("failed to read definitions directory: %w", err)
	}

	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		filePath := filepath.Join(m.definitionsPath, file.Name())
		definition, err := m.LoadDefinition(filePath)
		if err != nil {
			fmt.Printf("Warning: failed to load definition %s: %v\n", file.Name(), err)
			continue
		}

		m.definitions[definition.ID] = definition
	}

	return nil
}

// LoadDefinition loads a single CLI definition from a JSON file
func (m *CLIDefinitionsManager) LoadDefinition(filePath string) (*CLIDefinition, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read definition file: %w", err)
	}

	var definition CLIDefinition
	if err := json.Unmarshal(data, &definition); err != nil {
		return nil, fmt.Errorf("failed to parse definition JSON: %w", err)
	}

	return &definition, nil
}

// SaveDefinition saves a CLI definition to a JSON file
func (m *CLIDefinitionsManager) SaveDefinition(definition *CLIDefinition) error {
	// Ensure directory exists
	if err := os.MkdirAll(m.definitionsPath, 0755); err != nil {
		return fmt.Errorf("failed to create definitions directory: %w", err)
	}

	filePath := filepath.Join(m.definitionsPath, definition.ID+".json")

	// Marshal to JSON with indentation
	data, err := json.MarshalIndent(definition, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal definition: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write definition file: %w", err)
	}

	// Update in-memory cache
	m.definitions[definition.ID] = definition

	return nil
}

// GetDefinition retrieves a CLI definition by ID
func (m *CLIDefinitionsManager) GetDefinition(id string) (*CLIDefinition, error) {
	definition, exists := m.definitions[id]
	if !exists {
		return nil, fmt.Errorf("definition not found: %s", id)
	}
	return definition, nil
}

// GetAllDefinitions returns all loaded CLI definitions
func (m *CLIDefinitionsManager) GetAllDefinitions() []*CLIDefinition {
	definitions := make([]*CLIDefinition, 0, len(m.definitions))
	for _, def := range m.definitions {
		definitions = append(definitions, def)
	}
	return definitions
}

// GetDefinitionsByCategory returns definitions filtered by category
func (m *CLIDefinitionsManager) GetDefinitionsByCategory(category string) []*CLIDefinition {
	definitions := make([]*CLIDefinition, 0)
	for _, def := range m.definitions {
		if def.Category == category {
			definitions = append(definitions, def)
		}
	}
	return definitions
}

// CreateNodeFromDefinition creates a new CLINode from a CLI definition
func CreateNodeFromDefinition(definition *CLIDefinition, position Point) *CLINode {
	node := &CLINode{
		ID:            generateUUID(),
		DefinitionID:  definition.ID,
		Name:          definition.Name,
		Position:      position,
		Size:          Size{Width: 300, Height: 150},
		Environment:   definition.Environment,
		Executable:    definition.Executable,
		Script:        definition.Script,
		InputSockets:  make([]Socket, 0),
		OutputSockets: make([]Socket, 0),
		Category:      definition.Category,
		TestStatus:    TestNotRun,
	}

	// Create sockets from argument definitions
	for _, argDef := range definition.Arguments {
		socket := Socket{
			ID:           generateUUID(),
			NodeID:       node.ID,
			ArgumentFlag: argDef.Flag,
			Type:         argDef.Type,
			SocketSide:   argDef.SocketSide,
			Value:        argDef.DefaultValue,
			IsRequired:   argDef.IsRequired,
			DefaultValue: argDef.DefaultValue,
			Description:  argDef.Description,
			Validation:   argDef.Validation,
		}

		if socket.SocketSide == SocketInput {
			node.InputSockets = append(node.InputSockets, socket)
		} else {
			node.OutputSockets = append(node.OutputSockets, socket)
		}
	}

	return node
}

// generateUUID generates a proper UUID
func generateUUID() string {
	return uuid.New().String()
}
