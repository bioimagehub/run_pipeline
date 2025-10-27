package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

// LoadAllDefinitions loads all CLI definitions from the definitions directory (recursively)
func (m *CLIDefinitionsManager) LoadAllDefinitions() error {
	// Ensure directory exists
	if _, err := os.Stat(m.definitionsPath); os.IsNotExist(err) {
		if err := os.MkdirAll(m.definitionsPath, 0755); err != nil {
			return fmt.Errorf("failed to create definitions directory: %w", err)
		}
		return nil // Empty directory
	}

	// Walk through all subdirectories recursively
	err := filepath.Walk(m.definitionsPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and non-JSON files
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".json") {
			return nil
		}

		// Derive category from folder structure
		relPath, err := filepath.Rel(m.definitionsPath, path)
		if err != nil {
			appLogger.Printf("[ERROR] Failed to get relative path for %s: %v\n", path, err)
			return nil
		}

		category := deriveCategoryFromPath(relPath)

		if debugMode {
			appLogger.Printf("[DEBUG] Loading definition file: %s (category: %s)\n", info.Name(), category)
		}

		definition, err := m.LoadDefinitionWithCategory(path, category)
		if err != nil {
			appLogger.Printf("[ERROR] Failed to load definition %s: %v\n", info.Name(), err)
			return nil
		}

		m.definitions[definition.ID] = definition
		if debugMode {
			appLogger.Printf("[DEBUG] Successfully loaded definition: %s (ID: %s)\n", definition.Name, definition.ID)
		}
		return nil
	})

	return err
}

// deriveCategoryFromPath converts a file path to a category string
// Example: "Image Processing/Filtering/gaussian_filter.json" -> "Image Processing > Filtering"
func deriveCategoryFromPath(relPath string) string {
	dir := filepath.Dir(relPath)

	// If file is in root directory, use "Uncategorized"
	if dir == "." {
		return "Uncategorized"
	}

	// Replace path separators with " > " for hierarchical display
	category := strings.ReplaceAll(dir, string(filepath.Separator), " > ")
	return category
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

// LoadDefinitionWithCategory loads a CLI definition and sets its category
func (m *CLIDefinitionsManager) LoadDefinitionWithCategory(filePath string, category string) (*CLIDefinition, error) {
	definition, err := m.LoadDefinition(filePath)
	if err != nil {
		return nil, err
	}

	// Override category with folder-based category
	definition.Category = category

	return definition, nil
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
		Icon:          definition.Icon,
		Color:         definition.Color,
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
			Value:        "", // Always start with empty value - use DefaultValue for placeholders
			IsRequired:   argDef.IsRequired,
			DefaultValue: argDef.DefaultValue,
			Description:  argDef.Description,
			Validation:   argDef.Validation,
			SkipEmit:     argDef.SkipEmit,
		}

		if socket.SocketSide == SocketInput {
			node.InputSockets = append(node.InputSockets, socket)
		} else {
			node.OutputSockets = append(node.OutputSockets, socket)
		}
	}

	// Resolve placeholder defaults in output sockets based on input socket defaults
	resolveDefaultPlaceholders(node)

	return node
}

// resolveDefaultPlaceholders resolves placeholder templates in output socket defaults
// This is called ONCE when a node is created to set up proper default values
func resolveDefaultPlaceholders(node *CLINode) {
	// Build a map of input socket default values for quick lookup
	inputDefaults := make(map[string]string)
	for _, inputSocket := range node.InputSockets {
		// Convert flag to placeholder format: --output-folder -> output_folder
		flagName := strings.TrimPrefix(inputSocket.ArgumentFlag, "--")
		flagName = strings.ReplaceAll(flagName, "-", "_")
		inputDefaults[flagName] = inputSocket.DefaultValue
	}

	// Resolve placeholders in output socket defaults
	for i := range node.OutputSockets {
		socket := &node.OutputSockets[i]
		if socket.DefaultValue == "" {
			continue
		}

		// Check if default value contains placeholders
		if !strings.Contains(socket.DefaultValue, "<") || !strings.Contains(socket.DefaultValue, ">") {
			continue
		}

		resolved := socket.DefaultValue

		// Replace all placeholders in the default value
		for flagName, inputDefault := range inputDefaults {
			// Handle transformations like <input_search_pattern:dirname>
			transformPattern := fmt.Sprintf("<%s:([^>]+)>", flagName)
			re := regexp.MustCompile(transformPattern)
			resolved = re.ReplaceAllStringFunc(resolved, func(match string) string {
				submatch := re.FindStringSubmatch(match)
				if len(submatch) > 1 {
					transform := submatch[1]
					return applyDefaultTransform(inputDefault, transform)
				}
				return match
			})

			// Simple placeholder replacement: <output_folder> -> actual value
			placeholder := fmt.Sprintf("<%s>", flagName)
			resolved = strings.ReplaceAll(resolved, placeholder, inputDefault)
		}

		// Update the default value with the resolved version
		socket.DefaultValue = resolved
	}
}

// applyDefaultTransform applies transformations for default value resolution
func applyDefaultTransform(value, transform string) string {
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

// generateUUID generates a proper UUID
func generateUUID() string {
	return uuid.New().String()
}
