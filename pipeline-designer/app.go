package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

// App struct
type App struct {
	ctx                context.Context
	definitionsManager *CLIDefinitionsManager
	startupFilePath    string
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// GetStartupFilePath returns the file path passed via command line
func (a *App) GetStartupFilePath() string {
	return a.startupFilePath
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx

	// Initialize CLI definitions manager
	// Try multiple locations for cli_definitions folder
	possiblePaths := []string{
		filepath.Join(".", "cli_definitions"),                 // Next to executable
		filepath.Join("..", "cli_definitions"),                // One level up
		filepath.Join("pipeline-designer", "cli_definitions"), // In project folder
	}

	var definitionsPath string
	for _, path := range possiblePaths {
		absPath, _ := filepath.Abs(path)
		if _, err := filepath.Glob(filepath.Join(absPath, "*.json")); err == nil {
			// Check if directory exists and has JSON files
			if files, err := filepath.Glob(filepath.Join(absPath, "*.json")); err == nil && len(files) > 0 {
				definitionsPath = absPath
				break
			}
		}
	}

	if definitionsPath == "" {
		// Default to current directory
		definitionsPath = filepath.Join(".", "cli_definitions")
	}

	fmt.Printf("Loading CLI definitions from: %s\n", definitionsPath)

	a.definitionsManager = NewCLIDefinitionsManager(definitionsPath)

	// Load all definitions
	if err := a.definitionsManager.LoadAllDefinitions(); err != nil {
		fmt.Printf("Error loading CLI definitions: %v\n", err)
	} else {
		defs := a.definitionsManager.GetAllDefinitions()
		fmt.Printf("Successfully loaded %d CLI definitions\n", len(defs))
		for _, def := range defs {
			fmt.Printf("  - %s (%s)\n", def.Name, def.ID)
		}
	}
} // LoadPipeline loads a YAML pipeline file and converts it to visual format
func (a *App) LoadPipeline(filePath string) (*Pipeline, error) {
	return LoadYAMLPipeline(filePath)
}

// SavePipeline saves a visual pipeline back to YAML format
func (a *App) SavePipeline(pipeline *Pipeline, filePath string) error {
	return SaveYAMLPipeline(pipeline, filePath)
}

// GetCLIDefinitions returns all available CLI tool definitions
func (a *App) GetCLIDefinitions() []*CLIDefinition {
	return a.definitionsManager.GetAllDefinitions()
}

// GetCLIDefinitionsByCategory returns definitions filtered by category
func (a *App) GetCLIDefinitionsByCategory(category string) []*CLIDefinition {
	return a.definitionsManager.GetDefinitionsByCategory(category)
}

// CreateNodeFromDefinition creates a new node from a CLI definition
func (a *App) CreateNodeFromDefinition(definitionID string, x float64, y float64) (*CLINode, error) {
	definition, err := a.definitionsManager.GetDefinition(definitionID)
	if err != nil {
		return nil, err
	}

	node := CreateNodeFromDefinition(definition, Point{X: x, Y: y})
	return node, nil
}

// Greet returns a greeting for the given name
func (a *App) Greet(name string) string {
	return fmt.Sprintf("Hello %s, It's show time!", name)
}

// GetFilesFromPattern queries files using Python's get_files_to_process2
func (a *App) GetFilesFromPattern(searchPattern string, searchSubfolders bool) (*FileListResult, error) {
	return GetFilesFromPattern(searchPattern, searchSubfolders)
}

// GetFileListPreview returns a preview of matched files
func (a *App) GetFileListPreview(searchPattern string, searchSubfolders bool) string {
	return GetFileListPreview(searchPattern, searchSubfolders)
}

// OpenFileDialog shows a file picker dialog for opening YAML files
func (a *App) OpenFileDialog() (string, error) {
	filePath, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Open Pipeline",
		Filters: []runtime.FileFilter{
			{DisplayName: "YAML Files (*.yaml, *.yml)", Pattern: "*.yaml;*.yml"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	return filePath, err
}

// SaveFileDialog shows a file picker dialog for saving YAML files
func (a *App) SaveFileDialog(defaultFilename string) (string, error) {
	filePath, err := runtime.SaveFileDialog(a.ctx, runtime.SaveDialogOptions{
		Title:           "Save Pipeline",
		DefaultFilename: defaultFilename,
		Filters: []runtime.FileFilter{
			{DisplayName: "YAML Files (*.yaml)", Pattern: "*.yaml"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	return filePath, err
}

// CreateEmptyPipeline creates an empty YAML pipeline file
func (a *App) CreateEmptyPipeline(filePath string) error {
	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Create empty pipeline YAML
	emptyYAML := `# Pipeline created with Pipeline Designer
run:
  # Add your pipeline steps here
`

	if err := os.WriteFile(filePath, []byte(emptyYAML), 0644); err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}

	return nil
}
