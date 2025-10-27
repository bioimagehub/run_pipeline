package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

// Global logger and debug mode flag
var (
	appLogger *log.Logger
	debugMode bool
)

// App struct
type App struct {
	ctx                context.Context
	definitionsManager *CLIDefinitionsManager
	startupFilePath    string
}

// NewApp creates a new App application struct
func NewApp() *App {
	// Get executable directory for log file
	exePath, err := os.Executable()
	if err != nil {
		exePath = "."
	}
	exeDir := filepath.Dir(exePath)
	logFilePath := filepath.Join(exeDir, "pipeline-designer.log")

	// Open or create log file (append mode)
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		// Fallback to temp directory
		logFilePath = filepath.Join(os.TempDir(), "pipeline-designer.log")
		logFile, _ = os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	}

	if logFile != nil {
		// Initialize logger with file output
		appLogger = log.New(logFile, "[PIPELINE-DESIGNER] ", log.Ldate|log.Ltime|log.Lshortfile)
		// Note: We don't close the file here as it needs to stay open for the app lifetime
	} else {
		// Fallback to stderr if file creation fails
		appLogger = log.New(os.Stderr, "[PIPELINE-DESIGNER] ", log.Ldate|log.Ltime|log.Lshortfile)
	}

	// Check for debug mode via environment variable
	if os.Getenv("DEBUG") != "" || os.Getenv("PIPELINE_DEBUG") != "" {
		debugMode = true
		appLogger.Println("DEBUG MODE ENABLED")
	}

	appLogger.Println("App instance created")
	return &App{}
} // GetStartupFilePath returns the file path passed via command line
func (a *App) GetStartupFilePath() string {
	return a.startupFilePath
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	appLogger.Println("Application startup initiated")
	if debugMode {
		appLogger.Printf("[DEBUG] Context initialized: %v\n", ctx != nil)
	}

	// Initialize CLI definitions manager
	// Try multiple locations for cli_definitions folder
	possiblePaths := []string{
		filepath.Join(".", "cli_definitions"),                 // Next to executable
		filepath.Join("..", "cli_definitions"),                // One level up
		filepath.Join("pipeline-designer", "cli_definitions"), // In project folder
	}

	if debugMode {
		appLogger.Printf("[DEBUG] Searching for cli_definitions in %d locations\n", len(possiblePaths))
	}

	var definitionsPath string
	for i, path := range possiblePaths {
		absPath, _ := filepath.Abs(path)
		if debugMode {
			appLogger.Printf("[DEBUG] Checking path %d: %s\n", i+1, absPath)
		}
		if _, err := filepath.Glob(filepath.Join(absPath, "*.json")); err == nil {
			// Check if directory exists and has JSON files
			if files, err := filepath.Glob(filepath.Join(absPath, "*.json")); err == nil && len(files) > 0 {
				definitionsPath = absPath
				if debugMode {
					appLogger.Printf("[DEBUG] Found definitions path with %d JSON files\n", len(files))
				}
				break
			}
		}
	}

	if definitionsPath == "" {
		// Default to current directory
		definitionsPath = filepath.Join(".", "cli_definitions")
		appLogger.Printf("[WARN] No definitions found, using default: %s\n", definitionsPath)
	}

	appLogger.Printf("Loading CLI definitions from: %s\n", definitionsPath)

	a.definitionsManager = NewCLIDefinitionsManager(definitionsPath)

	// Load all definitions
	if err := a.definitionsManager.LoadAllDefinitions(); err != nil {
		appLogger.Printf("[ERROR] Failed to load CLI definitions: %v\n", err)
	} else {
		defs := a.definitionsManager.GetAllDefinitions()
		appLogger.Printf("Successfully loaded %d CLI definitions\n", len(defs))
		for _, def := range defs {
			if debugMode {
				appLogger.Printf("[DEBUG]   - %s (%s) [Category: %s, Icon: %s]\n", def.Name, def.ID, def.Category, def.Icon)
			} else {
				appLogger.Printf("  - %s (%s)\n", def.Name, def.ID)
			}
		}
	}
	appLogger.Println("Application startup complete")
} // LoadPipeline loads a YAML pipeline file and converts it to visual format
func (a *App) LoadPipeline(filePath string) (*Pipeline, error) {
	appLogger.Printf("Loading pipeline from: %s\n", filePath)
	if debugMode {
		appLogger.Printf("[DEBUG] LoadPipeline called with path: %s\n", filePath)
	}
	pipeline, err := LoadYAMLPipeline(filePath)
	if err != nil {
		appLogger.Printf("[ERROR] Failed to load pipeline: %v\n", err)
		return nil, err
	}
	if debugMode {
		appLogger.Printf("[DEBUG] Pipeline loaded successfully with %d nodes and %d connections\n", len(pipeline.Nodes), len(pipeline.Connections))
	}
	return pipeline, nil
}

// SavePipeline saves a visual pipeline back to YAML format
func (a *App) SavePipeline(pipeline *Pipeline, filePath string) error {
	appLogger.Printf("Saving pipeline to: %s\n", filePath)
	if debugMode {
		appLogger.Printf("[DEBUG] SavePipeline called with %d nodes, %d connections\n", len(pipeline.Nodes), len(pipeline.Connections))
	}
	err := SaveYAMLPipeline(pipeline, filePath)
	if err != nil {
		appLogger.Printf("[ERROR] Failed to save pipeline: %v\n", err)
		return err
	}
	appLogger.Println("Pipeline saved successfully")
	return nil
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
	if debugMode {
		appLogger.Printf("[DEBUG] CreateNodeFromDefinition called: ID=%s, x=%.2f, y=%.2f\n", definitionID, x, y)
	}

	definition, err := a.definitionsManager.GetDefinition(definitionID)
	if err != nil {
		appLogger.Printf("[ERROR] Failed to get definition '%s': %v\n", definitionID, err)
		return nil, err
	}

	if debugMode {
		appLogger.Printf("[DEBUG] Found definition: %s (Category: %s)\n", definition.Name, definition.Category)
	}

	node := CreateNodeFromDefinition(definition, Point{X: x, Y: y})

	if debugMode {
		appLogger.Printf("[DEBUG] Node created: ID=%s, Name=%s, InputSockets=%d, OutputSockets=%d\n",
			node.ID, node.Name, len(node.InputSockets), len(node.OutputSockets))
	}

	appLogger.Printf("Created node '%s' at position (%.0f, %.0f)\n", node.Name, x, y)
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
	if debugMode {
		appLogger.Println("[DEBUG] OpenFileDialog called")
	}
	filePath, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Open Pipeline",
		Filters: []runtime.FileFilter{
			{DisplayName: "YAML Files (*.yaml, *.yml)", Pattern: "*.yaml;*.yml"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	if err != nil {
		appLogger.Printf("[ERROR] OpenFileDialog error: %v\n", err)
	} else if filePath != "" {
		appLogger.Printf("File selected: %s\n", filePath)
	} else {
		if debugMode {
			appLogger.Println("[DEBUG] OpenFileDialog cancelled")
		}
	}
	return filePath, err
}

// SaveFileDialog shows a file picker dialog for saving YAML files
func (a *App) SaveFileDialog(defaultFilename string) (string, error) {
	if debugMode {
		appLogger.Printf("[DEBUG] SaveFileDialog called with default: %s\n", defaultFilename)
	}
	filePath, err := runtime.SaveFileDialog(a.ctx, runtime.SaveDialogOptions{
		Title:           "Save Pipeline",
		DefaultFilename: defaultFilename,
		Filters: []runtime.FileFilter{
			{DisplayName: "YAML Files (*.yaml)", Pattern: "*.yaml"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	if err != nil {
		appLogger.Printf("[ERROR] SaveFileDialog error: %v\n", err)
	} else if filePath != "" {
		appLogger.Printf("Save location selected: %s\n", filePath)
	} else {
		if debugMode {
			appLogger.Println("[DEBUG] SaveFileDialog cancelled")
		}
	}
	return filePath, err
}

// CreateEmptyPipeline creates an empty YAML pipeline file
func (a *App) CreateEmptyPipeline(filePath string) error {
	if debugMode {
		appLogger.Printf("[DEBUG] CreateEmptyPipeline called: %s\n", filePath)
	}

	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		appLogger.Printf("[ERROR] Failed to create directory '%s': %v\n", dir, err)
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Create empty pipeline YAML
	emptyYAML := `# Pipeline created with Pipeline Designer
run:
  # Add your pipeline steps here
`

	if err := os.WriteFile(filePath, []byte(emptyYAML), 0644); err != nil {
		appLogger.Printf("[ERROR] Failed to create file '%s': %v\n", filePath, err)
		return fmt.Errorf("failed to create file: %w", err)
	}

	appLogger.Printf("Empty pipeline created: %s\n", filePath)
	return nil
}

// LogFrontend logs messages from the frontend to the backend log file
func (a *App) LogFrontend(message string) {
	if appLogger != nil {
		appLogger.Printf("[FRONTEND] %s\n", message)
	}
}
