package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
	"gopkg.in/yaml.v3"
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

	// Delete existing log file if it exists to prevent it from growing too large
	if _, err := os.Stat(logFilePath); err == nil {
		os.Remove(logFilePath)
	}

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
	// Get executable directory first - this is the most reliable location
	exePath, err := os.Executable()
	if err != nil {
		appLogger.Printf("[ERROR] Failed to get executable path: %v\n", err)
		exePath = "."
	}
	exeDir := filepath.Dir(exePath)

	// Try multiple locations for cli_definitions folder, prioritizing executable directory
	possiblePaths := []string{
		filepath.Join(exeDir, "cli_definitions"),              // Next to executable (PRIORITY)
		filepath.Join(".", "cli_definitions"),                 // Current working directory
		filepath.Join("..", "cli_definitions"),                // One level up
		filepath.Join("pipeline-designer", "cli_definitions"), // In project folder
	}

	if debugMode {
		appLogger.Printf("[DEBUG] Executable directory: %s\n", exeDir)
		appLogger.Printf("[DEBUG] Searching for cli_definitions in %d locations\n", len(possiblePaths))
	}

	var definitionsPath string
	for i, path := range possiblePaths {
		absPath, _ := filepath.Abs(path)
		if debugMode {
			appLogger.Printf("[DEBUG] Checking path %d: %s\n", i+1, absPath)
		}
		// Check if directory exists
		if info, err := os.Stat(absPath); err == nil && info.IsDir() {
			// Check if it has JSON files (in root or subdirectories)
			hasDefinitions := false
			filepath.Walk(absPath, func(path string, info os.FileInfo, err error) error {
				if err == nil && !info.IsDir() && strings.HasSuffix(info.Name(), ".json") {
					hasDefinitions = true
					return filepath.SkipAll // Stop walking once we find one JSON file
				}
				return nil
			})

			if hasDefinitions {
				definitionsPath = absPath
				if debugMode {
					appLogger.Printf("[DEBUG] Found definitions path with JSON files: %s\n", absPath)
				}
				break
			} else if debugMode {
				appLogger.Printf("[DEBUG] Directory exists but no JSON files found\n")
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
	pipeline, err := LoadYAMLPipeline(filePath, a.definitionsManager)
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

// CountFilesMatchingPattern counts files matching a glob pattern
// Supports %YAML%, %REPO% and other path tokens from .env
// yamlFilePath should be the path to the currently open YAML file (for resolving %YAML%)
func (a *App) CountFilesMatchingPattern(pattern string, yamlFilePath string) int {
	if pattern == "" {
		return 0
	}

	// Substitute path tokens (%YAML%, %REPO%, etc.)
	envVars := a.GetEnvVariables()

	// Override YAML with the directory of the current YAML file if provided
	if yamlFilePath != "" {
		envVars["YAML"] = filepath.Dir(yamlFilePath)
	}

	resolvedPattern := pattern
	for key, value := range envVars {
		token := "%" + key + "%"
		resolvedPattern = strings.ReplaceAll(resolvedPattern, token, value)
	}

	// Convert forward slashes to OS-specific separators
	resolvedPattern = filepath.FromSlash(resolvedPattern)

	// Always log the path resolution for debugging
	appLogger.Printf("[COUNT_FILES] Original pattern: '%s'\n", pattern)
	appLogger.Printf("[COUNT_FILES] YAML file path: '%s'\n", yamlFilePath)
	appLogger.Printf("[COUNT_FILES] YAML directory: '%s'\n", envVars["YAML"])
	appLogger.Printf("[COUNT_FILES] Resolved pattern: '%s'\n", resolvedPattern)

	// Use filepath.Glob to match files
	matches, err := filepath.Glob(resolvedPattern)
	if err != nil {
		appLogger.Printf("[COUNT_FILES] Error: %v\n", err)
		return 0
	}

	count := len(matches)
	appLogger.Printf("[COUNT_FILES] Found %d matching files\n", count)
	if count > 0 && count <= 5 {
		// Log first few matches for debugging
		for i, match := range matches {
			appLogger.Printf("[COUNT_FILES]   Match %d: %s\n", i+1, match)
		}
	}

	return count
} // OpenFileDialog shows a file picker dialog for opening YAML files
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

// GetEnvVariables reads the .env file and returns a map of custom variables
func (a *App) GetEnvVariables() map[string]string {
	envVars := make(map[string]string)

	// Get project root directory (where run_pipeline.exe is located)
	exePath, err := os.Executable()
	if err != nil {
		appLogger.Printf("[ERROR] Failed to get executable path: %v\n", err)
		return envVars
	}

	// Navigate up from pipeline-designer/build/bin to project root
	projectRoot := filepath.Dir(exePath)    // bin
	projectRoot = filepath.Dir(projectRoot) // build
	projectRoot = filepath.Dir(projectRoot) // pipeline-designer
	projectRoot = filepath.Dir(projectRoot) // run_pipeline

	// Standard variables that are always present
	envVars["REPO"] = projectRoot

	// YAML directory is typically pipeline_configs
	yamlDir := filepath.Join(projectRoot, "pipeline_configs")
	envVars["YAML"] = yamlDir

	// Read .env file from project root
	envPath := filepath.Join(projectRoot, ".env")
	file, err := os.Open(envPath)
	if err != nil {
		// .env file doesn't exist or can't be read - return defaults only
		appLogger.Printf("[WARN] Could not read .env file: %v\n", err)
		return envVars
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse KEY=VALUE format
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			// Skip standard conda/imagej paths (not useful as path tokens)
			if key == "CONDA_PATH" || key == "IMAGEJ_PATH" {
				continue
			}

			envVars[key] = value
		}
	}

	appLogger.Printf("Loaded %d environment variables from .env\n", len(envVars))
	return envVars
}

// GetPathTokens returns a list of available path tokens with their resolved values
func (a *App) GetPathTokens() []PathToken {
	envVars := a.GetEnvVariables()

	tokens := []PathToken{
		{
			Token:        "%REPO%",
			Description:  "Project root directory",
			ResolvedPath: envVars["REPO"],
		},
		{
			Token:        "%YAML%",
			Description:  "YAML file directory",
			ResolvedPath: envVars["YAML"],
		},
	}

	// Add custom environment variables (sorted for consistency)
	var customKeys []string
	for key := range envVars {
		if key != "REPO" && key != "YAML" {
			customKeys = append(customKeys, key)
		}
	}

	// Sort custom keys alphabetically
	for i := 0; i < len(customKeys); i++ {
		for j := i + 1; j < len(customKeys); j++ {
			if customKeys[i] > customKeys[j] {
				customKeys[i], customKeys[j] = customKeys[j], customKeys[i]
			}
		}
	}

	for _, key := range customKeys {
		tokens = append(tokens, PathToken{
			Token:        "%" + key + "%",
			Description:  "Custom environment variable",
			ResolvedPath: envVars[key],
		})
	}

	if debugMode {
		appLogger.Printf("[DEBUG] GetPathTokens returning %d tokens\n", len(tokens))
	}

	return tokens
}

// PathExists checks if a file or directory exists at the given path
func (a *App) PathExists(path string) bool {
	if path == "" {
		return false
	}

	// Substitute path tokens (%YAML%, %REPO%, etc.)
	envVars := a.GetEnvVariables()
	resolvedPath := path
	for key, value := range envVars {
		token := "%" + key + "%"
		resolvedPath = strings.ReplaceAll(resolvedPath, token, value)
	}

	// Convert forward slashes to OS-specific separators
	resolvedPath = filepath.FromSlash(resolvedPath)

	// Check if path exists
	_, err := os.Stat(resolvedPath)
	exists := err == nil

	// Always log path validation attempts for debugging
	appLogger.Printf("[PATH_CHECK] Input: '%s' -> Resolved: '%s' -> Exists: %v (Error: %v)\n", path, resolvedPath, exists, err)

	return exists
}

// WriteFile writes content to a file
func (a *App) WriteFile(filePath string, content string) error {
	if debugMode {
		appLogger.Printf("[DEBUG] WriteFile called: %s (%d bytes)\n", filePath, len(content))
	}

	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		appLogger.Printf("[ERROR] Failed to create directory '%s': %v\n", dir, err)
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Write file
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		appLogger.Printf("[ERROR] Failed to write file '%s': %v\n", filePath, err)
		return fmt.Errorf("failed to write file: %w", err)
	}

	appLogger.Printf("File written successfully: %s\n", filePath)
	return nil
}

// ReadFile reads content from a file
func (a *App) ReadFile(filePath string) (string, error) {
	if debugMode {
		appLogger.Printf("[DEBUG] ReadFile called: %s\n", filePath)
	}

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", filePath)
	}

	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		appLogger.Printf("[ERROR] Failed to read file '%s': %v\n", filePath, err)
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	if debugMode {
		appLogger.Printf("[DEBUG] File read successfully: %s (%d bytes)\n", filePath, len(data))
	}

	return string(data), nil
}

// RunSingleNode executes a single node by temporarily adding a 'stop' step after it in the main YAML
func (a *App) RunSingleNode(node *CLINode, yamlFilePath string) (string, error) {
	appLogger.Printf("Running single node: %s (ID: %s)\n", node.Name, node.ID)

	if yamlFilePath == "" {
		return "", fmt.Errorf("no YAML file loaded - save your pipeline first before running nodes")
	}

	// Load the current pipeline YAML
	yamlPipeline, err := a.readYAMLPipeline(yamlFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to load pipeline: %w", err)
	}

	// Find the index of this node in the pipeline
	nodeIndex := -1
	for i, step := range yamlPipeline.Run {
		if step.NodeID == node.ID {
			nodeIndex = i
			break
		}
	}

	if nodeIndex == -1 {
		return "", fmt.Errorf("node not found in pipeline - save your changes first")
	}

	// Create a backup of the original YAML
	originalSteps := yamlPipeline.Run

	// Insert a 'force' step right before the target node to ensure it runs
	// Then insert a 'stop' step right after it
	forceStep := YAMLStep{
		Name:    "__temp_force__",
		Type:    "force",
		Message: fmt.Sprintf("Force reprocessing: %s", node.Name),
	}

	stopStep := YAMLStep{
		Name:    "__temp_stop__",
		Type:    "stop",
		Message: fmt.Sprintf("Stopped after running: %s", node.Name),
	}

	// Build new steps list with force before and stop after the target node
	newSteps := make([]YAMLStep, 0, len(originalSteps)+2)
	newSteps = append(newSteps, originalSteps[:nodeIndex]...)
	newSteps = append(newSteps, forceStep)
	newSteps = append(newSteps, originalSteps[nodeIndex])
	newSteps = append(newSteps, stopStep)
	newSteps = append(newSteps, originalSteps[nodeIndex+1:]...)
	yamlPipeline.Run = newSteps

	// Write modified YAML temporarily
	if err := WriteYAMLPipeline(yamlPipeline, yamlFilePath); err != nil {
		return "", fmt.Errorf("failed to write modified YAML: %w", err)
	}

	appLogger.Printf("Inserted force+stop steps around node '%s' in: %s\n", node.Name, yamlFilePath)

	// Find run_pipeline.exe
	exePath, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("failed to get executable path: %w", err)
	}
	projectRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exePath))))
	runPipelinePath := filepath.Join(projectRoot, "run_pipeline.exe")

	// Check if run_pipeline.exe exists
	if _, err := os.Stat(runPipelinePath); os.IsNotExist(err) {
		// Restore original YAML before returning error
		yamlPipeline.Run = originalSteps
		WriteYAMLPipeline(yamlPipeline, yamlFilePath)
		return "", fmt.Errorf("run_pipeline.exe not found at: %s", runPipelinePath)
	}

	appLogger.Printf("Executing: %s %s\n", runPipelinePath, yamlFilePath)

	// Execute run_pipeline.exe with the modified YAML
	// The 'stop' step will halt execution after the target node
	cmd := exec.Command("cmd", "/c", "start", "cmd", "/k", runPipelinePath, yamlFilePath)
	cmd.Dir = projectRoot

	// Launch the terminal (non-blocking)
	runErr := cmd.Start()
	if runErr != nil {
		// Restore original YAML before returning error
		yamlPipeline.Run = originalSteps
		WriteYAMLPipeline(yamlPipeline, yamlFilePath)
		appLogger.Printf("[ERROR] Failed to launch terminal: %v\n", runErr)
		return "", fmt.Errorf("failed to launch terminal: %w", runErr)
	}

	appLogger.Printf("Launched execution in new terminal window.\n")
	outputStr := "Execution started in new terminal window...\n"
	outputStr += fmt.Sprintf("Running node: %s\n", node.Name)
	outputStr += "A 'force' step has been added before this node to ensure it runs.\n"
	outputStr += "A 'stop' step has been added after this node to halt execution.\n"
	outputStr += "Both steps will be automatically removed after execution starts.\n"

	// Wait a moment for the process to start
	time.Sleep(2 * time.Second)

	// Restore the original YAML (remove the force and stop steps)
	yamlPipeline.Run = originalSteps
	restoreErr := WriteYAMLPipeline(yamlPipeline, yamlFilePath)
	if restoreErr != nil {
		appLogger.Printf("[WARN] Failed to restore original YAML: %v\n", restoreErr)
		outputStr += "\nWarning: Failed to restore original YAML. You may need to manually remove the '__temp_force__' and '__temp_stop__' steps.\n"
	} else {
		appLogger.Printf("Restored original YAML (removed force and stop steps)\n")
	}

	// Resolve output glob patterns to show actual files created
	outputFilesInfo := resolveOutputGlobPatterns(node, projectRoot)
	if outputFilesInfo != "" {
		outputStr = outputStr + "\n\n" + outputFilesInfo
	}

	return outputStr, nil
}

// readYAMLPipeline reads and parses a YAML pipeline file
func (a *App) readYAMLPipeline(filePath string) (*YAMLPipeline, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read YAML file: %w", err)
	}

	var yamlPipeline YAMLPipeline
	if err := yaml.Unmarshal(data, &yamlPipeline); err != nil {
		return nil, fmt.Errorf("failed to parse YAML: %w", err)
	}

	return &yamlPipeline, nil
}

// resolveOutputGlobPatterns resolves glob patterns in output sockets to show actual files
func resolveOutputGlobPatterns(node *CLINode, baseDir string) string {
	if node.OutputSockets == nil || len(node.OutputSockets) == 0 {
		return ""
	}

	// Build a map of input socket values for placeholder substitution
	inputValues := make(map[string]string)
	if node.InputSockets != nil {
		for _, socket := range node.InputSockets {
			// Clean the flag name (remove -- prefix) and convert to placeholder format
			flagName := strings.TrimPrefix(socket.ArgumentFlag, "--")
			flagName = strings.ReplaceAll(flagName, "-", "_")
			inputValues[flagName] = socket.Value
		}
	}

	var result strings.Builder
	result.WriteString("=== Output Files Created ===\n")

	foundAnyFiles := false
	for _, socket := range node.OutputSockets {
		// Skip if no value
		if socket.Value == "" {
			continue
		}

		// Substitute placeholders like <output_folder> and <output_file_name_extension>
		pattern := socket.Value
		for key, value := range inputValues {
			placeholder := "<" + key + ">"
			pattern = strings.ReplaceAll(pattern, placeholder, value)
		}

		// Skip if pattern still contains unresolved placeholders or has no wildcards
		if strings.Contains(pattern, "<") || strings.Contains(pattern, ">") {
			appLogger.Printf("[DEBUG] Skipping pattern with unresolved placeholders: %s\n", pattern)
			continue
		}

		if !strings.Contains(pattern, "*") && !strings.Contains(pattern, "?") {
			appLogger.Printf("[DEBUG] Skipping pattern without wildcards: %s\n", pattern)
			continue
		}

		// Convert pattern to absolute path if it's relative
		if !filepath.IsAbs(pattern) {
			pattern = filepath.Join(baseDir, pattern)
		}

		// Resolve the glob pattern
		matches, err := filepath.Glob(pattern)
		if err != nil {
			appLogger.Printf("[WARN] Failed to resolve glob pattern '%s': %v\n", pattern, err)
			continue
		}

		if len(matches) > 0 {
			foundAnyFiles = true
			result.WriteString(fmt.Sprintf("\n%s:\n", socket.ArgumentFlag))
			result.WriteString(fmt.Sprintf("  Pattern: %s\n", pattern))

			// Show up to 10 files
			maxShow := 10
			if len(matches) > maxShow {
				for i := 0; i < maxShow; i++ {
					result.WriteString(fmt.Sprintf("  - %s\n", filepath.Base(matches[i])))
				}
				result.WriteString(fmt.Sprintf("  ... and %d more files\n", len(matches)-maxShow))
			} else {
				for _, match := range matches {
					result.WriteString(fmt.Sprintf("  - %s\n", filepath.Base(match)))
				}
			}
			result.WriteString(fmt.Sprintf("  Total: %d files\n", len(matches)))
		} else {
			// Pattern was valid but no files matched yet (maybe they haven't been created)
			appLogger.Printf("[DEBUG] No files matched pattern: %s\n", pattern)
		}
	}

	if !foundAnyFiles {
		return ""
	}

	return result.String()
}
