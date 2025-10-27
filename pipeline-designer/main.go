package main

import (
	"embed"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	// IMMEDIATE OUTPUT TEST - should appear in terminal
	fmt.Fprintln(os.Stderr, "========================================")
	fmt.Fprintln(os.Stderr, "Pipeline Designer Starting...")
	fmt.Fprintln(os.Stderr, "========================================")

	// Initialize main logger to stderr (more reliable for GUI apps)
	mainLogger := log.New(os.Stderr, "[MAIN] ", log.Ldate|log.Ltime|log.Lshortfile)

	// Define command line flags
	updateFlag := flag.String("update", "", "Update CLI definitions from source folder (e.g., 'standard_code')")
	helpFlag := flag.Bool("h", false, "Show help message")
	helpFlagLong := flag.Bool("help", false, "Show help message")
	flag.Parse()

	// Handle --help/-h flag
	if *helpFlag || *helpFlagLong {
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Pipeline Designer - Visual Pipeline Builder for BIPHUB")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "USAGE:")
		fmt.Fprintln(os.Stderr, "  pipeline-designer.exe [OPTIONS] [YAML_FILE]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "OPTIONS:")
		fmt.Fprintln(os.Stderr, "  -h, --help              Show this help message")
		fmt.Fprintln(os.Stderr, "  --update <folder>       Scan folder for Python scripts and generate CLI definitions")
		fmt.Fprintln(os.Stderr, "                          Example: --update standard_code")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "ARGUMENTS:")
		fmt.Fprintln(os.Stderr, "  YAML_FILE               Optional: Path to a YAML pipeline file to open on startup")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "ENVIRONMENT VARIABLES:")
		fmt.Fprintln(os.Stderr, "  DEBUG=1                 Enable debug logging (creates detailed logs)")
		fmt.Fprintln(os.Stderr, "  PIPELINE_DEBUG=1        Alternative debug mode variable")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "EXAMPLES:")
		fmt.Fprintln(os.Stderr, "  # Start GUI normally")
		fmt.Fprintln(os.Stderr, "  pipeline-designer.exe")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  # Open a specific pipeline file")
		fmt.Fprintln(os.Stderr, "  pipeline-designer.exe my_pipeline.yaml")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  # Generate CLI definitions from Python scripts")
		fmt.Fprintln(os.Stderr, "  pipeline-designer.exe --update standard_code")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  # Run with debug logging enabled")
		fmt.Fprintln(os.Stderr, "  $env:DEBUG=\"1\" ; pipeline-designer.exe")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "CLI DEFINITIONS LOCATION:")
		fmt.Fprintln(os.Stderr, "  Definitions are loaded from 'cli_definitions/' folder next to the executable.")
		fmt.Fprintln(os.Stderr, "  Generated definitions are saved to 'cli_definitions/Sandbox/' for review.")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "LOG FILE:")
		fmt.Fprintln(os.Stderr, "  Logs are written to: pipeline-designer.log (in the same directory as .exe)")
		fmt.Fprintln(os.Stderr, "")
		os.Exit(0)
	}

	// Handle --update flag before starting GUI
	if *updateFlag != "" {
		mainLogger.Printf("Update mode: scanning %s for new CLI definitions\n", *updateFlag)
		err := updateCLIDefinitions(*updateFlag, mainLogger)
		if err != nil {
			mainLogger.Printf("ERROR: Failed to update CLI definitions: %v\n", err)
			fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
			os.Exit(1)
		}
		mainLogger.Println("CLI definitions updated successfully!")
		fmt.Fprintln(os.Stderr, "✓ CLI definitions updated successfully!")
		os.Exit(0)
	}

	// Check for debug mode
	if os.Getenv("DEBUG") != "" || os.Getenv("PIPELINE_DEBUG") != "" {
		mainLogger.Println("DEBUG MODE ENABLED via environment variable")
		fmt.Fprintln(os.Stderr, "[DEBUG MODE ACTIVE]")
	}

	// Get command line arguments (after flag parsing)
	var filePath string
	if flag.NArg() > 0 {
		filePath = flag.Arg(0)
		mainLogger.Printf("Command line argument provided: %s\n", filePath)
	} else {
		mainLogger.Println("No command line arguments - will prompt for save location")
	}

	// Create an instance of the app structure
	app := NewApp()
	app.startupFilePath = filePath
	mainLogger.Printf("App initialized with startup file path: '%s'\n", filePath)

	// Create application with options
	mainLogger.Println("Starting Wails application...")
	err := wails.Run(&options.App{
		Title:  "pipeline-designer",
		Width:  1024,
		Height: 768,
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		BackgroundColour: &options.RGBA{R: 27, G: 38, B: 54, A: 1},
		OnStartup:        app.startup,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		mainLogger.Printf("[ERROR] Application error: %v\n", err)
		println("Error:", err.Error())
	} else {
		mainLogger.Println("Application exited normally")
	}
}

// updateCLIDefinitions scans the specified source folder and generates CLI definitions for new scripts
func updateCLIDefinitions(sourceFolder string, logger *log.Logger) error {
	logger.Printf("Scanning %s for Python scripts...\n", sourceFolder)

	// Get absolute paths
	repoRoot, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get working directory: %w", err)
	}

	// If we're in pipeline-designer subfolder, go up one level
	if filepath.Base(repoRoot) == "pipeline-designer" {
		repoRoot = filepath.Dir(repoRoot)
	}

	pythonDir := filepath.Join(repoRoot, sourceFolder, "python")
	cliDefsDir := filepath.Join(repoRoot, "pipeline-designer", "cli_definitions")
	sandboxDir := filepath.Join(cliDefsDir, "Sandbox")

	logger.Printf("Python scripts directory: %s\n", pythonDir)
	logger.Printf("CLI definitions directory: %s\n", cliDefsDir)
	logger.Printf("Sandbox directory: %s\n", sandboxDir)

	// Ensure sandbox directory exists
	if err := os.MkdirAll(sandboxDir, 0755); err != nil {
		return fmt.Errorf("failed to create sandbox directory: %w", err)
	}

	// Find the generate_cli_definition.py script
	generatorScript := filepath.Join(pythonDir, "generate_cli_definition.py")
	if _, err := os.Stat(generatorScript); err != nil {
		return fmt.Errorf("generator script not found: %s", generatorScript)
	}

	logger.Printf("Using generator script: %s\n", generatorScript)

	// Get all existing CLI definition IDs
	existingIDs := make(map[string]bool)
	err = filepath.WalkDir(cliDefsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".json" {
			// Extract ID from filename (without extension)
			id := filepath.Base(path)
			id = id[:len(id)-5] // Remove .json
			existingIDs[id] = true
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to scan existing definitions: %w", err)
	}

	logger.Printf("Found %d existing CLI definitions\n", len(existingIDs))

	// Find new scripts without definitions (recursively search subdirectories)
	type scriptInfo struct {
		fullPath string
		name     string
		id       string
	}
	var newScripts []scriptInfo

	err = filepath.WalkDir(pythonDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and non-Python files
		if d.IsDir() || filepath.Ext(d.Name()) != ".py" {
			return nil
		}

		// Skip private scripts and the generator itself
		if d.Name()[0] == '_' || d.Name() == "generate_cli_definition.py" {
			return nil
		}

		// Check if definition exists
		scriptID := d.Name()[:len(d.Name())-3] // Remove .py
		if !existingIDs[scriptID] {
			newScripts = append(newScripts, scriptInfo{
				fullPath: path,
				name:     d.Name(),
				id:       scriptID,
			})
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to scan python directory: %w", err)
	}

	if len(newScripts) == 0 {
		logger.Println("No new scripts found - all have CLI definitions!")
		fmt.Fprintln(os.Stderr, "✓ No new scripts found - all up to date!")
		return nil
	}

	logger.Printf("Found %d new scripts without CLI definitions:\n", len(newScripts))
	for i, script := range newScripts {
		logger.Printf("  %d. %s (in %s)\n", i+1, script.name, filepath.Dir(script.fullPath))
		fmt.Fprintf(os.Stderr, "  %d. %s\n", i+1, script.name)
	}

	// Generate CLI definitions for each new script
	pipelineConfigsDir := filepath.Join(repoRoot, "pipeline_configs")
	successCount := 0

	for _, script := range newScripts {
		scriptPath := script.fullPath
		scriptID := script.id

		logger.Printf("\nProcessing: %s\n", script.name)
		fmt.Fprintf(os.Stderr, "\nProcessing: %s\n", script.name)

		// Try to find matching YAML config
		yamlPath := ""
		yamlPattern := filepath.Join(pipelineConfigsDir, scriptID+"*.yaml")
		matches, _ := filepath.Glob(yamlPattern)
		if len(matches) > 0 {
			yamlPath = matches[0]
			logger.Printf("  Found matching YAML: %s\n", filepath.Base(yamlPath))
			fmt.Fprintf(os.Stderr, "  ✓ Found YAML: %s\n", filepath.Base(yamlPath))
		} else {
			logger.Printf("  No matching YAML found\n")
			fmt.Fprintf(os.Stderr, "  ⚠ No YAML config found\n")
		}

		// Build command to run generator
		args := []string{
			generatorScript,
			"--script", scriptPath,
			"--no-interactive",
			"--output-dir", sandboxDir,
		}

		if yamlPath != "" {
			args = append(args, "--from-yaml", yamlPath)
		}

		// Run the generator
		cmd := exec.Command("python", args...)
		cmd.Dir = repoRoot
		output, err := cmd.CombinedOutput()

		if err != nil {
			logger.Printf("  ERROR: %v\n", err)
			logger.Printf("  Output: %s\n", string(output))
			fmt.Fprintf(os.Stderr, "  ✗ Failed: %v\n", err)
			continue
		}

		logger.Printf("  ✓ Generated successfully\n")
		fmt.Fprintf(os.Stderr, "  ✓ Generated successfully\n")
		successCount++
	}

	logger.Printf("\n========================================\n")
	logger.Printf("Summary: Generated %d/%d CLI definitions\n", successCount, len(newScripts))
	logger.Printf("Location: %s\n", sandboxDir)
	logger.Printf("========================================\n")

	fmt.Fprintf(os.Stderr, "\n========================================\n")
	fmt.Fprintf(os.Stderr, "✓ Generated %d/%d new CLI definitions\n", successCount, len(newScripts))
	fmt.Fprintf(os.Stderr, "📂 Location: %s\n", sandboxDir)
	fmt.Fprintf(os.Stderr, "========================================\n")

	return nil
}
