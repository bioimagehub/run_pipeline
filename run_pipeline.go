package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath" // Added for handling file paths
	"reflect"
	"run_pipeline/go/find_anaconda_path"
	"strings"
	"time"

	"gopkg.in/yaml.v2" // YAML processing
)

// Requested Features
// Add a force reprocess tag

// Segment struct defines each segment of the pipeline with relevant attributes
type Segment struct {
	Name        string        `yaml:"name"`                  // The name of the segment
	Type        string        `yaml:"type,omitempty"`        // Optional: type of step (pause, stop, or normal)
	Message     string        `yaml:"message,omitempty"`     // Optional: message for pause/stop
	Environment string        `yaml:"environment,omitempty"` // The Python environment to use
	Commands    []interface{} `yaml:"commands,omitempty"`    // Commands to execute (can be strings or maps)
}

// Config struct to hold the overall configuration structure
type Config struct {
	Run []Segment `yaml:"run"` // Slice of segments representing the commands to be processed
}

// SegmentStatus holds execution status for a single segment
type SegmentStatus struct {
	Name          string `yaml:"name"`                     // The name of the segment
	LastProcessed string `yaml:"last_processed,omitempty"` // Timestamp of when this segment was last processed
	CodeVersion   string `yaml:"code_version,omitempty"`   // Git version/tag/commit hash used when this segment was processed
	RunDuration   string `yaml:"run_duration,omitempty"`   // Wall-clock duration of this segment (e.g., "1m23.456s")
}

// Status struct to hold execution status for all segments
type Status struct {
	Segments []SegmentStatus `yaml:"segments"` // Status for each segment
}

// Versioning: overridden via -ldflags, with embedded fallback and git as last resort
// Example: go build -ldflags "-X main.Version=v0.2.0 -X main.Commit=abc1234 -X main.BuildDate=2025-09-11"
var (
	Version   string
	Commit    string
	BuildDate string
)

// one-time warning flag for deprecated './' usage
var warnedDotSlash bool

// GetBaseDir returns the project root directory.
// It handles both `go run` (using working dir) and `go build` (using executable path).
func GetBaseDir() string {
	exePath, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	exeDir := filepath.Dir(exePath)

	// Check if the path includes a Go build cache temp dir (used by `go run`)
	if strings.Contains(exeDir, "go-build") || strings.Contains(exeDir, os.TempDir()) {
		// Likely running with `go run`, use the current working directory
		wd, err := os.Getwd()
		if err != nil {
			log.Fatal(err)
		}
		return wd
	}

	// Otherwise, likely a real executable (from `go build`)
	return exeDir
}

func resolvePath(v string, mainProgramDir, yamlDir string) string {
	// New explicit tokens (non-breaking):
	//   %REPO%/path -> resolved relative to mainProgramDir (repo/program root)
	//   %YAML%/path -> resolved relative to folder containing the YAML file
	//   %VAR%/path  -> resolved using variable VAR from .env file
	// These take precedence when present at the start of the string.
	if strings.HasPrefix(v, "%REPO%") {
		sub := strings.TrimPrefix(v, "%REPO%")
		sub = strings.TrimLeft(sub, "/\\")
		return filepath.Join(mainProgramDir, filepath.FromSlash(sub))
	}
	if strings.HasPrefix(v, "%YAML%") {
		sub := strings.TrimPrefix(v, "%YAML%")
		sub = strings.TrimLeft(sub, "/\\")
		return filepath.Join(yamlDir, filepath.FromSlash(sub))
	}

	// Check for custom environment variables from .env file
	if strings.HasPrefix(v, "%") && strings.Contains(v[1:], "%") {
		// Extract variable name between % signs
		endIdx := strings.Index(v[1:], "%")
		if endIdx != -1 {
			varName := v[1 : endIdx+1]
			envValue := getEnvValue(varName)
			if envValue != "" {
				sub := v[endIdx+2:] // Everything after the second %
				sub = strings.TrimLeft(sub, "/\\")
				if sub == "" {
					return envValue // Just return the env value if no path follows
				}
				return filepath.Join(envValue, filepath.FromSlash(sub))
			}
		}
	}

	// Backward-compatible behavior for leading ./
	if strings.HasPrefix(v, "./") {
		if !warnedDotSlash {
			fmt.Println("[deprecated] Use %REPO%/ or %YAML%/ instead of './'")
			warnedDotSlash = true
		}
		vTrim := strings.TrimPrefix(v, "./")
		if strings.HasSuffix(vTrim, ".py") || strings.HasSuffix(vTrim, ".ijm") || strings.HasSuffix(vTrim, ".exe") {
			return filepath.Join(mainProgramDir, vTrim)
		}
		return filepath.Join(yamlDir, vTrim)
	}

	// No changes for other paths (absolute or bare)
	return v
}

// askForAnacondaPath prompts the user for the Anaconda installation path and validates it.
func askForAnacondaPath() string {
	reader := bufio.NewReader(os.Stdin)
	var anacondaPath string
	valid := false

	for !valid {
		fmt.Print("Please provide the path to your Anaconda installation: ")
		inputPath, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Error reading user input: %v", err)
		}
		anacondaPath = strings.TrimSpace(inputPath)

		// Validate the installation directory
		if isValidAnacondaPath(anacondaPath) {
			// Run a simple test command
			if testPythonExecution(anacondaPath) {
				fmt.Printf("Valid Anaconda installation found at: %v\n", anacondaPath)
				saveToEnvFile("CONDA_PATH", anacondaPath)
				valid = true
			} else {
				fmt.Println("Python execution failed. Please check the path or installation.")
			}
		} else {
			fmt.Println("Invalid path. Please ensure it contains the 'envs' directory and 'python.exe'.")
		}
	}

	return anacondaPath
}

// findImageJPath searches for the ImageJ path in the .env file.
// It returns the path if found, or an error if the file does not exist or the key is not found.
func findImageJPath() (string, error) {
	envFilePath := filepath.Join(GetBaseDir(), ".env")
	file, err := os.ReadFile(envFilePath)
	if err != nil {
		return "", fmt.Errorf("error reading .env file: %v", err)
	}

	// Split the file content into lines
	lines := strings.Split(string(file), "\n")

	// Look for the IMAGEJ_PATH entry
	for _, line := range lines {
		if strings.HasPrefix(line, "IMAGEJ_PATH=") {
			// Return the path, stripping the key and any surrounding whitespace
			return strings.TrimSpace(strings.TrimPrefix(line, "IMAGEJ_PATH=")), nil
		}
	}

	return "", fmt.Errorf("IMAGEJ_PATH not found in .env file")
}

// isValidImageJExecutable checks if the specified path points to a valid ImageJ executable.
func isValidImageJExecutable(path string) bool {
	return isFile(path) // Check if the provided path is a valid file
}

// askForImageJPath prompts the user for the path to the ImageJ executable and validates it.
func askForImageJPath() string {
	reader := bufio.NewReader(os.Stdin)
	var imageJPath string
	valid := false

	for !valid {
		fmt.Print("Please provide the path to the ImageJ executable (ImageJ.exe): ")
		inputPath, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Error reading user input: %v", err)
		}
		imageJPath = strings.TrimSpace(inputPath)

		// Validate the path to the ImageJ executable
		if isValidImageJExecutable(imageJPath) {
			fmt.Printf("Valid ImageJ executable found at: %v\n", imageJPath)
			valid = true
			// Save valid ImageJ path to the .env file
			saveToEnvFile("IMAGEJ_PATH", imageJPath)
		} else {
			fmt.Println("Invalid path. Please ensure it points to 'ImageJ.exe'.")
		}
	}

	return imageJPath
}

// isValidAnacondaPath checks if the specified path is a valid Anaconda installation.
func isValidAnacondaPath(path string) bool {
	envsPath := filepath.Join(path, "envs")
	pythonPath := filepath.Join(path, "python.exe")
	return isDirectory(envsPath) && isFile(pythonPath)
}

// isDirectory checks if the specified path is a directory.
func isDirectory(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// isFile checks if the specified path is a file.
func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// testPythonExecution runs a simple Python command to check if Python is functional.
func testPythonExecution(anacondaPath string) bool {
	cmd := exec.Command(filepath.Join(anacondaPath, "python.exe"), "-c", "print('Hello from Python')")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Error executing Python: %v\n", err)
		return false
	}

	fmt.Printf("Python output: %s\n", output)
	return true
}

// saveToEnvFile saves the specified key-value pair to the .env file.
func saveToEnvFile(key, value string) {
	envFilePath := filepath.Join(GetBaseDir(), ".env")
	file, err := os.OpenFile(envFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Error opening .env file: %v", err)
	}
	defer file.Close()

	// Write the key-value pair to the .env file
	if _, err := file.WriteString(fmt.Sprintf("%s=%s\n", key, value)); err != nil {
		log.Fatalf("Error writing to .env file: %v", err)
	}
	fmt.Printf("%s path saved to .env file.\n", key)
}

// getEnvValue retrieves a value for the specified key from the .env file.
// Returns empty string if the key is not found or if there's an error reading the file.
func getEnvValue(key string) string {
	envFilePath := filepath.Join(GetBaseDir(), ".env")
	file, err := os.ReadFile(envFilePath)
	if err != nil {
		return "" // Return empty string if .env file doesn't exist or can't be read
	}

	// Split the file content into lines
	lines := strings.Split(string(file), "\n")

	// Look for the specified key
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, key+"=") {
			// Return the value, stripping the key and any surrounding whitespace
			return strings.TrimSpace(strings.TrimPrefix(line, key+"="))
		}
	}

	return "" // Key not found
}

// getGitVersion returns a descriptive git version string for the current repository.
// Tries `git describe --tags --dirty --always --long`, then falls back to short and full commit.
func getGitVersion() string {
	candidates := [][]string{
		{"git", "describe", "--tags", "--dirty", "--always", "--long"},
		{"git", "rev-parse", "--short", "HEAD"},
		{"git", "rev-parse", "HEAD"},
	}

	baseDir := GetBaseDir()
	for _, args := range candidates {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = baseDir
		out, err := cmd.CombinedOutput()
		if err == nil {
			v := strings.TrimSpace(string(out))
			if v != "" {
				return v
			}
		}
	}
	return "unknown"
}

// getVersion returns a stable version string in priority:
// 1) ldflags-injected Version (and optionally Commit) if set
// 2) embedded VERSION file (at build time)
// 3) git describe/commit if available
// 4) "unknown"
func getVersion() string {
	v := strings.TrimSpace(Version)
	c := strings.TrimSpace(Commit)
	if v != "" {
		if c != "" {
			return fmt.Sprintf("%s (%s)", v, c)
		}
		return v
	}
	// Try reading VERSION file next to the executable
	versionPath := filepath.Join(GetBaseDir(), "VERSION")
	if b, err := os.ReadFile(versionPath); err == nil {
		fileV := strings.TrimSpace(string(b))
		if fileV != "" {
			return fileV
		}
	}
	return getGitVersion()
}

// loadOrCreateStatus loads existing status file or creates a new one
func loadOrCreateStatus(statusPath string, segmentNames []string) Status {
	var status Status

	// Try to load existing status file
	if data, err := os.ReadFile(statusPath); err == nil {
		if err := yaml.Unmarshal(data, &status); err == nil {
			// Status file loaded successfully
			// Ensure all segments from config are present in status
			statusMap := make(map[string]*SegmentStatus)
			for i := range status.Segments {
				statusMap[status.Segments[i].Name] = &status.Segments[i]
			}

			// Add missing segments
			for _, name := range segmentNames {
				if _, exists := statusMap[name]; !exists {
					status.Segments = append(status.Segments, SegmentStatus{Name: name})
				}
			}
			return status
		}
	}

	// Create new status with all segment names
	status.Segments = make([]SegmentStatus, len(segmentNames))
	for i, name := range segmentNames {
		status.Segments[i] = SegmentStatus{Name: name}
	}
	return status
}

// saveStatus writes the status to the status YAML file
func saveStatus(statusPath string, status Status) error {
	data, err := yaml.Marshal(&status)
	if err != nil {
		return fmt.Errorf("error marshalling status YAML: %v", err)
	}

	err = os.WriteFile(statusPath, data, 0644)
	if err != nil {
		return fmt.Errorf("error writing status YAML file: %v", err)
	}
	return nil
}

// getSegmentStatus finds the status entry for a given segment name
func getSegmentStatus(status *Status, segmentName string) *SegmentStatus {
	for i := range status.Segments {
		if status.Segments[i].Name == segmentName {
			return &status.Segments[i]
		}
	}
	return nil
}

// Function to prepare command arguments for Python execution
func makePythonCommand(segment Segment, anacondaPath, mainProgramDir, yamlDir string) []string {
	cmdArgs := []string{"cmd", "/C"} // Windows command line execution prefix

	// Determine which environment to activate for Python
	if strings.ToLower(segment.Environment) == "base" {
		// If the environment is base, activate it directly
		cmdArgs = append(cmdArgs,
			anacondaPath+"\\Scripts\\activate.bat", // Script to activate Anaconda
			anacondaPath,
			"&&", // Use '&&' to chain commands together
		)
	} else {
		// For named environments, activate that environment
		cmdArgs = append(cmdArgs,
			anacondaPath+"\\Scripts\\activate.bat", // Script to activate Anaconda
			anacondaPath,
			"&&",
			"conda", "activate", segment.Environment, // Activate the specified conda environment
			"&&",
		)
	}

	// Loop through each command in the segment's command list
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			resolved := resolvePath(v, mainProgramDir, yamlDir)
			cmdArgs = append(cmdArgs, resolved)

		case map[interface{}]interface{}:
			for flag, value := range v {
				cmdArgs = append(cmdArgs, fmt.Sprintf("%v", flag))

				if value != nil && value != "null" {
					valStr := fmt.Sprintf("%v", value)
					resolved := resolvePath(valStr, mainProgramDir, yamlDir)
					cmdArgs = append(cmdArgs, resolved)
				}
			}

		default:
			log.Fatalf("unexpected type %v", reflect.TypeOf(v))
		}
	}

	return cmdArgs
}

// Function to prepare command arguments for uv execution
// Expects segment.Environment to be of the form "uv:<group>"
// makeUvCommand builds the uv command args and returns the desired Python version (if any)
// Supported environment syntax:
//
//	"uv:<group>"              -> uses default/fallback Python (3.11), includes dependency group
//	"uv@3.11:<group>"         -> forces Python 3.11
//	default Python may be overridden with env var UV_DEFAULT_PYTHON (e.g., 3.11 or 3.10)
func makeUvCommand(segment Segment, mainProgramDir, yamlDir string) ([]string, string) {
	// Determine the correct uv runner prefix, installing or falling back to pipx if needed
	cmdArgs := getUvRunnerPrefix(mainProgramDir)

	// Always run in isolated env to avoid picking up project .venv or system env
	cmdArgs = append(cmdArgs, "--isolated")

	// Extract optional Python version and group from environment string
	envLower := strings.ToLower(segment.Environment)
	uvPython := strings.TrimSpace(os.Getenv("UV_DEFAULT_PYTHON"))
	if uvPython == "" {
		uvPython = "3.11" // default for broad wheel availability
	}

	var group string
	if strings.HasPrefix(envLower, "uv@") {
		// format uv@<pyver>:<group>
		rest := strings.TrimPrefix(envLower, "uv@")
		parts := strings.SplitN(rest, ":", 2)
		if len(parts) == 2 {
			if parts[0] != "" {
				uvPython = parts[0]
			}
			group = parts[1]
		} else {
			// No group specified, just a version
			if parts[0] != "" {
				uvPython = parts[0]
			}
		}
	} else {
		group = strings.TrimPrefix(envLower, "uv:")
	}

	if group != "" {
		cmdArgs = append(cmdArgs, "--group", group)
	}

	// Loop through commands similar to Python/ImageJ path handling
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			resolved := resolvePath(v, mainProgramDir, yamlDir)
			cmdArgs = append(cmdArgs, resolved)
		case map[interface{}]interface{}:
			for flag, value := range v {
				cmdArgs = append(cmdArgs, fmt.Sprintf("%v", flag))
				if value != nil && value != "null" {
					valStr := fmt.Sprintf("%v", value)
					resolved := resolvePath(valStr, mainProgramDir, yamlDir)
					cmdArgs = append(cmdArgs, resolved)
				}
			}
		default:
			log.Fatalf("unexpected type %v", reflect.TypeOf(v))
		}
	}

	return cmdArgs, uvPython
}

// getUvRunnerPrefix ensures `uv` is available and returns the appropriate command prefix:
// - ["cmd","/C","uv","run"] when uv is on PATH
// - ["cmd","/C","pipx","run","uv","run"] as a fallback when uv is not on PATH but pipx is available
// It will attempt `pipx install uv` when uv is missing.
func getUvRunnerPrefix(mainProgramDir string) []string {
	have := func(name string) bool { _, err := exec.LookPath(name); return err == nil }

	// Prefer a repo-bundled uv.exe if present (no system install needed)
	externalUV := filepath.Join(mainProgramDir, "external", "UV")
	localCandidates := []string{
		filepath.Join(externalUV, "uv.exe"),
		filepath.Join(externalUV, "bin", "uv.exe"),
		filepath.Join(mainProgramDir, "uv.exe"),
		filepath.Join(mainProgramDir, "uv", "uv.exe"),
		filepath.Join(mainProgramDir, "tools", "uv.exe"),
		filepath.Join(mainProgramDir, "tools", "uv", "uv.exe"),
		filepath.Join(mainProgramDir, "bin", "uv.exe"),
	}
	for _, p := range localCandidates {
		if isFile(p) {
			return []string{"cmd", "/C", p, "run"}
		}
	}

	// Attempt an in-repo standalone install into external/UV using the official installer
	// Only try if uv isn't on PATH
	if !have("uv") {
		// Ensure destination dir exists
		_ = os.MkdirAll(externalUV, 0755)
		// PowerShell command to set install dir and run installer
		psCmd := fmt.Sprintf("$env:UV_INSTALL_DIR = '%s'; irm https://astral.sh/uv/0.8.18/install.ps1 | iex", externalUV)
		fmt.Printf("Attempting standalone uv install into %s...\n", externalUV)
		install := exec.Command("powershell", "-ExecutionPolicy", "ByPass", "-NoProfile", "-Command", psCmd)
		install.Stdout = os.Stdout
		install.Stderr = os.Stderr
		_ = install.Run()

		// Re-check local candidates after installation
		for _, p := range localCandidates {
			if isFile(p) {
				fmt.Printf("Found uv at %s after install.\n", p)
				return []string{"cmd", "/C", p, "run"}
			}
		}
	}

	if have("uv") {
		return []string{"cmd", "/C", "uv", "run"}
	}

	// Try pipx-based options
	if have("pipx") {
		// Quick ephemeral check: pipx run uv --version
		testCmd := exec.Command("cmd", "/C", "pipx", "run", "uv", "--version")
		if err := testCmd.Run(); err == nil {
			fmt.Println("uv not found; using 'pipx run uv' for this session.")
			return []string{"cmd", "/C", "pipx", "run", "uv", "run"}
		}

		// Try installing uv for persistent availability
		fmt.Println("uv not found; attempting 'pipx install uv' ...")
		install := exec.Command("cmd", "/C", "pipx", "install", "uv")
		install.Stdout = os.Stdout
		install.Stderr = os.Stderr
		_ = install.Run()

		// Re-evaluate availability
		if have("uv") {
			return []string{"cmd", "/C", "uv", "run"}
		}

		// Fallback to pipx run if it now works
		testCmd2 := exec.Command("cmd", "/C", "pipx", "run", "uv", "--version")
		if err := testCmd2.Run(); err == nil {
			fmt.Println("Using 'pipx run uv' as fallback. Consider adding uv to PATH via 'pipx install uv'.")
			return []string{"cmd", "/C", "pipx", "run", "uv", "run"}
		}
	}

	// If we get here, we couldn't find or install uv automatically
	fmt.Println("ERROR: 'uv' is not available and automatic installation failed.")
	fmt.Println("Please install uv manually, e.g. 'pipx install uv' or download the standalone binary.")
	log.Fatal("uv is required for 'uv:' environments")
	return []string{"cmd", "/C", "uv", "run"}
}

// Function to prepare command arguments for ImageJ execution
func makeImageJCommand(segment Segment, imageJPath, mainProgramDir, yamlDir string) []string {
	cmdArgs := []string{imageJPath, "--ij2", "--headless", "--console"} // Initialize command arguments

	// Loop through each command in the segment's command list
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			// Resolve paths for string commands and add to cmdArgs with proper encapsulation
			resolved := resolvePath(v, mainProgramDir, yamlDir)
			cmdArgs = append(cmdArgs, "--run", fmt.Sprintf("\"%s\"", resolved))

		case map[interface{}]interface{}:
			// For map commands, loop through entries
			for flag, value := range v {
				// Append the flag in the required format
				if value != nil && value != "null" {
					valStr := fmt.Sprintf("%v", value) // Convert value to string
					// Append it in the required format as "key='value'"
					cmdArgs = append(cmdArgs, fmt.Sprintf("\"%s='%s'\"", flag, valStr))
				}
			}

		default:
			log.Fatalf("unexpected type %v", reflect.TypeOf(v)) // Handle unexpected command types
		}
	}

	return cmdArgs
}

func main() {
	mainProgramDir := GetBaseDir()
	fmt.Printf("mainProgramDir: %v\n", mainProgramDir)

	// set working directory to the main program directory
	err := os.Chdir(mainProgramDir)
	if err != nil {
		log.Fatalf("Error changing directory: %v", err)
	}

	// Initialize a variable to hold the path to the YAML configuration file
	var yamlPath string
	forceReprocessing := false // Initialize the force reprocessing flag

	// Check if a path is passed as a command-line argument
	for _, arg := range os.Args[1:] {
		if arg == "-h" || arg == "--help" {
			fmt.Println("Usage:")
			fmt.Println("  your_program [options] <path_to_yaml>")
			fmt.Println("")
			fmt.Println("Options:")
			fmt.Println("  -f, --force_reprocessing  Process segments even if they have been previously processed.")
			fmt.Println("  -h, --help                Show help information.")
			fmt.Println("")
			fmt.Println("Arguments:")
			fmt.Println("  <path_to_yaml>       The path to the YAML configuration file.")
			fmt.Println("")
			fmt.Println("Path resolution tokens (can be used in YAML paths):")
			fmt.Println("  %REPO%/path          Resolves relative to program root directory")
			fmt.Println("  %YAML%/path          Resolves relative to YAML file directory")
			fmt.Println("  %VARNAME%/path       Resolves using VARNAME from .env file")
			fmt.Println("  ./path               Deprecated, use %REPO%/ or %YAML%/ instead")
			fmt.Println("")
			fmt.Println("YAML step types (for easy copy-paste):")
			fmt.Println("  # - type: normal (default) - Runs commands as usual.")
			fmt.Println("  # - type: pause  - Pauses the pipeline and waits for user to press Enter. Optional: add a 'message' field.")
			fmt.Println("  # - type: stop   - Stops the pipeline immediately. Optional: add a 'message' field.")
			fmt.Println("  # - type: force  - Enables force reprocessing for all subsequent segments (same as --force_reprocessing). Optional: add a 'message' field.")
			fmt.Println("# Example: Copy this block directly to your YAML file:")
			fmt.Println("run:")
			fmt.Println("- name: Collapse folder structure and save as .tif\n  environment: convert_to_tif\n  commands:\n    - python\n    - ./standard_code/python/convert_to_tif.py\n    - --input-file-or-folder: ./input\n    - --extension: .ims\n    - --projection-method: max\n    - --search-subfolders\n    - --collapse-delimiter: __\n- name: Enable force mode mid-pipeline\n  type: force\n  message: 'Reprocessing all subsequent steps.'\n- name: Pause for inspection\n  type: pause\n  message: 'Paused for user inspection.'\n- name: Stop pipeline\n  type: stop\n  message: 'Pipeline stopped intentionally.'")
			os.Exit(0)
		} else if arg == "--force_reprocessing" || arg == "-f" {
			forceReprocessing = true
		} else {
			yamlPath = arg // Assume the next argument is the YAML file path

			// Check if the YAML file path is valid
			if !isFile(yamlPath) {
				log.Fatalf("The specified YAML path is not a valid file: %v", yamlPath)
			}
		}
	}

	// TODO Check if the yaml file is inside the main program directory
	// Suggest to rather copy it to the main folder of the program
	// Then they can use relative paths to input folder and output folder

	// Read the YAML file contents
	data, err := os.ReadFile(yamlPath)
	if err != nil {
		log.Fatalf("error reading YAML file: %v", err)
	}

	// Unmarshal the YAML data into the configuration struct
	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		log.Fatalf("error unmarshalling YAML: %v", err)
	}

	// Get the directory of the YAML file for resolving data paths
	yamlDir := filepath.Dir(yamlPath)

	// Create status file path: inputname_status.yaml
	yamlBaseName := filepath.Base(yamlPath)
	yamlNameWithoutExt := strings.TrimSuffix(yamlBaseName, filepath.Ext(yamlBaseName))
	statusPath := filepath.Join(yamlDir, yamlNameWithoutExt+"_status.yaml")

	// Extract segment names for status initialization
	segmentNames := make([]string, len(config.Run))
	for i, segment := range config.Run {
		segmentNames[i] = segment.Name
	}

	// Load or create status file
	status := loadOrCreateStatus(statusPath, segmentNames)
	fmt.Printf("Using status file: %v\n", statusPath)

	// Iterate over each segment defined in the configuration
	for _, segment := range config.Run {
		// Handle pause and stop types
		stepType := strings.ToLower(segment.Type)
		switch stepType {
		case "pause":
			msg := segment.Message
			if msg == "" {
				msg = "Paused. Press Enter to continue."
			}
			fmt.Printf("[PAUSE] %s\n", msg)
			reader := bufio.NewReader(os.Stdin)
			_, _ = reader.ReadString('\n')
			continue
		case "stop":
			msg := segment.Message
			if msg == "" {
				msg = "Stopped by stop step. Exiting."
			}
			fmt.Printf("[STOP] %s\n", msg)
			os.Exit(0)
		case "force":
			msg := segment.Message
			if msg == "" {
				msg = "Force mode activated. All subsequent segments will be reprocessed."
			}
			fmt.Printf("[FORCE] %s\n", msg)
			forceReprocessing = true
			continue
		}

		// Get status for this segment
		segmentStatus := getSegmentStatus(&status, segment.Name)
		if segmentStatus == nil {
			// Segment not in status, add it
			status.Segments = append(status.Segments, SegmentStatus{Name: segment.Name})
			segmentStatus = &status.Segments[len(status.Segments)-1]
		}

		// If not in forceReprocessing mode, but we hit a segment that is not processed, activate forceReprocessing for all subsequent segments
		if !forceReprocessing && segmentStatus.LastProcessed == "" {
			fmt.Printf("Segment %s has not been processed. \n---Activating force reprocessing for all subsequent segments.---\n\n", segment.Name)
			forceReprocessing = true
		}

		// Check if this segment has already been processed
		if !forceReprocessing && segmentStatus.LastProcessed != "" {
			fmt.Printf("Skipping segment %s, already processed on %s\n", segment.Name, segmentStatus.LastProcessed)
			fmt.Println()
			continue // Skip to the next segment if already processed and not forcing reprocessing
		}

		fmt.Printf("Processing segment: %s\n", segment.Name)

		// Prepare command arguments for executing the environment and subsequent commands
		var cmdArgs []string // Declare cmdArgs here

		// Determine if the environment is going to be imageJ, uv, or conda-based Python
		fmt.Println(segment.Environment)
		if strings.ToLower(segment.Environment) == "imagej" {
			fmt.Println("running imageJ")
			imageJPath, err := findImageJPath()
			if err != nil {
				imageJPath = askForImageJPath()
			} else {
				fmt.Printf("Found ImageJ.exe: %v\n", imageJPath)
			}

			cmdArgs = makeImageJCommand(segment, imageJPath, mainProgramDir, yamlDir)
		} else if strings.HasPrefix(strings.ToLower(segment.Environment), "uv:") || strings.HasPrefix(strings.ToLower(segment.Environment), "uv@") {
			// uv-managed environment: no conda activation
			var uvPython string
			cmdArgs, uvPython = makeUvCommand(segment, mainProgramDir, yamlDir)
			// Propagate desired Python version to uv via environment
			// Use UV_PYTHON which uv honors for selecting interpreter
			if uvPython != "3.11" {
				fmt.Printf("[note] Requesting UV_PYTHON=%s (override via UV_DEFAULT_PYTHON). 3.11 is recommended for best wheel coverage on Windows.\n", uvPython)
			}
			cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
			cmd.Env = append(os.Environ(), fmt.Sprintf("UV_PYTHON=%s", uvPython))
			fmt.Printf("Constructed command: %v with UV_PYTHON=%s\n", cmdArgs, uvPython)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			startTime := time.Now()
			err = cmd.Run()
			if err != nil {
				fmt.Printf("Error executing command: %v\n", err)
				log.Fatalf("Error")
			}
			// Update status in status file
			segmentStatus.LastProcessed = time.Now().Format("2006-01-02")
			segmentStatus.CodeVersion = getVersion()
			segmentStatus.RunDuration = time.Since(startTime).String()
			if err := saveStatus(statusPath, status); err != nil {
				log.Fatalf("%v", err)
			}
			fmt.Println("")
			fmt.Println("")
			continue
		} else {
			anacondaPath, err := find_anaconda_path.FindAnacondaPath()
			if err != nil {
				anacondaPath = askForAnacondaPath()
			} else {
				fmt.Printf("Found Anaconda base: %v\n", anacondaPath)
			}

			cmdArgs = makePythonCommand(segment, anacondaPath, mainProgramDir, yamlDir)

		}
		// Print the constructed command arguments for debugging
		fmt.Printf("Constructed command: %v\n", cmdArgs)

		// Create the command using the constructed arguments
		cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
		cmd.Stdout = os.Stdout // Redirect standard output to console
		cmd.Stderr = os.Stderr // Redirect standard error to console

		// Execute the command
		startTime := time.Now()
		err = cmd.Run()
		if err != nil {
			fmt.Printf("Error executing command: %v\n", err)
			log.Fatalf("Error") // Log fatal error on execution failure
		}

		// Update status in status file
		segmentStatus.LastProcessed = time.Now().Format("2006-01-02")
		segmentStatus.CodeVersion = getVersion()
		segmentStatus.RunDuration = time.Since(startTime).String()
		if err := saveStatus(statusPath, status); err != nil {
			log.Fatalf("%v", err)
		}
		fmt.Println("") // Add some space between the segment prints
		fmt.Println("") // Add some space between the segment prints
	}

	// Prompt the user that processing is complete and wait for input, but auto-close after 10 seconds
	fmt.Print("Processing complete. Press Enter to exit (auto-closes in 10 seconds)...")
	inputCh := make(chan struct{})
	go func() {
		reader := bufio.NewReader(os.Stdin)
		_, _ = reader.ReadString('\n')
		inputCh <- struct{}{}
	}()
	select {
	case <-inputCh:
		// User pressed Enter
	case <-time.After(10 * time.Second):
		fmt.Println("\nNo input detected. Exiting automatically.")
	}
}
