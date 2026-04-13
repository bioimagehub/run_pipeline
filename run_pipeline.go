package main

import (
	"bufio"
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath" // Added for handling file paths
	"reflect"
	"run_pipeline/go/find_anaconda_path"
	"runtime"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v2" // YAML processing
)

// Requested Features
// Add a force reprocess tag

// Segment struct defines each segment of the pipeline with relevant attributes
type Segment struct {
	Name           string                 `yaml:"name"`                       // The name of the segment
	Type           string                 `yaml:"type,omitempty"`             // Optional: type of step (pause, stop, or normal)
	Message        string                 `yaml:"message,omitempty"`          // Optional: message for pause/stop
	Environment    string                 `yaml:"environment,omitempty"`      // The Python environment to use
	Env            map[string]interface{} `yaml:"env,omitempty"`              // Optional: environment variables for the segment
	UseLinuxDistro string                 `yaml:"use-linux-distro,omitempty"` // Optional: route Windows execution through WSL
	Commands       []interface{}          `yaml:"commands,omitempty"`         // Commands to execute (can be strings or maps)
}

// Config struct to hold the overall configuration structure
type Config struct {
	Run []Segment `yaml:"run"` // Slice of segments representing the commands to be processed
}

// SegmentStatus holds execution status for a single segment
type SegmentStatus struct {
	Name          string `yaml:"name"`                     // The name of the segment (for human readability)
	ContentHash   string `yaml:"content_hash"`             // Hash of segment content (primary matching key)
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

func isWindowsRuntime() bool {
	return runtime.GOOS == "windows"
}

func isLinuxRuntime() bool {
	return runtime.GOOS == "linux"
}

func venvPythonPath(venvPath string) string {
	if isWindowsRuntime() {
		return filepath.Join(venvPath, "Scripts", "python.exe")
	}
	return filepath.Join(venvPath, "bin", "python")
}

func condaBasePythonPath(anacondaPath string) string {
	if isWindowsRuntime() {
		return filepath.Join(anacondaPath, "python.exe")
	}
	return filepath.Join(anacondaPath, "bin", "python")
}

func condaExecutable(anacondaPath string) string {
	if isWindowsRuntime() {
		candidates := []string{
			filepath.Join(anacondaPath, "condabin", "conda.bat"),
			filepath.Join(anacondaPath, "Scripts", "conda.exe"),
		}
		for _, p := range candidates {
			if isFile(p) {
				return p
			}
		}
	} else {
		candidates := []string{
			filepath.Join(anacondaPath, "bin", "conda"),
			filepath.Join(anacondaPath, "condabin", "conda"),
		}
		for _, p := range candidates {
			if isFile(p) {
				return p
			}
		}
	}

	if p, err := exec.LookPath("conda"); err == nil {
		return p
	}

	return ""
}

func shellCommandPrefix() []string {
	if isWindowsRuntime() {
		return []string{"cmd", "/C"}
	}
	return []string{"bash", "-lc"}
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

func isWindowsDrivePath(value string) bool {
	if len(value) < 3 {
		return false
	}
	drive := value[0]
	if !((drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z')) {
		return false
	}
	return value[1] == ':' && (value[2] == '\\' || value[2] == '/')
}

func toWslPath(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return trimmed
	}
	if strings.HasPrefix(trimmed, `\\`) {
		log.Fatalf("UNC paths are not supported for WSL-routed segments yet: %s", value)
	}
	if !isWindowsDrivePath(trimmed) {
		return strings.ReplaceAll(trimmed, `\`, `/`)
	}

	drive := strings.ToLower(trimmed[:1])
	remainder := strings.ReplaceAll(trimmed[2:], `\`, `/`)
	remainder = strings.TrimPrefix(remainder, "/")
	if remainder == "" {
		return "/mnt/" + drive
	}
	return "/mnt/" + drive + "/" + remainder
}

func maybeTranslateArgumentForWsl(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return trimmed
	}
	if isWindowsDrivePath(trimmed) || strings.HasPrefix(trimmed, `\\`) {
		return toWslPath(trimmed)
	}
	if strings.Contains(trimmed, `\`) {
		return strings.ReplaceAll(trimmed, `\`, `/`)
	}
	return trimmed
}

func resolveRequestedWslDistro(request string) string {
	if !isWindowsRuntime() {
		return ""
	}
	if _, err := exec.LookPath("wsl"); err != nil {
		log.Fatal("use-linux-distro was requested, but wsl.exe was not found. Install WSL first.")
	}

	trimmed := strings.TrimSpace(request)
	if trimmed == "" || strings.EqualFold(trimmed, "default") {
		return ""
	}

	cmd := exec.Command("wsl", "-l", "-q")
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Failed to list WSL distros while resolving '%s': %v", request, err)
	}

	available := make([]string, 0)
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		available = append(available, line)
		if strings.EqualFold(line, trimmed) {
			return line
		}
	}

	if len(available) == 0 {
		log.Fatalf("No WSL distros are available, but use-linux-distro was requested for '%s'.", request)
	}
	log.Fatalf("WSL distro '%s' was not found. Available distros: %s", request, strings.Join(available, ", "))
	return ""
}

func resolveWslHomeDir(distro string) string {
	if !isWindowsRuntime() {
		return ""
	}
	cmdArgs := []string{"wsl"}
	if distro != "" {
		cmdArgs = append(cmdArgs, "-d", distro)
	}
	cmdArgs = append(cmdArgs, "--", "bash", "-lc", `printf %s "$HOME"`)
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Failed to resolve the WSL home directory: %v", err)
	}
	homeDir := strings.TrimSpace(string(out))
	if homeDir == "" {
		log.Fatal("Failed to resolve the WSL home directory: empty result")
	}
	return homeDir
}

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
			if isWindowsRuntime() {
				fmt.Println("Invalid path. Please ensure it contains the 'envs' directory and 'python.exe'.")
			} else {
				fmt.Println("Invalid path. Please ensure it contains the 'envs' directory and 'bin/python'.")
			}
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
	pythonPath := condaBasePythonPath(path)
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
	cmd := exec.Command(condaBasePythonPath(anacondaPath), "-c", "print('Hello from Python')")
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

// launchPipelineDesigner launches the pipeline designer GUI application
// If yamlPath is provided, it will be opened in the designer
// If yamlPath is empty, the user will be prompted to choose a save location
func launchPipelineDesigner(mainProgramDir string, yamlPath string) {
	if !isWindowsRuntime() {
		log.Fatal("Pipeline Designer is not implemented on Linux yet")
	}

	// Look for the pipeline designer executable
	designerPaths := []string{
		filepath.Join(mainProgramDir, "pipeline-designer", "build", "bin", "pipeline-designer.exe"),
		filepath.Join(mainProgramDir, "pipeline-designer", "pipeline-designer.exe"),
		filepath.Join(mainProgramDir, "build", "bin", "pipeline-designer.exe"),
		filepath.Join(mainProgramDir, "pipeline-designer.exe"),
	}

	var designerExe string
	for _, path := range designerPaths {
		if isFile(path) {
			designerExe = path
			break
		}
	}

	if designerExe == "" {
		fmt.Println("Error: Pipeline Designer executable not found.")
		fmt.Println("Expected location: pipeline-designer/build/bin/pipeline-designer.exe")
		fmt.Println("\nTo build the designer, run:")
		fmt.Println("  cd pipeline-designer")
		fmt.Println("  .\\build.ps1")
		log.Fatal("Pipeline Designer not found")
	}

	fmt.Printf("Launching Pipeline Designer: %s\n", designerExe)

	// Prepare command arguments
	var cmdArgs []string
	if yamlPath != "" {
		// If a YAML file was provided, pass it as an argument
		absYamlPath, err := filepath.Abs(yamlPath)
		if err == nil {
			cmdArgs = []string{absYamlPath}
			fmt.Printf("Opening YAML file: %s\n", absYamlPath)
		}
	} else {
		// No YAML file specified - designer will prompt for save location
		fmt.Println("Starting designer - you will be prompted to choose where to save your pipeline")
	}

	// Launch the designer
	cmd := exec.Command(designerExe, cmdArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Start()
	if err != nil {
		log.Fatalf("Error launching Pipeline Designer: %v", err)
	}

	fmt.Println("Pipeline Designer launched successfully")
	fmt.Println("You can close this window or press Ctrl+C to exit")

	// Wait for the designer to exit
	err = cmd.Wait()
	if err != nil {
		log.Printf("Pipeline Designer exited with error: %v", err)
	}
}

// computeSegmentHash creates a deterministic hash of a segment's content
// Includes: name, type, message, environment, and all commands
// This hash is used to detect if a segment has been modified
func computeSegmentHash(segment Segment) string {
	var builder strings.Builder

	// Include all fields that define the segment's behavior
	builder.WriteString("name:")
	builder.WriteString(segment.Name)
	builder.WriteString("|type:")
	builder.WriteString(segment.Type)
	builder.WriteString("|msg:")
	builder.WriteString(segment.Message)
	builder.WriteString("|env:")
	builder.WriteString(segment.Environment)
	if strings.TrimSpace(segment.UseLinuxDistro) != "" {
		builder.WriteString("|linux:")
		builder.WriteString(strings.TrimSpace(segment.UseLinuxDistro))
	}
	builder.WriteString("|cmds:")
	if len(segment.Env) > 0 {
		keys := make([]string, 0, len(segment.Env))
		for key := range segment.Env {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		builder.WriteString("|envvars:")
		for _, key := range keys {
			builder.WriteString(key)
			builder.WriteString("=")
			builder.WriteString(fmt.Sprintf("%v", segment.Env[key]))
			builder.WriteString(";")
		}
	}

	// Process commands deterministically
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			builder.WriteString(v)
			builder.WriteString("|")
		case map[interface{}]interface{}:
			// Sort keys for deterministic ordering
			keys := make([]string, 0, len(v))
			for k := range v {
				keys = append(keys, fmt.Sprintf("%v", k))
			}
			sort.Strings(keys)

			for _, k := range keys {
				builder.WriteString(k)
				builder.WriteString("=")
				builder.WriteString(fmt.Sprintf("%v", v[k]))
				builder.WriteString(";")
			}
			builder.WriteString("|")
		}
	}

	// Compute SHA256 hash
	hash := sha256.Sum256([]byte(builder.String()))
	return fmt.Sprintf("%x", hash)
}

func resolveSegmentEnv(segment Segment, mainProgramDir, yamlDir string) []string {
	if len(segment.Env) == 0 {
		return nil
	}

	keys := make([]string, 0, len(segment.Env))
	for key := range segment.Env {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	resolved := make([]string, 0, len(keys))
	for _, key := range keys {
		value := fmt.Sprintf("%v", segment.Env[key])
		resolved = append(resolved, fmt.Sprintf("%s=%s", key, resolvePath(value, mainProgramDir, yamlDir)))
	}
	return resolved
}

// determineForcePoint finds the first segment that needs reprocessing
// Returns -1 if all segments are up to date
// Returns the index of the first segment that triggers force mode otherwise
func determineForcePoint(config Config, status Status) int {
	// Build hash map from status file for O(1) lookup
	statusByHash := make(map[string]SegmentStatus)
	for _, s := range status.Segments {
		statusByHash[s.ContentHash] = s
	}

	for i, segment := range config.Run {
		// Skip special control types (but trigger on force type)
		stepType := strings.ToLower(segment.Type)
		if stepType == "pause" || stepType == "stop" {
			continue
		}
		if stepType == "force" {
			fmt.Printf("→ Force mode triggered by explicit 'force' step at '%s'\n", segment.Name)
			return i
		}

		// Compute current hash
		currentHash := computeSegmentHash(segment)

		// Check if this exact segment (by hash) was processed before
		prevStatus, exists := statusByHash[currentHash]

		if !exists {
			fmt.Printf("→ Segment '%s' not found in status (new or modified)\n", segment.Name)
			return i
		}

		if prevStatus.LastProcessed == "" {
			fmt.Printf("→ Segment '%s' exists in status but was never completed\n", segment.Name)
			return i
		}

		// Segment matches and was completed - continue checking
	}

	return -1 // All segments up to date
}

// loadOrCreateStatus loads existing status file or creates a new one
func loadOrCreateStatus(statusPath string, segments []Segment) Status {
	var status Status

	// Try to load existing status file
	if data, err := os.ReadFile(statusPath); err == nil {
		if err := yaml.Unmarshal(data, &status); err == nil {
			// Status file loaded successfully
			return status
		}
	}

	// Create new empty status (segments will be added as they're processed)
	status.Segments = make([]SegmentStatus, 0)
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

// getSegmentStatus finds the status entry for a given segment hash
func getSegmentStatus(status *Status, contentHash string) *SegmentStatus {
	for i := range status.Segments {
		if status.Segments[i].ContentHash == contentHash {
			return &status.Segments[i]
		}
	}
	return nil
}

func updateSegmentStatus(status *Status, statusPath string, segment Segment, contentHash string, startTime time.Time) {
	segmentStatus := getSegmentStatus(status, contentHash)
	if segmentStatus == nil {
		newStatus := SegmentStatus{
			Name:        segment.Name,
			ContentHash: contentHash,
		}
		status.Segments = append(status.Segments, newStatus)
		segmentStatus = &status.Segments[len(status.Segments)-1]
	}
	segmentStatus.LastProcessed = time.Now().Format("2006-01-02")
	segmentStatus.CodeVersion = getVersion()
	segmentStatus.RunDuration = time.Since(startTime).String()
	if err := saveStatus(statusPath, *status); err != nil {
		log.Fatalf("%v", err)
	}
}

// Function to prepare command arguments for Python execution
func makePythonCommand(segment Segment, anacondaPath, mainProgramDir, yamlDir string) []string {
	if !isWindowsRuntime() {
		condaExe := condaExecutable(anacondaPath)
		if condaExe == "" {
			log.Fatal("Conda executable not found. Please install conda or set CONDA_PATH correctly.")
		}

		cmdArgs := []string{condaExe, "run", "--no-capture-output"}
		if strings.EqualFold(segment.Environment, "base") {
			cmdArgs = append(cmdArgs, "-p", anacondaPath)
		} else {
			envPath := filepath.Join(anacondaPath, "envs", segment.Environment)
			if isDirectory(envPath) {
				cmdArgs = append(cmdArgs, "-p", envPath)
			} else {
				cmdArgs = append(cmdArgs, "-n", segment.Environment)
			}
		}

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

// ensureUvEnvironment ensures the environment-specific .venv exists with required dependency group installed
// Creates separate .venv_<group> directories for isolation (like conda envs)
// Only syncs if the specific .venv_<group> doesn't exist
func ensureUvEnvironment(mainProgramDir string, environment string) string {
	spec, err := parseUvEnvironment(environment)
	if err != nil {
		log.Fatal(err)
	}

	// Create environment-specific venv directory.
	// For explicit versions, keep separate env folders to avoid stale reuse.
	venvName := uvVenvName(spec)
	venvPath := filepath.Join(mainProgramDir, venvName)
	legacyVenvPath := filepath.Join(mainProgramDir, fmt.Sprintf(".venv_%s", spec.Group))

	if spec.PythonVersion != "" {
		fmt.Printf("Using UV environment spec '%s' -> dependency group '%s', Python %s\n", environment, spec.Group, spec.PythonVersion)
		fmt.Printf("Resolved UV venv path: %s\n", venvPath)
		if isDirectory(legacyVenvPath) && legacyVenvPath != venvPath {
			fmt.Printf("Found legacy unversioned env at %s; ignoring it for '%s'.\n", legacyVenvPath, environment)
		}
	}

	// Check if this specific environment already exists
	if isDirectory(venvPath) {
		// Environment exists, assume it's up to date
		// Users can delete .venv_<group> to force reinstall
		return venvPath
	}

	// Environment doesn't exist, create it with uv venv + uv pip install
	fmt.Printf("Creating UV environment for '%s' (first time setup)...\n", spec.Group)
	fmt.Printf("Installing dependency group: %s\n", spec.Group)

	uvExe := findUvExecutable(mainProgramDir)
	if uvExe == "" {
		log.Fatal("UV executable not found. Please install UV.")
	}

	// Step 1: Create the venv using uv venv
	venvArgs := []string{"venv", venvPath}
	if spec.PythonVersion != "" {
		venvArgs = append(venvArgs, "--python", spec.PythonVersion)
	}
	venvCmd := exec.Command(uvExe, venvArgs...)
	venvCmd.Dir = mainProgramDir
	venvCmd.Stdout = os.Stdout
	venvCmd.Stderr = os.Stderr

	fmt.Printf("Running: %s %s\n", uvExe, strings.Join(venvArgs, " "))
	if err := venvCmd.Run(); err != nil {
		log.Fatalf("Failed to create venv for '%s': %v", spec.Group, err)
	}

	// Step 2: Install packages using uv pip install with the specific venv
	// This reads from pyproject.toml and installs the specified group
	pipCmd := exec.Command(uvExe, "pip", "install", "--python", venvPath, "--group", spec.Group, ".")
	pipCmd.Dir = mainProgramDir
	pipCmd.Stdout = os.Stdout
	pipCmd.Stderr = os.Stderr

	fmt.Printf("Running: %s pip install --python %s --group %s .\n", uvExe, venvPath, spec.Group)
	if err := pipCmd.Run(); err != nil {
		log.Fatalf("Failed to install packages for '%s': %v", spec.Group, err)
	}

	fmt.Printf("✓ UV environment '%s' created successfully at %s\n", spec.Group, venvPath)
	return venvPath
}

type uvEnvironmentSpec struct {
	Group         string
	PythonVersion string
}

func uvVenvName(spec uvEnvironmentSpec) string {
	venvName := fmt.Sprintf(".venv_%s", spec.Group)
	if spec.PythonVersion != "" {
		versionSuffix := strings.NewReplacer(".", "_", "-", "_", " ", "", "/", "_", "\\", "_").Replace(spec.PythonVersion)
		venvName = fmt.Sprintf(".venv_%s_py%s", spec.Group, versionSuffix)
	}
	return venvName
}

func computeUvEnvironmentSyncHash(mainProgramDir string, environment string) string {
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(strings.TrimSpace(environment)))

	files := []string{
		filepath.Join(mainProgramDir, "pyproject.toml"),
		filepath.Join(mainProgramDir, "uv.lock"),
	}
	for _, filePath := range files {
		_, _ = hasher.Write([]byte("\nFILE:" + filepath.Base(filePath) + "\n"))
		if data, err := os.ReadFile(filePath); err == nil {
			_, _ = hasher.Write(data)
		}
	}

	return fmt.Sprintf("%x", hasher.Sum(nil))
}

// parseUvEnvironment parses environment values like "uv:group" and "uv@3.11:group".
func parseUvEnvironment(environment string) (uvEnvironmentSpec, error) {
	env := strings.TrimSpace(environment)
	lower := strings.ToLower(env)

	if strings.HasPrefix(lower, "uv@") {
		parts := strings.SplitN(env, ":", 2)
		if len(parts) != 2 {
			return uvEnvironmentSpec{}, fmt.Errorf("invalid UV environment format: %s (expected 'uv:group' or 'uv@3.11:group')", environment)
		}

		version := strings.TrimSpace(parts[0][3:])
		group := strings.TrimSpace(parts[1])
		if version == "" || group == "" {
			return uvEnvironmentSpec{}, fmt.Errorf("invalid UV environment format: %s (expected 'uv:group' or 'uv@3.11:group')", environment)
		}

		return uvEnvironmentSpec{Group: strings.ToLower(group), PythonVersion: version}, nil
	}

	if strings.HasPrefix(lower, "uv:") {
		group := strings.TrimSpace(env[3:])
		if group == "" {
			return uvEnvironmentSpec{}, fmt.Errorf("invalid UV environment format: %s (expected 'uv:group' or 'uv@3.11:group')", environment)
		}
		return uvEnvironmentSpec{Group: strings.ToLower(group)}, nil
	}

	return uvEnvironmentSpec{}, fmt.Errorf("invalid UV environment format: %s (expected 'uv:group' or 'uv@3.11:group')", environment)
}

func getRequestedUvPython(spec uvEnvironmentSpec) string {
	if spec.PythonVersion != "" {
		return spec.PythonVersion
	}
	if v := strings.TrimSpace(os.Getenv("UV_DEFAULT_PYTHON")); v != "" {
		return v
	}
	return "3.11"
}

// findUvExecutable locates the uv executable
func findUvExecutable(mainProgramDir string) string {
	// Check local candidates first
	externalUV := filepath.Join(mainProgramDir, "external", "UV")
	localCandidates := []string{}
	if isWindowsRuntime() {
		localCandidates = append(localCandidates,
			filepath.Join(externalUV, "uv.exe"),
			filepath.Join(externalUV, "bin", "uv.exe"),
			filepath.Join(mainProgramDir, "uv.exe"),
			filepath.Join(mainProgramDir, "uv", "uv.exe"),
			filepath.Join(mainProgramDir, "tools", "uv.exe"),
			filepath.Join(mainProgramDir, "tools", "uv", "uv.exe"),
			filepath.Join(mainProgramDir, "bin", "uv.exe"),
		)
	} else {
		localCandidates = append(localCandidates,
			filepath.Join(externalUV, "uv"),
			filepath.Join(externalUV, "bin", "uv"),
			filepath.Join(mainProgramDir, "uv"),
			filepath.Join(mainProgramDir, "tools", "uv"),
			filepath.Join(mainProgramDir, "tools", "uv", "uv"),
			filepath.Join(mainProgramDir, "bin", "uv"),
		)
	}

	for _, p := range localCandidates {
		if isFile(p) {
			return p
		}
	}

	// Check if uv is on PATH
	if path, err := exec.LookPath("uv"); err == nil {
		return path
	}

	return ""
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
	spec, err := parseUvEnvironment(segment.Environment)
	if err != nil {
		log.Fatal(err)
	}

	// Ensure environment-specific .venv_<group> exists with required group
	venvPath := ensureUvEnvironment(mainProgramDir, segment.Environment)
	uvPython := getRequestedUvPython(spec)

	// Build command to run Python from the environment-specific venv
	venvPython := venvPythonPath(venvPath)
	cmdArgs := []string{venvPython}

	// Loop through commands to build the full command line
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			// Skip standalone "python" strings as we're already using venv python
			if strings.ToLower(v) == "python" {
				continue
			}
			resolved := resolvePath(v, mainProgramDir, yamlDir)
			cmdArgs = append(cmdArgs, resolved)
		case map[interface{}]interface{}:
			for flag, value := range v {
				flagStr := fmt.Sprintf("%v", flag)
				// Skip "python" key in maps as we're already using venv python
				if strings.ToLower(flagStr) == "python" {
					// Just add the value (the script path)
					if value != nil && value != "null" {
						valStr := fmt.Sprintf("%v", value)
						resolved := resolvePath(valStr, mainProgramDir, yamlDir)
						cmdArgs = append(cmdArgs, resolved)
					}
					continue
				}

				cmdArgs = append(cmdArgs, flagStr)
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

func makeWslUvCommand(segment Segment, mainProgramDir, yamlDir, distro string) ([]string, string) {
	spec, err := parseUvEnvironment(segment.Environment)
	if err != nil {
		log.Fatal(err)
	}

	wslRepoDir := toWslPath(mainProgramDir)
	uvPython := getRequestedUvPython(spec)
	repoSlug := strings.NewReplacer(" ", "_", ":", "_", `\\`, "_", "/", "_").Replace(filepath.Base(mainProgramDir))
	wslHomeDir := resolveWslHomeDir(distro)
	venvPath := path.Join(wslHomeDir, ".run_pipeline_wsl_venvs", repoSlug, uvVenvName(spec))
	syncHash := computeUvEnvironmentSyncHash(mainProgramDir, segment.Environment)
	stampPath := path.Join(venvPath, ".run_pipeline_uv_sync_hash")

	translatedArgs := make([]string, 0)
	for _, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			if strings.ToLower(v) == "python" {
				continue
			}
			resolved := resolvePath(v, mainProgramDir, yamlDir)
			translatedArgs = append(translatedArgs, maybeTranslateArgumentForWsl(resolved))
		case map[interface{}]interface{}:
			for flag, value := range v {
				flagStr := fmt.Sprintf("%v", flag)
				if strings.ToLower(flagStr) == "python" {
					if value != nil && value != "null" {
						valStr := fmt.Sprintf("%v", value)
						resolved := resolvePath(valStr, mainProgramDir, yamlDir)
						translatedArgs = append(translatedArgs, maybeTranslateArgumentForWsl(resolved))
					}
					continue
				}

				translatedArgs = append(translatedArgs, flagStr)
				if value != nil && value != "null" {
					valStr := fmt.Sprintf("%v", value)
					resolved := resolvePath(valStr, mainProgramDir, yamlDir)
					translatedArgs = append(translatedArgs, maybeTranslateArgumentForWsl(resolved))
				}
			}
		default:
			log.Fatalf("unexpected type %v", reflect.TypeOf(v))
		}
	}

	quotedArgs := make([]string, 0, len(translatedArgs))
	for _, arg := range translatedArgs {
		quotedArgs = append(quotedArgs, shellQuote(arg))
	}

	setupCommands := []string{
		"set -euo pipefail",
		"cd " + shellQuote(wslRepoDir),
		"export UV_LINK_MODE=copy",
		"if ! command -v uv >/dev/null 2>&1; then echo " + shellQuote("UV not found inside the selected Linux distro. Install uv in WSL with: wget -qO- https://astral.sh/uv/install.sh | sh. Then ensure uv is on PATH, e.g. add $HOME/.local/bin to PATH in ~/.bashrc and restart the shell.") + " >&2; exit 1; fi",
		"mkdir -p " + shellQuote(path.Dir(venvPath)),
	}
	for _, envEntry := range resolveSegmentEnv(segment, mainProgramDir, yamlDir) {
		parts := strings.SplitN(envEntry, "=", 2)
		setupCommands = append(setupCommands, "export "+parts[0]+"="+shellQuote(parts[1]))
	}
	if uvPython != "" {
		setupCommands = append(
			setupCommands,
			"if [ ! -d "+shellQuote(venvPath)+" ]; then uv venv "+shellQuote(venvPath)+" --python "+shellQuote(uvPython)+"; fi",
			"if [ ! -f "+shellQuote(stampPath)+" ] || ! grep -qxF "+shellQuote(syncHash)+" "+shellQuote(stampPath)+"; then uv pip install --python "+shellQuote(path.Join(venvPath, "bin", "python"))+" --group "+shellQuote(spec.Group)+" . && printf %s "+shellQuote(syncHash)+" > "+shellQuote(stampPath)+"; fi",
		)
	} else {
		setupCommands = append(
			setupCommands,
			"if [ ! -d "+shellQuote(venvPath)+" ]; then uv venv "+shellQuote(venvPath)+"; fi",
			"if [ ! -f "+shellQuote(stampPath)+" ] || ! grep -qxF "+shellQuote(syncHash)+" "+shellQuote(stampPath)+"; then uv pip install --python "+shellQuote(path.Join(venvPath, "bin", "python"))+" --group "+shellQuote(spec.Group)+" . && printf %s "+shellQuote(syncHash)+" > "+shellQuote(stampPath)+"; fi",
		)
	}
	runCommand := shellQuote(path.Join(venvPath, "bin", "python"))
	if len(quotedArgs) > 0 {
		runCommand += " " + strings.Join(quotedArgs, " ")
	}
	setupCommands = append(setupCommands, runCommand)

	cmdArgs := []string{"wsl"}
	if distro != "" {
		cmdArgs = append(cmdArgs, "-d", distro)
	}
	cmdArgs = append(cmdArgs, "--", "bash", "-lc", strings.Join(setupCommands, "; "))
	return cmdArgs, uvPython
}

// getUvRunnerPrefix ensures `uv` is available and returns the appropriate command prefix:
// - ["cmd","/C","uv","run"] when uv is on PATH
// - ["cmd","/C","pipx","run","uv","run"] as a fallback when uv is not on PATH but pipx is available
// It will attempt `pipx install uv` when uv is missing.
func getUvRunnerPrefix(mainProgramDir string) []string {
	have := func(name string) bool { _, err := exec.LookPath(name); return err == nil }
	shellPrefix := shellCommandPrefix()

	// Prefer a repo-bundled uv executable if present (no system install needed)
	externalUV := filepath.Join(mainProgramDir, "external", "UV")
	localCandidates := []string{}
	if isWindowsRuntime() {
		localCandidates = append(localCandidates,
			filepath.Join(externalUV, "uv.exe"),
			filepath.Join(externalUV, "bin", "uv.exe"),
			filepath.Join(mainProgramDir, "uv.exe"),
			filepath.Join(mainProgramDir, "uv", "uv.exe"),
			filepath.Join(mainProgramDir, "tools", "uv.exe"),
			filepath.Join(mainProgramDir, "tools", "uv", "uv.exe"),
			filepath.Join(mainProgramDir, "bin", "uv.exe"),
		)
	} else {
		localCandidates = append(localCandidates,
			filepath.Join(externalUV, "uv"),
			filepath.Join(externalUV, "bin", "uv"),
			filepath.Join(mainProgramDir, "uv"),
			filepath.Join(mainProgramDir, "tools", "uv"),
			filepath.Join(mainProgramDir, "tools", "uv", "uv"),
			filepath.Join(mainProgramDir, "bin", "uv"),
		)
	}
	for _, p := range localCandidates {
		if isFile(p) {
			return append(append([]string{}, shellPrefix...), p, "run")
		}
	}

	// Attempt an in-repo standalone install into external/UV using the official installer
	// Only try if uv isn't on PATH
	if !have("uv") && isWindowsRuntime() {
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
				return append(append([]string{}, shellPrefix...), p, "run")
			}
		}
	}

	if have("uv") {
		return append(append([]string{}, shellPrefix...), "uv", "run")
	}

	if !isWindowsRuntime() {
		log.Fatal("uv is required for 'uv:' environments. Install uv and ensure it is on PATH.")
	}

	// Try pipx-based options
	if have("pipx") {
		// Quick ephemeral check: pipx run uv --version
		testCmd := exec.Command("cmd", "/C", "pipx", "run", "uv", "--version")
		if err := testCmd.Run(); err == nil {
			fmt.Println("uv not found; using 'pipx run uv' for this session.")
			return append(append([]string{}, shellPrefix...), "pipx", "run", "uv", "run")
		}

		// Try installing uv for persistent availability
		fmt.Println("uv not found; attempting 'pipx install uv' ...")
		install := exec.Command("cmd", "/C", "pipx", "install", "uv")
		install.Stdout = os.Stdout
		install.Stderr = os.Stderr
		_ = install.Run()

		// Re-evaluate availability
		if have("uv") {
			return append(append([]string{}, shellPrefix...), "uv", "run")
		}

		// Fallback to pipx run if it now works
		testCmd2 := exec.Command("cmd", "/C", "pipx", "run", "uv", "--version")
		if err := testCmd2.Run(); err == nil {
			fmt.Println("Using 'pipx run uv' as fallback. Consider adding uv to PATH via 'pipx install uv'.")
			return append(append([]string{}, shellPrefix...), "pipx", "run", "uv", "run")
		}
	}

	// If we get here, we couldn't find or install uv automatically
	fmt.Println("ERROR: 'uv' is not available and automatic installation failed.")
	fmt.Println("Please install uv manually, e.g. 'pipx install uv' or download the standalone binary.")
	log.Fatal("uv is required for 'uv:' environments")
	return append(append([]string{}, shellPrefix...), "uv", "run")
}

// Function to prepare command arguments for ImageJ execution
func makeImageJCommand(segment Segment, imageJPath, mainProgramDir, yamlDir string) []string {
	cmdArgs := []string{imageJPath, "--ij2", "--console"} // Initialize command arguments (GUI mode)

	var macroPath string
	var params []string

	// Loop through each command in the segment's command list
	for i, cmd := range segment.Commands {
		switch v := cmd.(type) {
		case string:
			// First string is the macro file path
			if i == 0 {
				macroPath = resolvePath(v, mainProgramDir, yamlDir)
			} else {
				// Additional strings are treated as separate arguments (if any)
				resolved := resolvePath(v, mainProgramDir, yamlDir)
				cmdArgs = append(cmdArgs, resolved)
			}

		case map[interface{}]interface{}:
			// Map entries become script parameters: key=value
			for flag, value := range v {
				if value != nil && value != "null" {
					flagStr := fmt.Sprintf("%v", flag)
					valStr := fmt.Sprintf("%v", value)
					// Resolve path if it looks like a path placeholder
					resolvedVal := resolvePath(valStr, mainProgramDir, yamlDir)
					params = append(params, fmt.Sprintf("%s=%s", flagStr, resolvedVal))
				}
			}

		default:
			log.Fatalf("unexpected type %v", reflect.TypeOf(v))
		}
	}

	// Build the --run command
	if macroPath != "" {
		// Convert macro path to forward slashes
		macroPath = strings.ReplaceAll(macroPath, "\\", "/")

		if len(params) > 0 {
			// Sort params for deterministic ordering
			sort.Strings(params)
			// Convert paths to forward slashes for ImageJ compatibility
			var quotedParams []string
			for _, p := range params {
				// Replace backslashes with forward slashes in the parameter value
				p = strings.ReplaceAll(p, "\\", "/")
				// Quote the value part after = sign
				parts := strings.SplitN(p, "=", 2)
				if len(parts) == 2 {
					quotedParams = append(quotedParams, fmt.Sprintf("%s='%s'", parts[0], parts[1]))
				} else {
					quotedParams = append(quotedParams, p)
				}
			}
			paramsStr := strings.Join(quotedParams, ",")
			cmdArgs = append(cmdArgs, "--run", macroPath, paramsStr)
		} else {
			// No parameters, just run the macro
			cmdArgs = append(cmdArgs, "--run", macroPath)
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

	// Initialize CLI flag state
	var yamlPath string        // Path to YAML config
	forceReprocessing := false // --force_reprocessing / -f
	designMode := false        // --design / -d
	printHash := false         // --print-hash / -ph: preview status YAML without executing

	// Check if a path is passed as a command-line argument
	for _, arg := range os.Args[1:] {
		if arg == "-h" || arg == "--help" {
			fmt.Println("Usage:")
			fmt.Println("  run_pipeline.exe [options] <path_to_yaml>")
			fmt.Println("")
			fmt.Println("Options:")
			fmt.Println("  -d, --design              Launch the visual pipeline designer GUI.")
			fmt.Println("  -f, --force_reprocessing  Process segments even if they have been previously processed.")
			fmt.Println("  -ph, --print-hash         Print a preview of the *_status.yaml file (no execution).")
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
			fmt.Println("- name: Convert ND2 to OME-TIFF\n  environment: uv@3.11:default\n  commands:\n    - python\n    - '%REPO%/standard_code/python/convert_to_tif.py'\n    - --input-search-pattern: '%YAML%/input_data/**/*.nd2'\n    - --output-folder: '%YAML%/output_data'\n- name: Enable force mode mid-pipeline\n  type: force\n  message: 'Reprocessing all subsequent steps.'\n- name: Pause for inspection\n  type: pause\n  message: 'Paused for user inspection.'\n- name: Stop pipeline\n  type: stop\n  message: 'Pipeline stopped intentionally.'")
			os.Exit(0)
		} else if arg == "--design" || arg == "-d" {
			designMode = true
		} else if arg == "--force_reprocessing" || arg == "-f" {
			forceReprocessing = true
		} else if arg == "--print-hash" || arg == "-ph" {
			printHash = true
		} else {
			yamlPath = arg // Assume the next argument is the YAML file path

			// Check if the YAML file path is valid
			if !isFile(yamlPath) {
				log.Fatalf("The specified YAML path is not a valid file: %v", yamlPath)
			}
		}
	}

	// Handle design mode
	if designMode {
		launchPipelineDesigner(mainProgramDir, yamlPath)
		return
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

	// If --print-hash was requested, output a synthesized status YAML and exit early
	if printHash {
		// Build status preview without executing anything
		preview := Status{Segments: make([]SegmentStatus, 0)}
		codeVersion := getVersion()
		processedDate := time.Now().Format("2006-01-02")
		for _, segment := range config.Run {
			stepType := strings.ToLower(segment.Type)
			if stepType == "pause" || stepType == "stop" || stepType == "force" {
				continue // Skip control steps in status
			}
			hash := computeSegmentHash(segment)
			preview.Segments = append(preview.Segments, SegmentStatus{
				Name:          segment.Name,
				ContentHash:   hash,
				LastProcessed: processedDate,
				CodeVersion:   codeVersion,
				RunDuration:   "", // Unknown until executed
			})
		}
		out, err := yaml.Marshal(&preview)
		if err != nil {
			log.Fatalf("error marshalling preview status YAML: %v", err)
		}
		fmt.Println("# Preview of status file (no execution performed)")
		fmt.Print(string(out))
		return
	}

	// Get the directory of the YAML file for resolving data paths
	yamlDir := filepath.Dir(yamlPath)

	// Create status file path: inputname_status.yaml
	yamlBaseName := filepath.Base(yamlPath)
	yamlNameWithoutExt := strings.TrimSuffix(yamlBaseName, filepath.Ext(yamlBaseName))
	statusPath := filepath.Join(yamlDir, yamlNameWithoutExt+"_status.yaml")

	// Load or create status file
	status := loadOrCreateStatus(statusPath, config.Run)
	fmt.Printf("Using status file: %v\n", statusPath)

	// Determine where force mode should start (unless already forced via CLI)
	forcePointIndex := -1
	if !forceReprocessing {
		forcePointIndex = determineForcePoint(config, status)

		if forcePointIndex == -1 {
			fmt.Println("\n✓ All segments up to date. No reprocessing needed.")
			fmt.Println("  Use --force to run anyway.")
			fmt.Println()

			// Even when everything is up to date, ensure 1-to-1 correspondence
			// by keeping only one status entry per config segment (in order)
			var cleanedSegments []SegmentStatus
			for i := 0; i < len(config.Run); i++ {
				stepType := strings.ToLower(config.Run[i].Type)
				if stepType == "pause" || stepType == "stop" || stepType == "force" {
					continue // Skip control segments
				}
				hash := computeSegmentHash(config.Run[i])
				// Find the status entry with this hash
				for _, s := range status.Segments {
					if s.ContentHash == hash {
						cleanedSegments = append(cleanedSegments, s)
						break // Only keep one entry per hash
					}
				}
			}
			// Only save if we actually cleaned something
			if len(cleanedSegments) != len(status.Segments) {
				status.Segments = cleanedSegments
				if err := saveStatus(statusPath, status); err != nil {
					log.Fatalf("Error saving cleaned status: %v", err)
				}
			}
		} else {
			fmt.Printf("\n⚠ Changes detected at segment %d: '%s'\n",
				forcePointIndex+1, config.Run[forcePointIndex].Name)
			fmt.Println("  → Force mode activated from this point onwards")
			fmt.Println("  → Clearing status entries from this point onwards")
			fmt.Println()

			// Keep only status entries that match config segments BEFORE the force point
			// This ensures 1-to-1 mapping between config and status
			var keptSegments []SegmentStatus
			for i := 0; i < forcePointIndex; i++ {
				stepType := strings.ToLower(config.Run[i].Type)
				if stepType == "pause" || stepType == "stop" || stepType == "force" {
					continue // Skip control segments
				}
				hash := computeSegmentHash(config.Run[i])
				// Find the status entry with this hash
				for _, s := range status.Segments {
					if s.ContentHash == hash {
						keptSegments = append(keptSegments, s)
						break // Only keep one entry per hash
					}
				}
			}
			status.Segments = keptSegments

			if err := saveStatus(statusPath, status); err != nil {
				log.Fatalf("Error saving cleaned status: %v", err)
			}
		}
	} else {
		fmt.Println("\n⚠ Force mode enabled via --force flag")
		fmt.Println("  → All segments will be reprocessed")
		fmt.Println()
		forcePointIndex = 0
	}

	// Build status lookup for fast checking during execution
	statusByHash := make(map[string]*SegmentStatus)
	for i := range status.Segments {
		statusByHash[status.Segments[i].ContentHash] = &status.Segments[i]
	}

	// Iterate over each segment defined in the configuration
	for i, segment := range config.Run {
		// Handle pause and stop types
		stepType := strings.ToLower(segment.Type)
		switch stepType {
		case "pause":
			msg := segment.Message
			if msg == "" {
				msg = "Paused. Press Enter to continue."
			}
			fmt.Printf("\n[PAUSE] %s\n", msg)
			reader := bufio.NewReader(os.Stdin)
			_, _ = reader.ReadString('\n')
			continue
		case "stop":
			msg := segment.Message
			if msg == "" {
				msg = "Stopped by stop step. Exiting."
			}
			fmt.Printf("\n[STOP] %s\n", msg)
			os.Exit(0)
		case "force":
			msg := segment.Message
			if msg == "" {
				msg = "Force mode activated. All subsequent segments will be reprocessed."
			}
			fmt.Printf("\n[FORCE] %s\n", msg)
			forcePointIndex = i
			forceReprocessing = true
			continue
		}

		// Compute hash for this segment
		currentHash := computeSegmentHash(segment)

		// Check if we should skip this segment
		shouldSkip := false
		if forcePointIndex == -1 || i < forcePointIndex {
			if prevStatus, exists := statusByHash[currentHash]; exists && prevStatus.LastProcessed != "" {
				fmt.Printf("✓ Skipping '%s' (unchanged, completed %s)\n\n",
					segment.Name, prevStatus.LastProcessed)
				shouldSkip = true
			}
		}

		if shouldSkip {
			continue
		}

		// Activate force mode if we've reached the force point
		if i >= forcePointIndex && forcePointIndex != -1 {
			forceReprocessing = true
		}

		fmt.Printf("▶ Processing segment: %s\n", segment.Name)

		// Prepare command arguments for executing the environment and subsequent commands
		var cmdArgs []string // Declare cmdArgs here
		useLinuxDistro := strings.TrimSpace(segment.UseLinuxDistro)
		lowerEnvironment := strings.ToLower(segment.Environment)

		if useLinuxDistro != "" {
			if isWindowsRuntime() {
				if lowerEnvironment == "imagej" || lowerEnvironment == "cmd" {
					log.Fatalf("use-linux-distro is not supported with environment '%s'", segment.Environment)
				}
				if !strings.HasPrefix(lowerEnvironment, "uv:") && !strings.HasPrefix(lowerEnvironment, "uv@") {
					log.Fatalf("use-linux-distro is currently only supported for UV environments. Segment '%s' uses '%s'", segment.Name, segment.Environment)
				}

				resolvedDistro := resolveRequestedWslDistro(useLinuxDistro)
				if resolvedDistro == "" {
					fmt.Printf("[info] Routing segment '%s' through the default WSL distro\n", segment.Name)
				} else {
					fmt.Printf("[info] Routing segment '%s' through WSL distro '%s'\n", segment.Name, resolvedDistro)
				}

				var uvPython string
				cmdArgs, uvPython = makeWslUvCommand(segment, mainProgramDir, yamlDir, resolvedDistro)
				fmt.Printf("Constructed command: %v with UV_PYTHON=%s\n", cmdArgs, uvPython)

				cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				startTime := time.Now()
				err := cmd.Run()
				if err != nil {
					fmt.Printf("Error executing command: %v\n", err)
					log.Fatalf("Error")
				}

				updateSegmentStatus(&status, statusPath, segment, currentHash, startTime)
				fmt.Println("")
				fmt.Println("")
				continue
			}

			if isLinuxRuntime() {
				fmt.Printf("[info] Already running on Linux; skipping distro selection for segment '%s'.\n", segment.Name)
			} else {
				log.Fatalf("use-linux-distro is only supported on Windows hosts with WSL or on Linux hosts. Current OS: %s", runtime.GOOS)
			}
		}

		// Determine if the environment is going to be imageJ, uv, cmd, or conda-based Python
		fmt.Println(segment.Environment)
		if lowerEnvironment == "cmd" {
			fmt.Println("running native shell command")
			// Build a single command string for shell execution
			var cmdString strings.Builder
			for i, cmd := range segment.Commands {
				switch v := cmd.(type) {
				case string:
					resolved := strings.TrimSpace(resolvePath(v, mainProgramDir, yamlDir))
					if i > 0 || cmdString.Len() > 0 {
						cmdString.WriteString(" ")
					}
					cmdString.WriteString(resolved)
				case map[interface{}]interface{}:
					for flag, value := range v {
						if cmdString.Len() > 0 {
							cmdString.WriteString(" ")
						}
						cmdString.WriteString(fmt.Sprintf("%v", flag))
						if value != nil && value != "null" {
							valStr := fmt.Sprintf("%v", value)
							resolved := strings.TrimSpace(resolvePath(valStr, mainProgramDir, yamlDir))
							cmdString.WriteString(" ")
							cmdString.WriteString(resolved)
						}
					}
				default:
					log.Fatalf("unexpected type %v", reflect.TypeOf(v))
				}
			}
			cmdLine := strings.TrimSpace(cmdString.String())
			// Resolve any remaining %REPO% or %YAML% tokens in the final command line
			cmdLine = resolvePath(cmdLine, mainProgramDir, yamlDir)
			shellPrefix := shellCommandPrefix()
			cmdArgs := append(shellPrefix, cmdLine)
			// Execute shell command immediately
			fmt.Printf("Constructed command: %v\n", cmdArgs)
			cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
			cmd.Env = append(os.Environ(), resolveSegmentEnv(segment, mainProgramDir, yamlDir)...)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			startTime := time.Now()
			err := cmd.Run()
			if err != nil {
				fmt.Printf("Error executing command: %v\n", err)
				log.Fatalf("Error")
			}
			// Update status in status file
			currentHash := computeSegmentHash(segment)
			updateSegmentStatus(&status, statusPath, segment, currentHash, startTime)
			fmt.Println("")
			fmt.Println("")
			continue
		} else if lowerEnvironment == "imagej" {
			if !isWindowsRuntime() {
				log.Fatal("ImageJ is not implemented on Linux yet")
			}
			fmt.Println("running imageJ")
			imageJPath, err := findImageJPath()
			if err != nil {
				imageJPath = askForImageJPath()
			} else {
				fmt.Printf("Found ImageJ.exe: %v\n", imageJPath)
			}

			cmdArgs = makeImageJCommand(segment, imageJPath, mainProgramDir, yamlDir)
		} else if strings.HasPrefix(lowerEnvironment, "uv:") || strings.HasPrefix(lowerEnvironment, "uv@") {
			// uv-managed environment: no conda activation
			var uvPython string
			cmdArgs, uvPython = makeUvCommand(segment, mainProgramDir, yamlDir)
			// Propagate desired Python version to uv via environment
			// Use UV_PYTHON which uv honors for selecting interpreter
			if uvPython != "3.11" {
				fmt.Printf("[note] Requesting UV_PYTHON=%s (override via UV_DEFAULT_PYTHON). 3.11 is recommended for best wheel coverage on Windows.\n", uvPython)
			}
			cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
			cmd.Env = append(os.Environ(), resolveSegmentEnv(segment, mainProgramDir, yamlDir)...)
			cmd.Env = append(cmd.Env, fmt.Sprintf("UV_PYTHON=%s", uvPython))
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
			updateSegmentStatus(&status, statusPath, segment, currentHash, startTime)
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
		cmd.Env = append(os.Environ(), resolveSegmentEnv(segment, mainProgramDir, yamlDir)...)
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
		updateSegmentStatus(&status, statusPath, segment, currentHash, startTime)
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
