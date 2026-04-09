package find_anaconda_path

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"github.com/joho/godotenv"
)

// FindAnacondaPath attempts to locate the Anaconda path using multiple strategies and returns an error if unsuccessful.
func FindAnacondaPath() (string, error) {
	return findAnacondaPath()
}

// The rest of the helper functions remain the same, prefacing them with lowercase if not intended for export

func findAnacondaPath() (string, error) {

	// Find anaconda using the .env file
	err := godotenv.Load()
	if err != nil {
		fmt.Println("No .env file found.")
	} else {
		condaPath, exists := os.LookupEnv("CONDA_PATH")
		if exists {
			condaPath = normalizeCondaBase(condaPath)
			return condaPath, nil
		} else {
			fmt.Println("CONDA_PATH is not set in .env file")
		}
	}

	// Check environment variables
	condaPath := os.Getenv("CONDA_PREFIX")

	if condaPath != "" {
		normalizedPath := strings.ReplaceAll(condaPath, "\\", "/")

		// Locate path to base env ( this is what is needed run_pipeline)
		components := strings.Split(normalizedPath, "/")
		envsIndex := -1
		for i, component := range components {
			if component == "envs" {
				envsIndex = i
			}
		}
		if envsIndex != -1 && envsIndex+1 < len(components) {
			// Construct the base path of conda
			condaPath = strings.Join(components[:envsIndex], `/`)
		}

		condaPath = normalizeCondaBase(condaPath)
		updateCondaPath(condaPath)
		return condaPath, nil

	}
	fmt.Println("Conda not found using CONDA_PREFIX")

	if base, err := getCondaBaseFromPath(); err == nil {
		base = normalizeCondaBase(base)
		updateCondaPath(base)
		return base, nil
	}

	if !isWindows() {
		if base, ok := findLinuxCondaBaseFromCommonLocations(); ok {
			base = normalizeCondaBase(base)
			updateCondaPath(base)
			return base, nil
		}
	}

	if isWindows() {
		// For windows only

		// Try where conda in cmd
		// cmd := exec.Command("cmd", "/C", "where conda")
		// output, err := cmd.CombinedOutput()
		// if err == nil {
		// 	for _, line := range strings.Split(string(output), "\n") {
		// 		condaPath := strings.TrimSpace(line)
		// 		if strings.TrimSpace(line) != "" {
		// 			updateCondaPath(condaPath)
		// 			return strings.TrimSpace(line), nil
		// 		}
		// 	}
		// }
		// fmt.Println("Conda not found using 'cmd where conda'")

		startMenu := filepath.Join(os.Getenv("APPDATA"), `Microsoft\Windows\Start Menu\Programs`)
		promptPath, err := findShortcutInStartMenu(startMenu, "Anaconda Prompt")
		if err == nil && promptPath != "" {
			targetPath, err := getShortcutTarget(promptPath)
			if err == nil {
				condaPath := extractAnacondaPath(targetPath)
				if condaPath != "" {
					condaPath = normalizeCondaBase(condaPath)
					updateCondaPath(condaPath)
					return condaPath, nil
				}
			}
		}
		fmt.Println("Conda not found using 'shortcut in Start meny'")
	}

	return "", fmt.Errorf("anaconda path could not be found, please define a")
}

func normalizeCondaBase(path string) string {
	normalized := strings.ReplaceAll(strings.TrimSpace(path), "\\", "/")
	normalized = strings.TrimSuffix(normalized, "/Scripts/activate")
	normalized = strings.TrimSuffix(normalized, "/Scripts/activate.bat")
	normalized = strings.TrimSuffix(normalized, "/bin/activate")
	return strings.TrimSuffix(normalized, "/")
}

func getCondaBaseFromPath() (string, error) {
	condaExe, err := exec.LookPath("conda")
	if err != nil {
		return "", err
	}

	cmd := exec.Command(condaExe, "info", "--base")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed running conda info --base: %w", err)
	}

	base := strings.TrimSpace(string(out))
	if base == "" {
		return "", fmt.Errorf("empty base path returned by conda info --base")
	}

	return base, nil
}

func findLinuxCondaBaseFromCommonLocations() (string, bool) {
	homeDir, _ := os.UserHomeDir()
	common := []string{
		filepath.Join(homeDir, "miniconda3"),
		filepath.Join(homeDir, "anaconda3"),
		"/opt/conda",
		"/opt/miniconda3",
		"/usr/local/miniconda3",
		"/usr/local/anaconda3",
	}

	for _, candidate := range common {
		if candidate == "" {
			continue
		}
		if isDirectory(filepath.Join(candidate, "envs")) && isFile(filepath.Join(candidate, "bin", "python")) {
			return candidate, true
		}
	}

	return "", false
}

// updateCondaPath updates or appends the CONDA_PATH variable in the .env file located in the program's directory.
func updateCondaPath(condaPath string) error {
	// Get the directory of the executable
	exeDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("could not determine working directory: %w", err)
	}

	// Define the relative path to the .env file
	envFilePath := filepath.Join(exeDir, ".env")

	fmt.Printf("Saving the Anaconda path to: %v\n", envFilePath)

	file, err := os.OpenFile(envFilePath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return fmt.Errorf("failed to open .env file: %w", err)
	}
	defer file.Close()

	// Read lines from the file
	scanner := bufio.NewScanner(file)
	var lines []string
	var condaPathUpdated bool

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "CONDA_PATH") {
			lines = append(lines, fmt.Sprintf("CONDA_PATH=%s", condaPath))
			condaPathUpdated = true
		} else {
			lines = append(lines, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading .env file: %w", err)
	}

	// If CONDA_PATH was not found and updated, append it
	if !condaPathUpdated {
		lines = append(lines, fmt.Sprintf("CONDA_PATH=%s", condaPath))
	}

	// Seek back to the beginning of the file to overwrite
	if _, err := file.Seek(0, 0); err != nil {
		return fmt.Errorf("error seeking .env file: %w", err)
	}

	// Overwrite file with updated lines and truncate the rest
	writer := bufio.NewWriter(file)
	for _, line := range lines {
		_, err := writer.WriteString(line + "\n")
		if err != nil {
			return fmt.Errorf("error writing to .env file: %w", err)
		}
	}
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("error flushing writes to .env file: %w", err)
	}

	return nil
}

func findShortcutInStartMenu(startMenuDir, shortcutName string) (string, error) {
	var foundPath string
	err := filepath.Walk(startMenuDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if strings.Contains(info.Name(), shortcutName) && strings.HasSuffix(info.Name(), ".lnk") {
			foundPath = path
			return filepath.SkipDir
		}
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("error walking the path %s: %v", startMenuDir, err)
	}

	if foundPath != "" {
		return foundPath, nil
	}
	return "", fmt.Errorf("shortcut not found in start menu")
}

func getShortcutTarget(shortcutPath string) (string, error) {
	powerShellCmd := fmt.Sprintf("(New-Object -COMObject WScript.Shell).CreateShortcut('%s').TargetPath", shortcutPath)
	cmd := exec.Command("powershell", "-Command", powerShellCmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	targetPath := strings.TrimSpace(string(output))

	powerShellCmdArgs := fmt.Sprintf("(New-Object -COMObject WScript.Shell).CreateShortcut('%s').Arguments", shortcutPath)
	cmdArgs := exec.Command("powershell", "-Command", powerShellCmdArgs)
	outputArgs, _ := cmdArgs.CombinedOutput()
	arguments := strings.TrimSpace(string(outputArgs))

	return fmt.Sprintf("%s %s", targetPath, arguments), nil
}

func extractAnacondaPath(arguments string) string {
	re := regexp.MustCompile(`[\w\:\\]+Anaconda[\w\\]*`)
	match := re.FindString(arguments)

	if match != "" {
		return match
	}
	return ""
}

func isWindows() bool {
	return runtime.GOOS == "windows"
}

func isDirectory(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
