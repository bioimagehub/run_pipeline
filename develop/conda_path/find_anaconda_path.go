// go/find_anaconda_path/find_anaconda_path.go
package find_anaconda_path

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/joho/godotenv"
)

// FindAnacondaPath attempts to locate the Anaconda path using multiple strategies and returns an error if unsuccessful.
func FindAnacondaPath() (string, error) {
	return findAnacondaPath()
}

func findAnacondaPath() (string, error) {
	// Find anaconda on this computer
	err := godotenv.Load()
	if err != nil {
		fmt.Println("No .env file found.")
	} else {
		condaPath, exists := os.LookupEnv("CONDA_PATH")
		if exists {
			return condaPath, nil
		} else {
			fmt.Println("CONDA_PATH is not set in .env file")
		}
	}

	condaPath := os.Getenv("CONDA_PREFIX")
	if condaPath != "" {
		return condaPath, nil
	}

	if isWindows() {
		cmd := exec.Command("cmd", "/C", "where conda")
		output, err := cmd.CombinedOutput()
		if err == nil {
			for _, line := range strings.Split(string(output), "\n") {
				if strings.TrimSpace(line) != "" {
					return strings.TrimSpace(line), nil
				}
			}
		}

		startMenu := filepath.Join(os.Getenv("APPDATA"), `Microsoft\Windows\Start Menu\Programs`)
		promptPath, err := findShortcutInStartMenu(startMenu, "Anaconda Prompt")
		if err == nil && promptPath != "" {
			targetPath, err := getShortcutTarget(promptPath)
			if err == nil {
				anacondaPath := extractAnacondaPath(targetPath)
				if anacondaPath != "" {
					return anacondaPath, nil
				}
			}
		}
	}

	return "", fmt.Errorf("Anaconda path could not be found")
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
	outputArgs, err := cmdArgs.CombinedOutput()
	if err != nil {
		return targetPath + " %PATH%", nil
	}

	arguments := strings.TrimSpace(string(outputArgs))
	return targetPath + " " + arguments, nil
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
	return strings.Contains(strings.ToLower(os.Getenv("OS")), "windows")
}
