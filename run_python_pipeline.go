package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath" // Added for handling file paths
	"reflect"
	"run_python_pipeline/go/find_anaconda_path"
	"strings"
	"time"

	"gopkg.in/yaml.v2" // YAML processing
)

// Requested Features
// Add a force reprocess tag

// Segment struct defines each segment of the pipeline with relevant attributes
type Segment struct {
	Name        string        `yaml:"name"`        // The name of the segment
	Environment string        `yaml:"environment"` // The Python environment to use
	Commands    []interface{} `yaml:"commands"`    // Commands to execute (can be strings or maps)
	//FileExtension string        `yaml:"file_extension"`           // File extension to consider for input files
	//InputDir      string        `yaml:"input_dir"`                // Directory for input files
	//OutputDir     string        `yaml:"output_dir"`               // Directory for output files
	LastProcessed string `yaml:"last_processed,omitempty"` // Timestamp of when this segment was last processed
}

// Config struct to hold the overall configuration structure
type Config struct {
	Run []Segment `yaml:"run"` // Slice of segments representing the commands to be processed
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
	if strings.HasPrefix(v, "./") {
		v = strings.TrimPrefix(v, "./")
		if strings.HasSuffix(v, ".py") {
			return filepath.Join(mainProgramDir, v)
		}
		return filepath.Join(yamlDir, v)
	}
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
				saveToEnvFile(anacondaPath)
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

// saveToEnvFile saves the Anaconda path to a .env file.
func saveToEnvFile(path string) {
	envFilePath := filepath.Join(GetBaseDir(), ".env")
	file, err := os.OpenFile(envFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Error opening .env file: %v", err)
	}
	defer file.Close()

	if _, err := file.WriteString(fmt.Sprintf("CONDA_PATH=%s\n", path)); err != nil {
		log.Fatalf("Error writing to .env file: %v", err)
	}
	fmt.Println("Conda path saved to .env file.")
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

// Empty function for ImageJ command preparation
func makeImageJCommand(segment Segment) []string {
	// Still to implement
	fmt.Printf("segment: %v\n", segment)
	return nil
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

	// Check if a path is passed as a command-line argument
	if len(os.Args) > 1 {
		yamlPath = os.Args[1]
		fmt.Printf("Using YAML path from argument: %v\n", yamlPath)
	} else {
		// If not provided, prompt the user for the path
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Please provide the path to the YAML file: ")
		yamlPath, err = reader.ReadString('\n')
		yamlPath = strings.TrimSpace(yamlPath) // Remove any trailing newline characters
		if err != nil {
			log.Fatalf("Error reading user input: %v", err)
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

	// Iterate over each segment defined in the configuration
	for i, segment := range config.Run {
		// Check if this segment has already been processed
		if segment.LastProcessed != "" {
			fmt.Printf("Skipping segment %s, already processed on %s\n", segment.Name, segment.LastProcessed)
			continue // Skip to the next segment if already processed
		}

		fmt.Printf("Processing segment: %s\n", segment.Name)

		// Prepare command arguments for executing the environment and subsequent commands
		var cmdArgs []string // Declare cmdArgs here

		// Determine if the environment is going to be imageJ or Python
		if strings.ToLower(segment.Environment) == "iamgej" {
			cmdArgs = makeImageJCommand(segment)
			// TODO: Process the ImageJ commands
		} else {
			anacondaPath, err := find_anaconda_path.FindAnacondaPath()
			if err != nil {
				anacondaPath = askForAnacondaPath()
			} else {
				fmt.Printf("Found Anaconda base: %v\n", anacondaPath)
			}

			cmdArgs = makePythonCommand(segment, anacondaPath, mainProgramDir, yamlDir)

			// ... existing code ...
		}
		// Print the constructed command arguments for debugging
		fmt.Printf("Constructed command: %v\n", cmdArgs)

		// Create the command using the constructed arguments
		cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
		cmd.Stdout = os.Stdout // Redirect standard output to console
		cmd.Stderr = os.Stderr // Redirect standard error to console

		// Execute the command
		err = cmd.Run()
		if err != nil {
			fmt.Printf("Error executing command: %v\n", err)
			log.Fatalf("Error") // Log fatal error on execution failure
		}

		// Update last_processed with the current date if the command was successful
		config.Run[i].LastProcessed = time.Now().Format("2006-01-02")

		// Write the updated configuration back to the YAML file
		data, err = yaml.Marshal(&config) // Marshal the updated config struct
		if err != nil {
			log.Fatalf("error marshalling updated YAML: %v", err)
		}

		err = os.WriteFile(yamlPath, data, 0644) // Write it to the YAML path
		if err != nil {
			log.Fatalf("error writing YAML file: %v", err)
		}
		fmt.Println("") // Add some space between the segment prints
		fmt.Println("") // Add some space between the segment prints
	}

	// Prompt the user that processing is complete and wait for input
	fmt.Print("Processing complete. Press Enter to exit...")
	reader := bufio.NewReader(os.Stdin)
	_, _ = reader.ReadString('\n') // Wait for user input before exiting
}
