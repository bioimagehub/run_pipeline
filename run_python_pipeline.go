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

// Segment struct defines each segment of the pipeline with relevant attributes
type Segment struct {
	Name          string        `yaml:"name"`                     // The name of the segment
	Environment   string        `yaml:"environment"`              // The Python environment to use
	Commands      []interface{} `yaml:"commands"`                 // Commands to execute (can be strings or maps)
	FileExtension string        `yaml:"file_extension"`           // File extension to consider for input files
	InputDir      string        `yaml:"input_dir"`                // Directory for input files
	OutputDir     string        `yaml:"output_dir"`               // Directory for output files
	LastProcessed string        `yaml:"last_processed,omitempty"` // Timestamp of when this segment was last processed
}

// Config struct to hold the overall configuration structure
type Config struct {
	Run []Segment `yaml:"run"` // Slice of segments representing the commands to be processed
}

func main() {
	// Find the Anaconda installation path using a helper function
	anacondaPath, err := find_anaconda_path.FindAnacondaPath()
	if err != nil {
		log.Fatalf("Could not find Anaconda, please install and define in .env file: %v", err)
	} else {
		fmt.Printf("Found Anaconda base: %v\n", anacondaPath)
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

	// Get the directory of the executing binary (main program)
	executablePath, err := os.Executable()
	if err != nil {
		log.Fatalf("error getting executable path: %v", err)
	}
	mainProgramDir := filepath.Dir(executablePath) // Directory of the main program executable

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
		cmdArgs := []string{"cmd", "/C"} // Windows command line execution prefix
		// Determine which environment to activate for Python
		if segment.Environment == "base" {
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
			case string: // If the command is a string
				// Check if it is a relative path
				if strings.HasPrefix(v, "./") {
					if strings.HasSuffix(v, ".py") {
						// If it's a Python script, resolve it relative to the main program directory
						v = strings.TrimPrefix(v, "./")      // Remove the './' prefix
						v = filepath.Join(mainProgramDir, v) // Combine with main program path
					} else {
						// Otherwise, it's a data path, resolve it relative to the YAML file directory
						v = strings.TrimPrefix(v, "./") // Remove the './' prefix
						v = filepath.Join(yamlDir, v)   // Combine with YAML directory
					}
				}

				cmdArgs = append(cmdArgs, v) // Add the resolved path to command arguments
			case map[interface{}]interface{}: // If the command is a map (for flags and values)
				// Iterate through the map entries
				for flag, value := range v {
					cmdArgs = append(cmdArgs, fmt.Sprintf("%v", flag)) // Add flag to command arguments
					if value != nil && value != "null" {               // Check if value is valid
						cmdArgs = append(cmdArgs, fmt.Sprintf("%v", value)) // Add value to command arguments
					}
				}
			default:
				log.Fatalf("unexpected type: %v", reflect.TypeOf(v)) // Handle unexpected types
			}
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
	}

	// Write the updated configuration back to the YAML file
	data, err = yaml.Marshal(&config) // Marshal the updated config struct
	if err != nil {
		log.Fatalf("error marshalling updated YAML: %v", err)
	}

	err = os.WriteFile(yamlPath, data, 0644) // Write it to the YAML path
	if err != nil {
		log.Fatalf("error writing YAML file: %v", err)
	}

	// Prompt the user that processing is complete and wait for input
	fmt.Print("Processing complete. Press Enter to exit...")
	reader := bufio.NewReader(os.Stdin)
	_, _ = reader.ReadString('\n') // Wait for user input before exiting
}
