package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"reflect"
	"run_python_pipeline/go/find_anaconda_path"
	"strings"

	"gopkg.in/yaml.v2"
)

type Segment struct {
	Name          string        `yaml:"name"`
	Environment   string        `yaml:"environment"`
	Commands      []interface{} `yaml:"commands"` // Mixed types: strings or maps
	FileExtension string        `yaml:"file_extension"`
	InputDir      string        `yaml:"input_dir"`
	OutputDir     string        `yaml:"output_dir"`
}

type Config struct {
	Run []Segment         `yaml:"run"`
	Env map[string]string `yaml:"env"`
}

func main() {
	// Determine the location of the anaconda environment
	// This should be defined in an .env file in the same folder as the .go or .exe file
	// But this can also be automatically found with find_anaconda_path
	anacondaPath, err := find_anaconda_path.FindAnacondaPath()
	if err != nil {
		log.Fatalf("Could not find Anaconda please install and define  in .env file: %v", err)
	} else {
		fmt.Printf("Found anaconda base: %v\n", anacondaPath)
	}

	// All the input for running the program is defined in a .yaml file
	// See: .\run_pipeline\processing_pipelines for examples

	//
	// YAML file path handling
	var yamlPath string
	if len(os.Args) > 1 {
		yamlPath = os.Args[1]
		fmt.Printf("Using YAML path from argument: %v\n", yamlPath)
	} else {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Please provide the path to the YAML file: ")
		yamlPath, err = reader.ReadString('\n')
		yamlPath = strings.TrimSpace(yamlPath) // Trim whitespace and newline characters
		if err != nil {
			log.Fatalf("Error reading user input: %v", err)
		}
	}

	data, err := os.ReadFile(yamlPath)
	if err != nil {
		log.Fatalf("error reading YAML file: %v", err)
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		log.Fatalf("error unmarshalling YAML: %v", err)
	}

	// look for a .env file and check anaconda path

	for _, segment := range config.Run {
		fmt.Printf("Processing segment: %s\n", segment.Name)

		cmdArgs := []string{"cmd", "/C"}

		// Activate Anaconda environment
		if segment.Environment == "base" {
			cmdArgs = append(cmdArgs,
				anacondaPath+"\\Scripts\\activate.bat",
				anacondaPath,
				"&&",
			)
		} else {
			cmdArgs = append(cmdArgs,
				anacondaPath+"\\Scripts\\activate.bat",
				anacondaPath,
				"&&",
				"conda", "activate", segment.Environment,
				"&&",
			)
		}

		for _, cmd := range segment.Commands {
			switch v := cmd.(type) {
			case string:
				cmdArgs = append(cmdArgs, v)
			case map[interface{}]interface{}:
				for flag, value := range v {
					cmdArgs = append(cmdArgs, fmt.Sprintf("%v", flag))
					if value != nil && value != "null" {
						cmdArgs = append(cmdArgs, fmt.Sprintf("%v", value))
					}
				}
			default:
				log.Fatalf("unexpected type: %v", reflect.TypeOf(v))
			}
		}

		fmt.Printf("Constructed command: %v\n", cmdArgs)

		cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		err = cmd.Run()
		if err != nil {
			fmt.Printf("Error executing command: %v\n", err)
		}
	}

	// Ask the user to press Enter to exit
	fmt.Print("Processing complete. Press Enter to exit...")
	reader := bufio.NewReader(os.Stdin)
	_, _ = reader.ReadString('\n') // Wait for user input
}
