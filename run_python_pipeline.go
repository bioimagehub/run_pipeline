package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"reflect"
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
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Please provide the path to the YAML file: ")
	yamlPath, _ := reader.ReadString('\n')
	yamlPath = strings.TrimSpace(yamlPath) // Trim whitespace and newline characters

	data, err := os.ReadFile(yamlPath)
	if err != nil {
		log.Fatalf("error reading YAML file: %v", err)
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		log.Fatalf("error unmarshalling YAML: %v", err)
	}

	anacondaPath := config.Env["ANACONDA_PATH"]

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
	_, _ = reader.ReadString('\n') // Wait for user input
}
