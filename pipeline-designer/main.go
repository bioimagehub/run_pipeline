package main

import (
	"embed"
	"fmt"
	"log"
	"os"

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

	// Check for debug mode
	if os.Getenv("DEBUG") != "" || os.Getenv("PIPELINE_DEBUG") != "" {
		mainLogger.Println("DEBUG MODE ENABLED via environment variable")
		fmt.Fprintln(os.Stderr, "[DEBUG MODE ACTIVE]")
	} // Get command line arguments
	var filePath string
	if len(os.Args) > 1 {
		filePath = os.Args[1]
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
