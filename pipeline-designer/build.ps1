# Build script for Pipeline Designer
# This script builds the application and copies necessary resources

# Ensure we're in the pipeline-designer directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Building Pipeline Designer..." -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray

# Check if Node.js is installed
Write-Host "Checking prerequisites..." -ForegroundColor Cyan
$nodeVersion = node --version 2>$null
if (-not $nodeVersion) {
    Write-Host "ERROR: Node.js is not installed!" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}
Write-Host "Node.js $nodeVersion found" -ForegroundColor Green

# Check if Wails is installed
$wailsVersion = wails version 2>$null
if (-not $wailsVersion) {
    Write-Host "ERROR: Wails CLI is not installed!" -ForegroundColor Red
    Write-Host "Installing Wails CLI..." -ForegroundColor Yellow
    go install github.com/wailsapp/wails/v2/cmd/wails@latest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install Wails CLI!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Wails CLI installed" -ForegroundColor Green
}
else {
    Write-Host "Wails CLI found" -ForegroundColor Green
}

# Install frontend dependencies if node_modules doesn't exist
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Set-Location frontend
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install frontend dependencies!" -ForegroundColor Red
        Set-Location ..
        exit 1
    }
    Set-Location ..
    Write-Host "Frontend dependencies installed" -ForegroundColor Green
}
else {
    Write-Host "Frontend dependencies already installed" -ForegroundColor Green
}

# Build the application
Write-Host "Building application..." -ForegroundColor Cyan
wails build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Copying CLI definitions..." -ForegroundColor Cyan

# Remove old cli_definitions to avoid nesting issues
if (Test-Path "build\bin\cli_definitions") {
    Remove-Item -Path "build\bin\cli_definitions" -Recurse -Force
}

# Copy CLI definitions folder structure recursively to build directory
Copy-Item -Path "cli_definitions" -Destination "build\bin\cli_definitions" -Recurse -Force

Write-Host "" -ForegroundColor Green
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "Executable: $scriptPath\build\bin\pipeline-designer.exe" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "To run: .\build\bin\pipeline-designer.exe" -ForegroundColor Cyan
