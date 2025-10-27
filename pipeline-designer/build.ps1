# Build script for Pipeline Designer
# This script builds the application and copies necessary resources

# Ensure we're in the pipeline-designer directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Building Pipeline Designer..." -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray

# Build the application
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
Write-Host "✓ Build complete!" -ForegroundColor Green
Write-Host "✓ Executable: $scriptPath\build\bin\pipeline-designer.exe" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "To run: .\build\bin\pipeline-designer.exe" -ForegroundColor Cyan
