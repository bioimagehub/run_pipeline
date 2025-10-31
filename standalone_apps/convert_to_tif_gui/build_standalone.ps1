# Build script for BIPHUB Image Converter standalone executable
# This creates a single .exe file with Python, Java (via scyjava), and all dependencies embedded

param(
    [switch]$Clean = $false,
    [switch]$Debug = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "BIPHUB Image Converter - Build Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Get version from parent VERSION file
$versionFile = "..\..\VERSION"
if (Test-Path $versionFile) {
    $version = Get-Content $versionFile -Raw
    $version = $version.Trim()
    Write-Host "Version: $version" -ForegroundColor Green
} else {
    $version = "1.0.0"
    Write-Host "Warning: VERSION file not found, using default: $version" -ForegroundColor Yellow
}

# Clean previous builds
if ($Clean) {
    Write-Host "`nCleaning previous builds..." -ForegroundColor Yellow
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "*.spec") { Remove-Item -Force "*.spec" }
    Write-Host "Cleaned!" -ForegroundColor Green
}

# Ensure local modules are copied
Write-Host "`nChecking for local modules..." -ForegroundColor Yellow
$modulesToCopy = @{
    "..\..\standard_code\python\convert_to_tif.py" = "local_convert_to_tif.py"
    "..\..\standard_code\python\bioimage_pipeline_utils.py" = "local_bioimage_pipeline_utils.py"
    "..\..\standard_code\python\extract_metadata.py" = "local_extract_metadata.py"
}

foreach ($source in $modulesToCopy.Keys) {
    $dest = $modulesToCopy[$source]
    if (-not (Test-Path $dest)) {
        Write-Host "Copying $dest..." -ForegroundColor Cyan
        Copy-Item $source $dest
        
        # Update imports in copied file
        $content = Get-Content $dest -Raw
        $content = $content -replace "import bioimage_pipeline_utils as rp", "import local_bioimage_pipeline_utils as rp"
        $content = $content -replace "import extract_metadata", "import local_extract_metadata as extract_metadata"
        Set-Content -Path $dest -Value $content
    } else {
        Write-Host "$dest already exists" -ForegroundColor Gray
    }
}

# Build with PyInstaller
Write-Host "`nBuilding executable with PyInstaller..." -ForegroundColor Yellow

$exeName = "BIPHUB_Image_Converter_v$version"

# PyInstaller arguments
$pyinstallerArgs = @(
    "--name=$exeName",
    "--onefile",
    "--windowed",  # No console window
    "--clean",
    
    # Icon (if exists)
    # "--icon=icon.ico",
    
    # Hidden imports for packages that PyInstaller might miss
    "--hidden-import=scyjava",
    "--hidden-import=jpype",
    "--hidden-import=jpype._jvmfinder",
    "--hidden-import=bioio",
    "--hidden-import=bioio.readers",
    "--hidden-import=bioio_bioformats",
    "--hidden-import=bioio_ome_tiff",
    "--hidden-import=bioio_tifffile",
    "--hidden-import=bioio_nd2",
    "--hidden-import=nd2reader",
    "--hidden-import=dask",
    "--hidden-import=dask.array",
    "--hidden-import=cupy",
    "--hidden-import=pystackreg",
    "--hidden-import=imaris_ims_file_reader",
    "--hidden-import=h5py",
    "--hidden-import=gooey",
    "--hidden-import=wx",
    "--hidden-import=colored",
    "--hidden-import=xsdata_pydantic_basemodel",
    "--hidden-import=xsdata_pydantic_basemodel.hooks",
    "--hidden-import=xsdata",
    
    # Collect all data files for scyjava/jpype
    "--collect-all=scyjava",
    "--collect-all=jpype",
    "--collect-all=bioio",
    "--collect-all=bioio_bioformats",
    "--collect-all=bioio_ome_tiff",
    "--collect-all=bioio_tifffile",
    "--collect-all=bioio_nd2",
    "--collect-all=xsdata",
    "--collect-all=xsdata_pydantic_basemodel",
    
    # Gooey resources
    "--collect-all=gooey",
    
    # Local modules
    "--add-data=local_convert_to_tif.py;.",
    "--add-data=local_bioimage_pipeline_utils.py;.",
    "--add-data=local_extract_metadata.py;.",
    
    "convert_to_tif_gui.py"
)

if ($Debug) {
    # Add console for debugging
    $pyinstallerArgs = $pyinstallerArgs | Where-Object { $_ -ne "--windowed" }
    $pyinstallerArgs += "--console"
    Write-Host "Debug mode: Building with console window" -ForegroundColor Magenta
}

Write-Host "`nRunning PyInstaller..." -ForegroundColor Cyan
& .\.venv\Scripts\pyinstaller.exe $pyinstallerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=====================================" -ForegroundColor Green
    Write-Host "Build completed successfully!" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "`nExecutable location:" -ForegroundColor Cyan
    Write-Host "  dist\$exeName.exe" -ForegroundColor White
    
    $exePath = "dist\$exeName.exe"
    if (Test-Path $exePath) {
        $size = (Get-Item $exePath).Length / 1MB
        Write-Host "`nFile size: $([math]::Round($size, 2)) MB" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n=====================================" -ForegroundColor Red
    Write-Host "Build FAILED!" -ForegroundColor Red
    Write-Host "=====================================" -ForegroundColor Red
    exit 1
}
