#!/bin/bash

# Build script for Ehrenreich Collection Segmenter on macOS
# Run from the deployment folder

echo "========================================"
echo "Building Ehrenreich Collection Segmenter"
echo "========================================"
echo

# Check if we're in the deployment folder
if [ ! -f "ehrenreich_segmenter_macos.spec" ]; then
    echo "Error: Please run this script from the deployment folder!"
    echo "Make sure ehrenreich_segmenter_macos.spec exists in this directory."
    exit 1
fi

# Check if the app folder exists
if [ ! -f "../app/main.py" ]; then
    echo "Error: Cannot find the app folder or main.py!"
    echo "Make sure the app folder exists in the parent directory."
    exit 1
fi

echo "Cleaning previous builds..."
if [ -d "build" ]; then
    rm -rf "build"
fi

if [ -d "dist" ]; then
    rm -rf "dist"
fi

echo
echo "Starting PyInstaller build using virtual environment..."
echo "This may take a few minutes..."
echo

# Navigate to parent directory and activate virtual environment
cd ..

# Check for virtual environment
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
elif [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found"
    echo "Looking for .venv/bin/activate or venv/bin/activate"
    echo "Using system Python - this may cause issues"
fi

# Check if PyInstaller is available
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: PyInstaller not found!"
    echo "Please install PyInstaller: pip install pyinstaller"
    exit 1
fi

cd deployment
echo "Building with PyInstaller..."
pyinstaller --clean ehrenreich_segmenter_macos.spec

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "          BUILD SUCCESSFUL!"
    echo "========================================"
    echo
    echo "Application bundle location:"
    echo "  $(pwd)/dist/EhrenreichSegmenter.app/"
    echo
    echo "The entire 'dist/EhrenreichSegmenter.app' bundle"
    echo "can be distributed as a standalone macOS application."
    echo
    
    # Check if the app bundle exists and get its size
    if [ -d "dist/EhrenreichSegmenter.app" ]; then
        app_size=$(du -sh "dist/EhrenreichSegmenter.app" | cut -f1)
        echo "Application bundle size: $app_size"
        
        # Create a ZIP archive for distribution
        echo
        echo "Creating ZIP archive for distribution..."
        cd dist
        zip -r "EhrenreichSegmenter-macOS.zip" "EhrenreichSegmenter.app"
        cd ..
        
        if [ -f "dist/EhrenreichSegmenter-macOS.zip" ]; then
            zip_size=$(du -sh "dist/EhrenreichSegmenter-macOS.zip" | cut -f1)
            echo "ZIP archive created: EhrenreichSegmenter-macOS.zip ($zip_size)"
        fi
        
        # Rename dist to dist-mac for consistency with Windows version
        echo "Renaming dist folder to dist-mac"
        mv "dist" "dist-mac"
        
        echo
        echo "Cleaning up build folder to save space..."
        rm -rf "build"
        echo
        
        echo "Build complete! You can now:"
        echo "1. Run the app directly: open dist-mac/EhrenreichSegmenter.app"
        echo "2. Distribute the ZIP file: dist-mac/EhrenreichSegmenter-macOS.zip"
        echo
        
    else
        echo "Warning: Application bundle not found at dist/EhrenreichSegmenter.app!"
        exit 1
    fi
    
else
    echo
    echo "========================================"
    echo "           BUILD FAILED!"
    echo "========================================"
    echo
    echo "Check the output above for error details."
    echo "Common solutions:"
    echo "- Make sure the virtual environment exists (.venv or venv folder)"
    echo "- Check if there are missing modules in the spec file"
    echo "- Verify that the app runs with 'python ../app/run.py'"
    echo "- Ensure PyInstaller is installed: pip install pyinstaller"
    echo "- On macOS, ensure you have Xcode command line tools installed"
    echo
    exit 1
fi