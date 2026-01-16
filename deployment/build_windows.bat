@echo off
REM Build script for Ehrenreich Collection Segmenter
REM Run from the deployment folder

echo ========================================
echo Building Ehrenreich Collection Segmenter
echo ========================================
echo.

REM Check if we're in the deployment folder
if not exist "ehrenreich_segmenter_windows.spec" (
    echo Error: Please run this script from the deployment folder!
    echo Make sure ehrenreich_segmenter_windows.spec exists in this directory.
    pause
    exit /b 1
)

REM Check if the app folder exists
if not exist "..\app\main.py" (
    echo Error: Cannot find the app folder or main.py!
    echo Make sure the app folder exists in the parent directory.
    pause
    exit /b 1
)

echo Cleaning previous builds...
if exist "build" (
    rmdir /s /q "build"
)

if exist "dist-win" (
    rmdir /s /q "dist-win"
)

echo.
echo Starting PyInstaller build using virtual environment...
echo This may take a few minutes...
echo.

REM Activate virtual environment and run PyInstaller
cd ..
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Virtual environment not found at .venv\Scripts\activate.bat
    echo Using system Python - this may cause issues
)

cd deployment
echo Building with PyInstaller...
pyinstaller --clean ehrenreich_segmenter_windows.spec

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo          BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable location: 
    echo   %cd%\dist-win\EhrenreichSegmenter\
    echo.
    echo Main executable:
    echo   %cd%\dist-win\EhrenreichSegmenter\EhrenreichSegmenter.exe
    echo.
    echo The entire 'dist-win\EhrenreichSegmenter' folder
    echo can be distributed as a standalone application.
    echo.
    
    REM Check if executable exists and get its size
    if exist "dist-win\EhrenreichSegmenter\EhrenreichSegmenter.exe" (
        for %%A in ("dist-win\EhrenreichSegmenter\EhrenreichSegmenter.exe") do (
            echo Executable size: %%~zA bytes
        )

        echo Generating ZIP archive (exe + _internal folder)
        powershell -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%cd%\dist\EhrenreichSegmenter\*' -DestinationPath '%cd%\dist\EhrenreichSegmenter-Windows.zip' -Force"

        echo renaming dist folder to dist-win
        ren "dist" "dist-win"
        echo.

        echo Deleting build folder to save space...
        rmdir /s /q "build"
        echo.

        exit /b 0

    ) else (
        echo No executable found!
    )

) else (
    echo.
    echo ========================================
    echo           BUILD FAILED!
    echo ========================================
    echo.
    echo Check the output above for error details.
    echo Common solutions:
    echo - Make sure the virtual environment exists (.venv folder)
    echo - Check if there are missing modules in the spec file
    echo - Verify that the app runs with 'python ..\app\run.py'
    echo - Ensure PyInstaller is installed: pip install pyinstaller
    echo.
)

pause