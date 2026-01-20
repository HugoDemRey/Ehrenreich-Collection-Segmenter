# Ehrenreich Collection Segmenter
[![GitHub release](https://img.shields.io/github/v/release/HugoDemRey/Ehrenreich-Collection-Segmenter)](https://github.com/HugoDemRey/Ehrenreich-Collection-Segmenter/releases/latest)

Automated opera segmentation for the historic Ehrenreich Collection using computational audio analysis techniques.

## Overview

This project addresses the challenge of automatically segmenting opera recordings into their structural components (arias, ensembles, orchestral interludes). Working with the historic [Ehrenreich Collection](https://www.hkb-interpretation.ch/projekte/ehrenreich-collection), a unique archive of bootleg opera recordings spanning 1965 to 2010, this research develops computational methods to identify movement boundaries that would otherwise require manual annotation by experts. The project combines multiple audio signal processing methods, and a user interface design.

This interface is an interactive PyQt6 application that was developed to allow any user to explore different segmentation methods, tune parameters in real-time, and compare results across multiple approaches.

## Documentation
A comprehensive document detailing the methodologies, experiments and results of this project, including video demonstrations of the application and a detailed list of all parameters, is available online:

👉 [Read the Documentation](https://hugodemrey.github.io/Ehrenreich-Collection-Segmenter/)


## Features

### Multi-Algorithm Segmentation
- **Silence Detection**: Identifies segment boundaries based on audio energy thresholds
- **HRPS Segmentation**: Uses Harmonic-Residual-Percussive Separation for advanced structural analysis  
- **Novelty Curve Analysis**: Detects changes in spectral, harmonic, and rhythmic content
- **Combination Methods**: Merge results from multiple algorithms for enhanced accuracy
- **Feature Alignment**: Align external recordings using dynamic time warping (DTW)

### Interactive Analysis Interface
- **Real-time Parameter Tuning**: Adjust segmentation parameters and see results instantly
- **Interactive Timeline**: Visualize audio waveforms with detected transitions
- **Comparative Analysis**: Run multiple segmentation approaches simultaneously
- **Audio Preview**: Listen to detected segments with integrated playback controls

### Database Integration
- **Naxos Integration**: Search and align with commercial recordings for validation
- **Chromagram Alignment**: Advanced alignment algorithms for music comparison

### Visualization & Export
- **Interactive Plots**: Matplotlib-based visualizations
- **Export Sessions**: Save transitions analysis results and projects.
- **Session Management**: Persistent storage of analysis sessions and configurations

## Download the Application
You can download the latest standalone executable of the Ehrenreich Collection Segmenter from the releases page (Windows and MacOS supported):

👉 [Latest release](https://github.com/HugoDemRey/Ehrenreich-Collection-Segmenter/releases/latest)

## Running the Application Source Code (Development Mode)
The following instructions will tell you how to run the project in development mode, i.e., directly from the source code. If you want to run a standalone executable, please refer to the [download](#download-the-application) section just above.

This project uses [UV](https://docs.astral.sh/uv/) for fast and reliable Python dependency management.

### 1. Install UV

Installation Link: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Setup Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HugoDemRey/Ehrenreich-Collection-Segmenter.git
   cd Ehrenreich-Collection-Segmenter
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   UV will automatically:
   - Create a virtual environment
   - Install all required dependencies from `pyproject.toml`
   - Set up the project for development

### 3. Run the Application

From the repository root directory:

```bash
uv run app/run.py
```

This will launch the interactive PyQt6 application.

## Architecture

The application follows a clean **Model-View-Controller (MVC)** architecture:

### Core Components

- `app/`: PyQt6 user interface application
  - `models/`: Data models and business logic
  - `views/`: UI components and widgets  
  - `controllers/`: Coordination between models and views
  - `src/`: Core audio processing and analysis algorithms
    - `audio/`: Audio file handling and signal processing
    - `audio_features/`: Feature extraction (novelty curves, HRPS, chromagrams)
    - `interfaces/`: Abstract interfaces for extensible components
    - `naxos/`: Database integration and web scraping
    - `utils/`: Utility functions and helpers
- `index_data/`: Data used for the github pages report (_index.html_)
- `deployment/`: Build scripts for creating executables
- `optimization/`: Parameter tuning (optuna scripts)
- `index.html`: Project report homepage (accessible [here](https://hugodemrey.github.io/Ehrenreich-Collection-Segmenter/))


## Building Executables
Before trying to build the app, it's important that you install UV and synchronize the project with `uv sync` (see above). This will generate a `.venv` folder that is required for building the application.
### Windows 
To create standalone executables for Windows, run the build script inside the `deployment` folder:

```bash
./build_windows.bat
```
This will generate an executable (.exe) in the `deployment/dist-win` directory and its corresponding compressed archive (`.zip`).

### MacOS

For MacOS, run (also from `deployment`):

```bash
./build_macos.sh
```
This will generate an executable (.app) in the `deployment/dist-macos` directory and its corresponding compressed archive (`.zip`).

---

*Semester project by Hugo Demule, supervised by Dr. Yannis Rammos at EPFL Digital and Cognitive Musicology Laboratory.*
