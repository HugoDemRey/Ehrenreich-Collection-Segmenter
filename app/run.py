#!/usr/bin/env python3
"""
Launcher script for the Ehrenreich Collection Segmenter.
Optimized for PyInstaller builds.
"""

import os
import sys


def main():
    """Main launcher function."""
    print("Ehrenreich Segmenter Launcher")
    print("=" * 40)
    # Change to app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    print("Changed working directory to app folder:", app_dir)

    # Import and run the application
    import traceback

    try:
        print("Starting Ehrenreich Segmenter...")
        from main import main as app_main

        app_main()
    except ImportError as e:
        print(f"Import error: {e}")
        traceback.print_exc()
        print("Make sure all files are in the correct location")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
