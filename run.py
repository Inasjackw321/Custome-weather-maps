#!/usr/bin/env python3
"""
Weather Risk Maps - Easy Launcher

Double-click this file to launch the Weather Risk Maps application.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append('numpy')

    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')

    try:
        import cartopy
    except ImportError:
        missing.append('cartopy')

    try:
        import requests
    except ImportError:
        missing.append('requests')

    try:
        import scipy
    except ImportError:
        missing.append('scipy')

    try:
        from PIL import Image
    except ImportError:
        missing.append('pillow')

    return missing

def install_dependencies(packages):
    """Install missing packages."""
    print("Installing missing dependencies...")
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
    print("Dependencies installed successfully!\n")

def main():
    print("=" * 50)
    print("  Weather Risk Maps Generator")
    print("=" * 50)
    print()

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ").strip().lower()
        if response == 'y':
            install_dependencies(missing)
        else:
            print("Cannot run without dependencies. Exiting.")
            sys.exit(1)

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'weather_risk_maps.py')

    # Launch the main application
    print("Launching application...")
    os.chdir(script_dir)

    # Import and run
    sys.path.insert(0, script_dir)
    from weather_risk_maps import main as app_main
    app_main()

if __name__ == '__main__':
    main()
