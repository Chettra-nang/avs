#!/usr/bin/env python3
"""
Setup script for Watch Cars Auto - Highway Simulation Viewer

This script helps you set up the environment and dependencies needed
to run the highway simulation viewer.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def setup_virtual_environment():
    """Set up virtual environment."""
    venv_path = Path("venv_watch_cars")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command(f"{sys.executable} -m venv {venv_path}", 
                      "Creating virtual environment"):
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    venv_path = Path("venv_watch_cars")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Linux/macOS
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_path} install --upgrade pip", 
                      "Upgrading pip"):
        return False
    
    # Install core dependencies
    core_deps = [
        "gymnasium>=0.28.0",
        "highway-env>=1.8.0", 
        "pygame>=2.1.0",
        "numpy>=1.21.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"{pip_path} install {dep}", 
                          f"Installing {dep}"):
            return False
    
    # Install optional dependencies
    optional_deps = [
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "pyarrow>=5.0.0"
    ]
    
    print("🔄 Installing optional dependencies...")
    for dep in optional_deps:
        try:
            subprocess.run(f"{pip_path} install {dep}", shell=True, 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {dep} (optional)")
    
    return True

def test_installation():
    """Test if the installation works."""
    venv_path = Path("venv_watch_cars")
    
    # Determine python path
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python"
    else:  # Linux/macOS
        python_path = venv_path / "bin" / "python"
    
    test_script = '''
import gymnasium as gym
import highway_env
import pygame
import numpy as np
print("✅ All core dependencies imported successfully!")

# Test environment creation
try:
    env = gym.make("highway-v0", render_mode="rgb_array")
    obs, info = env.reset()
    env.close()
    print("✅ Highway environment created successfully!")
except Exception as e:
    print(f"⚠️  Environment test failed: {e}")

print("🎉 Installation test completed!")
'''
    
    return run_command(f'{python_path} -c "{test_script}"', 
                      "Testing installation")

def create_run_script():
    """Create a convenient run script."""
    venv_path = Path("venv_watch_cars")
    
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python"
        script_content = f'''@echo off
echo Starting Highway Simulation Viewer...
{python_path} watch_cars_auto.py %*
'''
        script_file = "run_watch_cars.bat"
    else:  # Linux/macOS
        python_path = venv_path / "bin" / "python"
        script_content = f'''#!/bin/bash
echo "Starting Highway Simulation Viewer..."
{python_path} watch_cars_auto.py "$@"
'''
        script_file = "run_watch_cars.sh"
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    if os.name != 'nt':
        os.chmod(script_file, 0o755)
    
    print(f"✅ Created run script: {script_file}")

def main():
    """Main setup function."""
    print("🚗 Highway Simulation Viewer Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Set up virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test installation
    if not test_installation():
        return False
    
    # Create run script
    create_run_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the simulation:")
    if os.name == 'nt':
        print("   run_watch_cars.bat")
        print("   run_watch_cars.bat --scenario dense")
        print("   run_watch_cars.bat --agents 5 --collect-data")
    else:
        print("   ./run_watch_cars.sh")
        print("   ./run_watch_cars.sh --scenario dense")
        print("   ./run_watch_cars.sh --agents 5 --collect-data")
    
    print("\n2. Or activate the virtual environment manually:")
    if os.name == 'nt':
        print("   venv_watch_cars\\Scripts\\activate")
    else:
        print("   source venv_watch_cars/bin/activate")
    print("   python watch_cars_auto.py")
    
    print("\n3. List available scenarios:")
    print("   python watch_cars_auto.py --list-scenarios")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)