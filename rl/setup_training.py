#!/usr/bin/env python3
"""
Setup script for ambulance RL training environment.
Run this to install all dependencies and verify the setup.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def check_gpu():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available - {gpu_name}")
            return True
        else:
            print("! CUDA not available - will use CPU")
            return False
    except ImportError:
        print("! PyTorch not installed yet")
        return False


def main():
    print("=== Ambulance RL Training Setup ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    
    print(f"✓ Python {sys.version}")
    
    # Install requirements
    print("\n1. Installing requirements...")
    if not run_command(f"{sys.executable} -m pip install -r requirements_rl_training.txt"):
        print("Failed to install requirements")
        return False
    
    # Check GPU
    print("\n2. Checking GPU availability...")
    check_gpu()
    
    # Test imports
    print("\n3. Testing imports...")
    try:
        import torch
        import gymnasium
        import highway_env
        import stable_baselines3
        import open_clip
        import sentence_transformers
        print("✓ All packages imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Create output directories
    print("\n4. Creating output directories...")
    os.makedirs("runs/ppo_ambulance", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("✓ Output directories created")
    
    print("\n=== Setup Complete! ===")
    print("\nTo start training:")
    print("python train_ambulance_ppo_clip.py --profile smoke  # Quick test")
    print("python train_ambulance_ppo_clip.py --profile full   # Full training")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)