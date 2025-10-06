#!/usr/bin/env python3
"""
CUDA and PyTorch diagnostic script for RTX 5090 setup.
Run this to check your CUDA installation and PyTorch GPU support.
"""

import sys
import subprocess

def run_command(cmd):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    print("🔍 CUDA and PyTorch Diagnostic for RTX 5090")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check CUDA toolkit
    print("\n📋 CUDA Toolkit:")
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✅ CUDA toolkit installed")
        print(stdout)
    else:
        print("❌ CUDA toolkit not found")
        print("Install with: sudo apt install nvidia-cuda-toolkit")
    
    # Check nvidia-smi
    print("\n🖥️  GPU Status:")
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print("✅ NVIDIA drivers working")
        print(stdout)
    else:
        print("❌ nvidia-smi failed")
        print(f"Error: {stderr}")
    
    # Check PyTorch
    print("\n🔥 PyTorch CUDA Support:")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("❌ PyTorch CUDA support not available")
            print("Possible fixes:")
            print("1. Install CUDA-enabled PyTorch:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("2. Check CUDA drivers: nvidia-smi")
            print("3. Check CUDA toolkit: nvcc --version")
    
    except ImportError:
        print("❌ PyTorch not installed")
        print("Install with: pip install torch torchvision")
    
    # Check other packages
    print("\n📦 Package Versions:")
    packages = ["open-clip-torch", "transformers", "stable-baselines3", "gymnasium", "highway-env"]
    for pkg in packages:
        try:
            module = __import__(pkg.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: not installed")
    
    # Environment variables
    print("\n🌍 Environment Variables:")
    import os
    cuda_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"]
    for var in cuda_vars:
        value = os.environ.get(var, "not set")
        print(f"{var}: {value}")
    
    print("\n" + "=" * 50)
    print("🎯 Quick Fixes for Common Issues:")
    print("1. Install CUDA drivers: sudo apt install nvidia-driver-535")
    print("2. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
    print("3. Install PyTorch with CUDA:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("4. Reboot system after driver installation")
    print("5. Check with: python -c 'import torch; print(torch.cuda.is_available())'")

if __name__ == "__main__":
    main()