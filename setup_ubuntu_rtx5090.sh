#!/bin/bash
# Ubuntu Setup Script for RTX 5090 GPU-Accelerated Ambulance Data Collection

echo "🐧 Ubuntu RTX 5090 Setup for Ambulance Data Collection"
echo "======================================================"

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first:"
    echo "   sudo apt update"
    echo "   sudo apt install nvidia-driver-535"
    echo "   sudo reboot"
    exit 1
fi

echo "🔍 Checking GPU..."
nvidia-smi

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA toolkit not found. Installing..."
    sudo apt update
    sudo apt install nvidia-cuda-toolkit
fi

echo "🔍 CUDA Version:"
nvcc --version

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv avs_venv_ubuntu

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source avs_venv_ubuntu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (CRITICAL!)
echo "🚀 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA installation
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available - check installation')
    exit(1)
"

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements_rtx5090.txt

# Final verification
echo "✅ Setup complete! Testing GPU collection..."
python -c "
import torch
import sys
if not torch.cuda.is_available():
    print('❌ GPU setup failed')
    sys.exit(1)
    
print('🎉 GPU setup successful!')
print(f'Ready for RTX 5090 acceleration with {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🚀 Ready to run GPU-accelerated data collection!"
echo "💡 Use this command:"
echo "python collecting_ambulance_data/examples/final_gpu_accelerated_collection.py \\"
echo "    --episodes 10000 \\"
echo "    --max-steps 100 \\"
echo "    --gpu-intensity 20 \\"
echo "    --seed 42 \\"
echo "    --batch-size 8 \\"
echo "    --output-dir data/ambulance_dataset_150k"