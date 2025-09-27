#!/bin/bash

# Fast Ambulance Data Collection Runner
# This script provides different options for optimized data collection

echo "🚑 Fast Ambulance Data Collection Options"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "avs_venv" ]; then
    echo "❌ Virtual environment 'avs_venv' not found"
    echo "   Please create the virtual environment first"
    exit 1
fi

echo "🔧 Activating virtual environment..."
source avs_venv/bin/activate

# Check available options
echo ""
echo "Available optimization options:"
echo "1. Standard Collection (baseline)"
echo "2. Parallel Collection (multi-process)"
echo "3. GPU-Optimized Collection (GPU acceleration)"
echo "4. Quick Test (5 episodes per scenario)"
echo ""

# Get user choice
read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🚀 Running standard collection..."
        python collecting_ambulance_data/examples/basic_ambulance_collection.py \
            --episodes 50 \
            --max-steps 100 \
            --output-dir data/ambulance_standard \
            --batch-size 10
        ;;
    2)
        echo "🚀 Running parallel collection..."
        python collecting_ambulance_data/examples/parallel_ambulance_collection.py \
            --episodes 50 \
            --max-steps 100 \
            --output-dir data/ambulance_parallel \
            --max-workers 4 \
            --batch-optimize
        ;;
    3)
        echo "🚀 Running GPU-optimized collection..."
        python collecting_ambulance_data/examples/gpu_optimized_collection.py \
            --episodes 50 \
            --max-steps 100 \
            --output-dir data/ambulance_gpu \
            --batch-size 10 \
            --profile
        ;;
    4)
        echo "🚀 Running quick test collection..."
        python collecting_ambulance_data/examples/basic_ambulance_collection.py \
            --episodes 5 \
            --max-steps 50 \
            --output-dir data/ambulance_quick_test \
            --batch-size 5
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-4."
        exit 1
        ;;
esac

echo ""
echo "🏁 Collection completed!"
echo "📊 Check the output directory for results"