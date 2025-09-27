#!/bin/bash

# Quick Ambulance Visual Test Runner
# This script activates the environment and runs the visual test

echo "🚑 Ambulance Visual Test Runner"
echo "================================"

# Check if virtual environment exists
if [ ! -d "avs_venv" ]; then
    echo "❌ Virtual environment 'avs_venv' not found"
    echo "   Please create the virtual environment first"
    exit 1
fi

echo "🔧 Activating virtual environment..."
source avs_venv/bin/activate

echo "✅ Environment activated"
echo "🚀 Starting visual test..."
echo

# Run the visual test
python quick_ambulance_visual_test.py

echo
echo "🏁 Visual test completed"
echo "   If you saw the ambulance running on screen, configuration is correct!"