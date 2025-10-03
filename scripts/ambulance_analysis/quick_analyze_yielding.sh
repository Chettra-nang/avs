#!/bin/bash
# Quick script to analyze ambulance yielding behavior

echo "🚑 Ambulance Yielding Behavior Analysis"
echo "========================================"

# Activate virtual environment
cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate

# Find first available ambulance dataset
DATA_FILE=$(find /home/chettra/ITC/Research/AVs/data/ambulance_dataset -name "*_transitions.parquet" -type f | head -1)

if [ -z "$DATA_FILE" ]; then
    echo "❌ No ambulance dataset found!"
    exit 1
fi

echo "📊 Using dataset: $DATA_FILE"
echo ""

# Run analysis
python3 scripts/ambulance_analysis/visualize_yielding_behavior.py \
    --data "$DATA_FILE" \
    --animate \
    --output output/ambulance_yielding_analysis

echo ""
echo "✅ Analysis complete!"
echo "📁 Check output/ambulance_yielding_analysis/ for results"
