#!/bin/bash
# Complete offline RL training pipeline for RTX 5090
# Usage: cd /home/chettra/ITC/Research/AVs && bash run_offline_training.sh

set -e  # Exit on error

echo "========================================"
echo "Offline RL Training Pipeline"
echo "========================================"
echo ""

# Check if running from AVs directory
if [ ! -f "scripts/export_offline_dataset.py" ]; then
    echo "❌ Error: Must run from AVs directory (/home/chettra/ITC/Research/AVs)"
    echo "Usage: cd /home/chettra/ITC/Research/AVs && bash run_offline_training.sh"
    exit 1
fi

# Configuration
DATA_DIR="data/ambulance_dataset_diagnose"
DATASET_DIR="data/offline_dataset"
CHECKPOINT_DIR="checkpoints"

echo "Step 1: Export dataset from parquet files"
echo "=========================================="
python3 scripts/export_offline_dataset.py \
    --input "$DATA_DIR" \
    --output "$DATASET_DIR" \
    --format npz

if [ $? -ne 0 ]; then
    echo "❌ Dataset export failed!"
    exit 1
fi

echo ""
echo "Step 2: Verify pipeline on RTX 5090"
echo "=========================================="
python3 scripts/verify_offline_pipeline.py \
    --data-dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "⚠️  Verification failed! Check errors above."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "Step 3: Choose training method"
echo "=========================================="
echo "1) Offline DQN (value-based RL)"
echo "2) Behavior Cloning (imitation learning)"
echo "3) Both (DQN + BC in parallel)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Training Offline DQN..."
        echo "=========================================="
        python3 offline_rl/trainers/train_offline_dqn.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_dqn" \
            --epochs 100 \
            --batch-size 256 \
            --lr 3e-4 \
            --device cuda
        ;;
    
    2)
        echo ""
        echo "Training Behavior Cloning..."
        echo "=========================================="
        python3 offline_rl/trainers/train_bc.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/bc_pretrain" \
            --epochs 50 \
            --batch-size 512 \
            --lr 1e-4 \
            --device cuda
        ;;
    
    3)
        echo ""
        echo "Training both DQN and BC in parallel..."
        echo "=========================================="
        
        # DQN in background
        echo "Starting DQN training in background..."
        python3 offline_rl/trainers/train_offline_dqn.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_dqn" \
            --epochs 100 \
            --batch-size 256 \
            --lr 3e-4 \
            --device cuda &
        DQN_PID=$!
        
        # BC in foreground
        echo "Starting BC training..."
        python3 offline_rl/trainers/train_bc.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/bc_pretrain" \
            --epochs 50 \
            --batch-size 512 \
            --lr 1e-4 \
            --device cuda
        
        # Wait for DQN to finish
        echo ""
        echo "Waiting for DQN training to complete..."
        wait $DQN_PID
        ;;
    
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✅ Training complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  - $CHECKPOINT_DIR/offline_dqn/"
echo "  - $CHECKPOINT_DIR/bc_pretrain/"
echo ""
echo "Next steps:"
echo "  1. Evaluate trained models on test scenarios"
echo "  2. Visualize training metrics (metrics.json)"
echo "  3. Fine-tune with online PPO (optional)"
echo ""
