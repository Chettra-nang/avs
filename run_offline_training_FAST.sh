#!/bin/bash
# ULTRA-FAST offline RL training for RTX 5090
# Optimized for maximum speed with large batches + mixed precision
# Usage: cd /home/chettra/ITC/Research/AVs && bash run_offline_training_FAST.sh

set -e

echo "========================================"
echo "⚡ ULTRA-FAST Offline RL Training"
echo "RTX 5090 Optimized (33GB VRAM)"
echo "========================================"
echo ""

# Check if running from AVs directory
if [ ! -f "scripts/export_offline_dataset.py" ]; then
    echo "❌ Error: Must run from AVs directory"
    echo "Usage: cd AVs && bash run_offline_training_FAST.sh"
    exit 1
fi

# Configuration
DATA_DIR="data/ambulance_dataset_diagnose"
DATASET_DIR="data/offline_dataset"
CHECKPOINT_DIR="checkpoints"

# RTX 5090 optimized batch sizes
BC_BATCH=4096      # 4K batch (BC is lightweight)
DQN_BATCH=2048     # 2K batch (DQN has target network)
PPO_BATCH=1536     # 1.5K batch (PPO has actor+critic)

echo "🚀 RTX 5090 Optimizations Enabled:"
echo "   - Mixed Precision (FP16): 2x speedup"
echo "   - TF32 matmul: 1.5x speedup"
echo "   - Large batches: Maximum GPU utilization"
echo "   - Multi-worker data loading: 8 workers"
echo "   - Persistent workers: Reduce overhead"
echo ""
echo "Expected training times:"
echo "   BC:  ~2-3 minutes  (50 epochs × 3-4 sec/epoch)"
echo "   DQN: ~8-10 minutes (100 epochs × 5-6 sec/epoch)"
echo "   PPO: ~10-12 minutes (100 epochs × 6-7 sec/epoch)"
echo "   ALL: ~20-25 minutes total"
echo ""

# Step 1: Export dataset
if [ ! -f "$DATASET_DIR/offline_dataset.npz" ]; then
    echo "Step 1: Export dataset"
    echo "=========================================="
    python3 scripts/export_offline_dataset.py \
        --input "$DATA_DIR" \
        --output "$DATASET_DIR" \
        --format npz

    if [ $? -ne 0 ]; then
        echo "❌ Dataset export failed!"
        exit 1
    fi
else
    echo "✅ Dataset already exported: $DATASET_DIR/offline_dataset.npz"
fi

echo ""
echo "Step 2: Choose training method"
echo "=========================================="
echo "1) BC only          (~2-3 min)  - Imitation learning"
echo "2) DQN only         (~8-10 min) - Value-based RL"
echo "3) PPO only         (~10-12 min) - Policy gradient RL (BEST)"
echo "4) Train all three  (~20-25 min) - Complete comparison"
echo ""
read -p "Enter choice [1-4]: " choice

start_time=$(date +%s)

case $choice in
    1)
        echo ""
        echo "⚡ Training BC with batch size $BC_BATCH..."
        echo "=========================================="
        python3 offline_rl/trainers/train_bc.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/bc_pretrain" \
            --epochs 50 \
            --batch-size $BC_BATCH \
            --lr 1e-4 \
            --device cuda
        ;;
    
    2)
        echo ""
        echo "⚡ Training DQN with batch size $DQN_BATCH..."
        echo "=========================================="
        python3 offline_rl/trainers/train_offline_dqn.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_dqn" \
            --epochs 100 \
            --batch-size $DQN_BATCH \
            --lr 3e-4 \
            --device cuda
        ;;
    
    3)
        echo ""
        echo "⚡ Training PPO with batch size $PPO_BATCH..."
        echo "=========================================="
        python3 offline_rl/trainers/train_offline_ppo.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_ppo" \
            --epochs 100 \
            --batch-size $PPO_BATCH \
            --lr 3e-4 \
            --device cuda
        ;;
    
    4)
        echo ""
        echo "⚡ Training all three methods (fastest → slowest)..."
        echo "=========================================="
        
        # BC (fastest)
        echo ""
        echo "[1/3] BC Training..."
        python3 offline_rl/trainers/train_bc.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/bc_pretrain" \
            --epochs 50 \
            --batch-size $BC_BATCH \
            --lr 1e-4 \
            --device cuda
        
        bc_time=$(date +%s)
        bc_duration=$((bc_time - start_time))
        echo "✅ BC completed in ${bc_duration}s (~$(($bc_duration / 60))m)"
        
        # DQN (medium)
        echo ""
        echo "[2/3] DQN Training..."
        python3 offline_rl/trainers/train_offline_dqn.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_dqn" \
            --epochs 100 \
            --batch-size $DQN_BATCH \
            --lr 3e-4 \
            --device cuda
        
        dqn_time=$(date +%s)
        dqn_duration=$((dqn_time - bc_time))
        echo "✅ DQN completed in ${dqn_duration}s (~$(($dqn_duration / 60))m)"
        
        # PPO (best performance)
        echo ""
        echo "[3/3] PPO Training..."
        python3 offline_rl/trainers/train_offline_ppo.py \
            --dataset "$DATASET_DIR/offline_dataset.npz" \
            --output "$CHECKPOINT_DIR/offline_ppo" \
            --epochs 100 \
            --batch-size $PPO_BATCH \
            --lr 3e-4 \
            --device cuda
        
        ppo_time=$(date +%s)
        ppo_duration=$((ppo_time - dqn_time))
        echo "✅ PPO completed in ${ppo_duration}s (~$(($ppo_duration / 60))m)"
        ;;
    
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_minutes=$(($total_duration / 60))
total_seconds=$(($total_duration % 60))

echo ""
echo "========================================"
echo "✅ Training Complete!"
echo "========================================"
echo "Total time: ${total_minutes}m ${total_seconds}s"
echo ""
echo "Checkpoints saved to:"
if [ $choice -eq 1 ] || [ $choice -eq 4 ]; then
    echo "  - $CHECKPOINT_DIR/bc_pretrain/"
fi
if [ $choice -eq 2 ] || [ $choice -eq 4 ]; then
    echo "  - $CHECKPOINT_DIR/offline_dqn/"
fi
if [ $choice -eq 3 ] || [ $choice -eq 4 ]; then
    echo "  - $CHECKPOINT_DIR/offline_ppo/"
fi
echo ""
echo "Performance comparison:"
if [ $choice -eq 4 ]; then
    echo "  BC:  ${bc_duration}s (~$(($bc_duration / 60))m)"
    echo "  DQN: ${dqn_duration}s (~$(($dqn_duration / 60))m)"
    echo "  PPO: ${ppo_duration}s (~$(($ppo_duration / 60))m)"
fi
echo ""
echo "Next steps:"
echo "  1. Check metrics: cat checkpoints/*/metrics.json"
echo "  2. Compare performance across methods"
echo "  3. Evaluate best model on test scenarios"
echo "  4. Deploy to production!"
echo ""

