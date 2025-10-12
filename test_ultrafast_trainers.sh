#!/bin/bash
# Quick test of ultra-fast trainers before full run
# Tests with minimal epochs to verify compatibility

set -e

echo "========================================"
echo "🧪 ULTRA-FAST TRAINERS - QUICK TEST"
echo "PyTorch 2.5.1 + Python 3.13.3"
echo "========================================"
echo ""

# Check files exist
FEATURES="data/offline_dataset/clip_features.npz"

if [ ! -f "$FEATURES" ]; then
    echo "❌ Error: Pre-computed features not found: $FEATURES"
    echo "   Run: python3 scripts/precompute_clip_features.py first"
    exit 1
fi

echo "✅ Pre-computed features found"
echo ""

# Test BC (fastest - 1 epoch should be <1s)
echo "Test 1/3: BC Ultra-Fast Trainer"
echo "=========================================="
python3 offline_rl/trainers/train_bc_ultrafast.py \
    --dataset "$FEATURES" \
    --output checkpoints/test_bc \
    --epochs 1 \
    --batch-size 4096 \
    --device cuda

echo ""
echo "✅ BC trainer working!"
echo ""

# Test DQN (1 epoch should be <1s)
echo "Test 2/3: DQN Ultra-Fast Trainer"
echo "=========================================="
python3 offline_rl/trainers/train_dqn_ultrafast.py \
    --dataset "$FEATURES" \
    --output checkpoints/test_dqn \
    --epochs 1 \
    --batch-size 2048 \
    --device cuda

echo ""
echo "✅ DQN trainer working!"
echo ""

# Test PPO (1 epoch should be <1s)
echo "Test 3/3: PPO Ultra-Fast Trainer"
echo "=========================================="
python3 offline_rl/trainers/train_ppo_ultrafast.py \
    --dataset "$FEATURES" \
    --output checkpoints/test_ppo \
    --epochs 1 \
    --batch-size 1536 \
    --device cuda

echo ""
echo "✅ PPO trainer working!"
echo ""

echo "========================================"
echo "✅ ALL TESTS PASSED!"
echo "========================================"
echo ""
echo "All three ultra-fast trainers are working correctly."
echo "Ready for full training run!"
echo ""
echo "Next step:"
echo "  bash run_offline_training_ULTRAFAST.sh"
echo "  Choose option 4 (all three methods)"
echo ""

# Clean up test checkpoints
rm -rf checkpoints/test_bc checkpoints/test_dqn checkpoints/test_ppo
echo "Test checkpoints cleaned up."
echo ""
