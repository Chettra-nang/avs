# Ambulance RL Training with Dataset - Complete Pipeline

## 🚀 Ready for RTX 5090 Training with Your Dataset

This folder contains a **complete RL training pipeline** that uses your collected ambulance dataset for optimal performance.

### 📁 Files Created:
- ✅ `train_ambulance_rl_with_dataset.py` - **Complete pipeline (BC + RL)**
- ✅ `train_bc_from_dataset.py` - Behavior cloning from your dataset
- ✅ `ambulance_dataset_io.py` - Dataset loading utilities
- ✅ `train_ambulance_ppo_clip.py` - Standalone PPO training
- ✅ `clip_embedder.py` - CLIP vision encoder  
- ✅ `language_embedder.py` - Text embedding
- ✅ `ambulance_highway_wrapper.py` - Environment wrapper
- ✅ `features_extractor_clip.py` - Multi-modal feature extractor
- ✅ `requirements_rl_training.txt` - All dependencies
- ✅ `setup_training.py` - Automated setup script

### 🎯 **Complete Training Pipeline**

## Method 1: Full Pipeline (Recommended)
```bash
# Setup
cd avs/rl
python setup_training.py

# Complete pipeline: BC + RL training
python train_ambulance_rl_with_dataset.py \
  --data_dir /path/to/your/ambulance_dataset_fast \
  --algorithm ppo \
  --steps 500000 \
  --seeds 3 \
  --device cuda
```

## Method 2: Step-by-Step Training
```bash
# Step 1: Behavior Cloning from your dataset
python train_bc_from_dataset.py \
  --data_dir /path/to/your/ambulance_dataset_fast \
  --epochs 50 \
  --device cuda

# Step 2: RL training with BC warm-start
python train_ambulance_ppo_clip.py \
  --profile full \
  --steps 500000 \
  --seeds 3 \
  --device cuda
```

## Method 3: RL Only (Skip Dataset)
```bash
# Train RL without using dataset
python train_ambulance_rl_with_dataset.py \
  --algorithm ppo \
  --skip_bc \
  --steps 500000 \
  --device cuda
```

### 📊 **Expected Performance on RTX 5090:**
- **BC Training**: ~10-20 minutes (50 epochs)
- **RL Training**: ~3-4 hours (500K steps)
- **Combined throughput**: ~2000-3000 steps/second
- **Memory usage**: ~8-12GB VRAM

### 🔧 **Algorithm Options:**
```bash
# PPO (recommended for continuous control)
--algorithm ppo

# DQN (good for discrete actions)
--algorithm dqn
```

### 📈 **Monitor Training:**
```bash
tensorboard --logdir runs
```

### 🎯 **Training Pipeline Benefits:**

1. **Behavior Cloning** (uses your dataset):
   - Learns from expert demonstrations
   - Provides intelligent initialization
   - Reduces RL training time

2. **Reinforcement Learning** (improves beyond dataset):
   - Explores new strategies
   - Optimizes for reward
   - Handles novel scenarios

3. **Combined Approach**:
   - Best of both worlds
   - Faster convergence
   - Better final performance

### 📂 **Dataset Format Expected:**
```
your_ambulance_dataset/
├── manifests/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── episodes/
    ├── episode_data_files
    └── ...
```

### 🔥 **RTX 5090 Optimizations:**
- Large batch sizes (512)
- Multiple parallel environments
- Efficient CLIP processing
- Memory-optimized data loading

### ⚡ **Quick Commands for RTX 5090:**
```bash
# Ultra-fast training (recommended)
python train_ambulance_rl_with_dataset.py \
  --data_dir ../data/ambulance_dataset_fast \
  --algorithm ppo \
  --steps 1000000 \
  --seeds 5 \
  --bc_epochs 100 \
  --device cuda

# Monitor progress
tensorboard --logdir runs --port 6006
```

---
*Updated: October 6, 2025*  
*Complete pipeline ready for RTX 5090 training with your ambulance dataset* 🚀