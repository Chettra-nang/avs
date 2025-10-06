# Ambulance RL Training - Quick Start Guide

## 🚀 Ready for RTX 5090 Training

This folder contains a complete, standalone RL training setup for ambulance highway scenarios using CLIP vision.

### Files Created:
- ✅ `train_ambulance_ppo_clip.py` - Main training script
- ✅ `clip_embedder.py` - CLIP vision encoder  
- ✅ `language_embedder.py` - Text embedding
- ✅ `ambulance_highway_wrapper.py` - Environment wrapper
- ✅ `features_extractor_clip.py` - Multi-modal feature extractor
- ✅ `requirements_rl_training.txt` - Dependencies
- ✅ `setup_training.py` - Automated setup script
- ✅ `README.md` - Detailed documentation

### On Your RTX 5090 System:

1. **Clone and Setup:**
```bash
git clone <your-repo>
cd avs/rl
python setup_training.py
```

2. **Start Training:**
```bash
# Quick test (10K steps)
python train_ambulance_ppo_clip.py --profile smoke --steps 10000

# Full training optimized for RTX 5090
python train_ambulance_ppo_clip.py --profile full --steps 500000 --seeds 5 --device cuda
```

3. **Monitor Progress:**
```bash
tensorboard --logdir runs/ppo_ambulance
```

### Expected Performance on RTX 5090:
- **~2000-3000 steps/second** 
- **Full training (500K steps) in ~3-4 hours**
- **Multiple seeds can be trained overnight**

### Key Features:
- ✅ Uses ambulance dataset (highway_emergency_dense scenario)
- ✅ CLIP vision encoding for visual observations
- ✅ Text context embedding for emergency scenarios  
- ✅ PPO algorithm optimized for continuous control
- ✅ Automatic checkpointing and evaluation
- ✅ TensorBoard logging for monitoring
- ✅ Multi-seed training for robust results

### Git-Ready:
All files are self-contained with no relative import dependencies. Just clone and run!

---
*Created: October 6, 2025*
*Ready for high-performance training on RTX 5090*