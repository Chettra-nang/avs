# Ambulance RL Training with CLIP Vision

This directory contains standalone RL training scripts for ambulance highway scenarios using CLIP vision encoding.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_rl_training.txt
```

### 2. Run Training

**Quick Test (10K steps):**
```bash
python train_ambulance_ppo_clip.py --profile smoke --steps 10000
```

**Full Training (300K steps):**
```bash
python train_ambulance_ppo_clip.py --profile full --steps 300000
```

**RTX 5090 Optimized Training:**
```bash
python train_ambulance_ppo_clip.py --profile full --steps 500000 --seeds 3 --device cuda
```

### 3. Monitor Training
```bash
tensorboard --logdir runs/ppo_ambulance
```

## Files Overview

- `train_ambulance_ppo_clip.py` - Main training script
- `clip_embedder.py` - CLIP vision encoder
- `language_embedder.py` - Text embedder using sentence-transformers
- `ambulance_highway_wrapper.py` - Environment wrapper that adds CLIP+text features
- `features_extractor_clip.py` - Features extractor for multi-modal observations
- `requirements_rl_training.txt` - Required packages

## Configuration

The training script uses reasonable defaults but can be customized via command line:

- `--steps`: Total training steps (default: 300000)
- `--seeds`: Number of random seeds to train (default: 5)
- `--profile`: "smoke" for quick testing, "full" for complete training
- `--device`: "auto", "cpu", "cuda", or "mps"
- `--config`: Path to custom JSON config file

## Expected Performance

- **RTX 5090**: ~2000-3000 steps/second
- **RTX 4090**: ~1500-2500 steps/second
- **RTX 3080**: ~800-1200 steps/second
- **CPU only**: ~50-100 steps/second

## Outputs

- Model checkpoints: `ppo_clip_ambulance_<profile>_seed<N>.zip`
- Tensorboard logs: `runs/ppo_ambulance/seed_<N>/`
- Best models: `runs/ppo_ambulance/seed_<N>/best/`
- VecNormalize stats: `runs/ppo_ambulance/seed_<N>/vecnorm.pkl`

## Scenario

The training uses the "highway_emergency_dense" scenario where an ambulance must:
- Navigate through dense highway traffic
- Reach the hospital quickly
- Maintain safety while using emergency protocols
- Use both visual (CLIP) and textual context for decision making

## Troubleshooting

**CUDA out of memory**: Reduce batch_size or n_envs in config
**ImportError**: Make sure all dependencies are installed
**Slow training**: Ensure CUDA is available and being used