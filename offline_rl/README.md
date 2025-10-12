# Offline RL Training Scripts

This directory contains offline reinforcement learning trainers for the ambulance highway navigation project.

## 📁 Directory Structure

```
offline_rl/
├── rl_langvision/              # Vision-language modules (CLIP + text encoders)
│   ├── clip_embedder.py       # CLIP ViT-B/32 image encoder
│   ├── amb_highway_wrapper_clip.py  # Gym wrapper for multimodal observations
│   ├── cached_embedder.py     # Pre-computed text embeddings
│   ├── language_embedder.py   # On-the-fly text encoding
│   ├── features_extractor_clip.py   # SB3 feature extractor
│   ├── reward_wrappers.py     # Custom reward shaping
│   └── yielding_traffic.py    # Emergency vehicle behavior
│
└── trainers/
    ├── train_offline_dqn.py   # Offline DQN trainer
    └── train_bc.py            # Behavior cloning trainer
```

## 🚀 Quick Start

### 1. Export Dataset (from AVs root directory)

```bash
cd /home/chettra/ITC/Research/AVs

python3 scripts/export_offline_dataset.py \
    --input data/ambulance_dataset_diagnose \
    --output data/offline_dataset
```

### 2. Train Offline DQN

```bash
python3 offline_rl/trainers/train_offline_dqn.py \
    --dataset data/offline_dataset/offline_dataset.npz \
    --output checkpoints/offline_dqn \
    --epochs 100 \
    --batch-size 256 \
    --device cuda
```

### 3. Train Behavior Cloning

```bash
python3 offline_rl/trainers/train_bc.py \
    --dataset data/offline_dataset/offline_dataset.npz \
    --output checkpoints/bc_pretrain \
    --epochs 50 \
    --batch-size 512 \
    --device cuda
```

## 📦 Dependencies

**Required** (install in your venv):
```bash
pip install torch torchvision  # PyTorch with CUDA support
pip install open-clip-torch    # CLIP model
pip install sentence-transformers  # Text embeddings
pip install numpy pandas pyarrow pillow tqdm
```

**Check installation**:
```bash
python3 << 'EOF'
import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

import open_clip
print(f"✅ OpenCLIP installed")

from sentence_transformers import SentenceTransformer
print(f"✅ Sentence Transformers installed")
EOF
```

## 🔧 Training Options

### Offline DQN

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | required | Path to .npz dataset |
| `--output` | required | Checkpoint output directory |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 256 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target network soft update rate |
| `--device` | cuda | Device (cuda/cpu) |

### Behavior Cloning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | required | Path to .npz dataset |
| `--output` | required | Checkpoint output directory |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 512 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--val-split` | 0.1 | Validation split ratio |
| `--device` | cuda | Device (cuda/cpu) |

## 📊 Output Files

After training, you'll find:

```
checkpoints/
├── offline_dqn/
│   ├── best_model.pt          # Best model by Q-value
│   ├── final_model.pt         # Final epoch checkpoint
│   ├── checkpoint_epoch20.pt  # Intermediate checkpoints
│   └── metrics.json           # Training curves
│
└── bc_pretrain/
    ├── best_model.pt          # Best model by validation accuracy
    ├── final_model.pt         # Final epoch checkpoint
    └── metrics.json           # Training curves
```

## 🐛 Troubleshooting

### Import Error: No module named 'torch'

Install PyTorch in your virtual environment:
```bash
source /path/to/avs_venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Import Error: No module named 'rl_langvision'

Make sure you're running from the AVs root directory:
```bash
cd /home/chettra/ITC/Research/AVs
python3 offline_rl/trainers/train_offline_dqn.py ...
```

### CUDA out of memory

Reduce batch size:
```bash
python3 offline_rl/trainers/train_offline_dqn.py \
    --batch-size 128 \  # Instead of 256
    ...
```

## 📚 Architecture

### CLIP-based Q-Network

```
Grayscale (C,H,W) → Convert to RGB → CLIP ViT-B/32 → 512-d embedding
                                                            ↓
                                                    MLP Q-network
                                                            ↓
                                                    Q-values for 5 actions
```

### Behavior Cloning Policy

```
Grayscale (C,H,W) → Convert to RGB → CLIP ViT-B/32 → 512-d embedding
                                                            ↓
                                                    MLP policy head
                                                            ↓
                                                    Action logits
```

## 🎯 Next Steps

1. **Evaluate trained models** - Create evaluation script
2. **Visualize training curves** - Plot metrics.json
3. **Fine-tune with online PPO** - Use BC checkpoint as initialization
4. **Test on new scenarios** - Collect more data and retrain

## 📄 License

See main AVs project LICENSE.
