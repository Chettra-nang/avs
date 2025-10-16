# CLIP-RLDrive Runbook — end-to-end

This document records a step-by-step guide to reproduce the CLIP-RLDrive pipeline in this repository: from environment setup, through image/text dataset collection, CLIP fine-tuning, DQN training with a CLIP-shaped reward, and final evaluation.

Use this as the authoritative playbook on the RTX5090 (or any CUDA machine). Commands assume the repository root is `/home/chettra/ITC/Research/AVs` and the virtualenv is `avs_venv` in the same folder.

## Overview

- Collect visual frames + per-frame instruction text (image, text pairs) from highway-env scenarios using the collector: `highway_datacollection/tools/collect_clip_frames.py`.
- Fine-tune an OpenCLIP model (ViT-B style) on the collected dataset using `clip_finetune/finetune_clip.py` and export normalized action text embeddings (`action_embeds.npz`).
- Add a CLIP-based reward term in the environment via `highway_datacollection/wrappers/clip_reward_wrapper.py` (already present) and train an RL agent with Stable-Baselines3 using `scripts/train_dqn.py`.
- Evaluate policies numerically and by running episodes to collect success/collision metrics.

## Table of contents

- Prerequisites and hardware checks
- Environment setup (packages and venv)
- Data collection (commands per scenario)
- Preparing/inspecting the dataset
- CLIP fine-tuning (smoke and full runs)
- Verifying embeddings and quick sanity checks
- DQN training with CLIP reward (smoke and full experiments)
- Evaluation and metrics collection
- Common issues & troubleshooting
- Reproducibility and logging notes

---

## 1) Prerequisites & hardware checks

- On the machine you will use (RTX5090), verify NVIDIA driver and CUDA are available:

```bash
nvidia-smi
# optional: nvcc --version
```

- Ensure there is at least 20 GB free disk space for model checkpoints and caches. The CLIP checkpoint we use is ~850 MB; downloading several models and running experiments will need a few extra GB.

## 2) Project environment

All instructions assume use of the repository venv at `avs_venv`. If you prefer another environment, adapt paths accordingly.

```bash
# from repo root
source /home/chettra/ITC/Research/AVs/avs_venv/bin/activate

# Upgrade packaging utilities
pip install --upgrade pip setuptools wheel

# Install GPU-compatible PyTorch (choose the wheel matching your CUDA version). Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install open_clip_torch highway-env stable-baselines3[extra] gymnasium pillow numpy pandas tqdm
```

Notes:
- If you're using conda, prefer installing `pytorch` via conda with the correct cudatoolkit.
- If `open_clip_torch` installation fails due to a proxy or slow network, download the wheel manually or use a machine with internet access to prefetch.

---

## 3) Data collection

We collect one dataset per scenario. The collector script is:

- `highway_datacollection/tools/collect_clip_frames.py`

It writes PNG frames to `data/highway_multimodal_dataset/frames/<scenario>/` and a CSV `data/highway_multimodal_dataset/texts/<scenario>.csv` with columns:

```
frame_file,instruction,action_id,scenario,episode,step
```

Recommended baseline: collect 500–1,000 frames per scenario. We collected 1000 per scenario in this runbook.

Example commands (run one per scenario):

```bash
# highway-like
python -m highway_datacollection.tools.collect_clip_frames \
  --scenario free_flow \
  --n-episodes 1000 --max-steps 200 --target-frames 1000 --n-agents 1 \
  --output-dir data/highway_multimodal_dataset --every-n-steps 1 --force-render \
  > logs/collect_free_flow.log 2>&1 &

# merge-like
python -m highway_datacollection.tools.collect_clip_frames \
  --scenario lane_closure \
  --n-episodes 1000 --max-steps 200 --target-frames 1000 --n-agents 1 \
  --output-dir data/highway_multimodal_dataset --every-n-steps 1 --force-render \
  > logs/collect_lane_closure.log 2>&1 &

# intersection-like
python -m highway_datacollection.tools.collect_clip_frames \
  --scenario stop_and_go \
  --n-episodes 1000 --max-steps 200 --target-frames 1000 --n-agents 1 \
  --output-dir data/highway_multimodal_dataset --every-n-steps 1 --force-render \
  > logs/collect_stop_and_go.log 2>&1 &
```

What to check while collecting:
- `ls -1 data/highway_multimodal_dataset/frames/<scenario> | wc -l` to watch counts.
- Confirm CSV exists: `head data/highway_multimodal_dataset/texts/<scenario>.csv`
- Look for log errors in `logs/collect_*.log`.

---

## 4) Preparing the dataset

You can fine-tune per-scenario, or combine multiple scenario CSVs into one file.

To combine the three CSVs while preserving the header:

```bash
mkdir -p data/highway_multimodal_dataset/texts/combined || true
(head -n 1 data/highway_multimodal_dataset/texts/free_flow.csv && \
 tail -n +2 -q data/highway_multimodal_dataset/texts/free_flow.csv \
    data/highway_multimodal_dataset/texts/lane_closure.csv \
    data/highway_multimodal_dataset/texts/stop_and_go.csv) \
  > data/highway_multimodal_dataset/texts/all_captions.csv
```

Verify that files and counts look correct:

```bash
ls -l data/highway_multimodal_dataset/frames/
wc -l data/highway_multimodal_dataset/texts/*.csv
head -n 5 data/highway_multimodal_dataset/texts/all_captions.csv
```

Important: the `finetune_clip` loader accepts either full paths in `frame_file` or filenames; it will locate images accordingly. If you move files, update the CSV paths.

---

## 5) CLIP fine-tuning

Script: `clip_finetune/finetune_clip.py`

Two recommended runs:

- Smoke (quick): 3 epochs to validate everything
- Full (paper): 15 epochs, lr 5e-4

Example smoke (already tested):

```bash
python clip_finetune/finetune_clip.py \
  --data-dir data/highway_multimodal_dataset/frames \
  --captions data/highway_multimodal_dataset/texts/stop_and_go.csv \
  --output clip_finetune/outputs_smoke \
  --model roberta-ViT-B-32 \
  --epochs 3 --batch-size 16 --lr 5e-4
```

Full run (combined dataset):

```bash
python clip_finetune/finetune_clip.py \
  --data-dir data/highway_multimodal_dataset/frames \
  --captions data/highway_multimodal_dataset/texts/all_captions.csv \
  --output clip_finetune/outputs_full \
  --model roberta-ViT-B-32 \
  --epochs 15 --batch-size 64 --lr 5e-4
```

Notes and tips:
- The script uses `open_clip`. On first run the model weights will be downloaded — allow a few minutes and ensure good network connectivity.
- If `freeze_except_projection` leaves no trainable parameters, the script will warn and unfreeze the model for the smoke run; for a proper run you should refine the freeze logic to unfreeze only projection layers / last visual block.
- The script writes:
  - `clip_finetune/outputs_full/clip_finetuned.pt`
  - `clip_finetune/outputs_full/action_embeds.npz` (contains normalized embeddings for action ids `'0'`, `'1'`, `'2'`)
  - `clip_finetune/outputs_full/meta.json`

GPU optimizations (RTX5090):
- Increase `--batch-size` to 64 or 128 if memory allows.
- Consider using mixed precision (AMP). The current script does not use AMP; see the troubleshooting section for recommended changes.

---

## 6) Quick verification of embeddings

Sanity check to compute a similarity between a sample image and the saved action embedding:

```python
import numpy as np, torch, open_clip
from PIL import Image

npz = np.load('clip_finetune/outputs_full/action_embeds.npz')
emb0 = npz['0']
model, _, preprocess = open_clip.create_model_and_transforms('roberta-ViT-B-32', pretrained='laion2b_s12b_b32k')
model.eval().cuda()
img = preprocess(Image.open('data/highway_multimodal_dataset/frames/stop_and_go/stop_and_go_ep0000_s0000.png')).unsqueeze(0).cuda()
with torch.no_grad():
    im = model.encode_image(img)
    im = im / im.norm(dim=-1, keepdim=True)
    sim = (im.cpu().numpy() @ emb0.reshape(-1,1)).squeeze()
    print('similarity to action 0:', sim)
```

If similarity values are in [-1,1] and show expected ordering, embeddings are reasonable.

---

## 7) DQN training with CLIP reward

Script: `scripts/train_dqn.py` (accepts `--use-clip`, `--clip-embeds`, `--w-clip`, `--clip-model-name` flags).

Quick smoke run (5k steps) to validate the end-to-end training loop with CLIP reward:

```bash
python scripts/train_dqn.py \
  --env-id highway-fast-v0 \
  --total-timesteps 5000 \
  --use-clip \
  --clip-embeds clip_finetune/outputs_full/action_embeds.npz \
  --clip-model-name roberta-ViT-B-32 \
  --w-clip 1.2 \
  --device cuda
```

Full training (paper baseline):

```bash
python scripts/train_dqn.py \
  --env-id intersection-v1 \
  --total-timesteps 8000 \
  --use-clip \
  --clip-embeds clip_finetune/outputs_full/action_embeds.npz \
  --clip-model-name roberta-ViT-B-32 \
  --w-clip 1.2 \
  --device cuda
```

Notes:
- `ClipRewardWrapper` computes `r_clip` each env step and adds `w_clip * r_clip` to the base reward. The wrapper also stores `r_clip` in `info['r_clip']` so you can log it.
- If `train_dqn.py` does not accept `--device`, edit the script to pass `device='cuda'` to SB3.

---

## 8) Evaluation & metrics

Create an evaluation harness to run N episodes with the final policy and compute:
- success rate (arrive at goal), collision rate, timeout rate
- average episode return
- confusion matrix between CLIP-suggested action (argmax similarity) and agent action

Minimal evaluation snippet (use after loading a trained SB3 model):

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

model = DQN.load('path/to/dqn_checkpoint')
env = ... # create env same as training (use factory)
n_episodes = 100
results = {'success':0,'collision':0,'timeout':0,'return':[]}
for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    tot = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        tot += r
        done = terminated or truncated
    # Choose criteria for success/collision from env info or logs
    # results['return'].append(tot)

print('Average return:', np.mean(results['return']))
```

Customize evaluation metrics using environment-specific info fields.

---

## 9) Troubleshooting & common errors

- Pretrained tag missing: open_clip may not have every pretrained tag. Use an available tag; in this repo we used `laion2b_s12b_b32k`.
- `optimizer got an empty parameter list`: freezing utility removed trainable params. For smoke runs the finetune script now unfreezes the model automatically; for real training modify `freeze_except_projection` to only unfreeze intended small parameter sets.
- `FileNotFoundError` in DataLoader: CSV `frame_file` path may have subfolders. The finetune loader was patched to handle full paths and filename-only rows.
- Slow downloads/timeouts: pre-download weights or confirm network access to Hugging Face.
- GPU OOM: reduce batch size, use gradient accumulation, or enable fp16/AMP.

---

## 10) Reproducibility & logging

- Use a timestamped output directory for each experiment, e.g. `runs/clip_finetune/2025-10-17_01/` and `runs/dqn_clip/2025-10-17_01/`.
- Save `clip_finetune/outputs_full/meta.json` alongside embeddings to capture the model and prompts used.
- Record the full commit hash of the repository when starting experiments:

```bash
git rev-parse --short HEAD > runs/clip_finetune/COMMIT.txt
```

---

## 11) Next steps & experiments

- Sweep `w_clip` in [0.4, 1.2, 2.0], `lr` in [5e-5, 5e-4], and DQN replay/batch sizes.
- Try PPO with the same CLIP shaping to compare stability.
- Replace the simple 3-action prompts with richer sentence variants and test fine-tuning robustness.

If you want, I can:
- Add an AMP (fp16) option to `clip_finetune/finetune_clip.py` and `scripts/train_dqn.py`.
- Create a wrapper script `run_experiment.sh` that executes collection → finetune → train → evaluate automatically and logs everything.

---

If you'd like, I can now generate the `run_experiment.sh` script and an optional small evaluation helper to automate the steps. Which automation should I produce next?
