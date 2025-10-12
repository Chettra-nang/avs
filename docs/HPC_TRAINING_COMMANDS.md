## HPC Training runbook — from local preprocessing to training on HPC

This document contains copy-pasteable commands to prepare data and model artifacts locally, push code to your git remote, transfer heavy artifacts to an HPC, and run full CLIP fine-tuning and RL training there. Adjust paths, module names, and scheduler flags to match your cluster.

### Quick overview
- Local: prepare dataset → extract frames → create labels (auto/manual) → fine-tune CLIP (optionally small smoke locally) → extract CLIP image embeddings (.npz) → package artifacts.
- Remote (HPC): clone repo / create environment → transfer artifacts (rsync/scp or S3) → run full CLIP fine-tune and RL training with SLURM job scripts → retrieve models & logs.

---

### 1) Assumptions & useful paths
- Repo root: `AVs/` (you are working inside `/home/chettra/ITC/Research/AVs`).
- Important scripts in repo:
  - `AVs/scripts/extract_frames_for_clip.py` — extract images + `frames.csv` from Parquet episodes
  - `AVs/scripts/auto_label_frames.py` — auto-label a balanced set of images
  - `AVs/scripts/finetune_clip.py` — fine-tune CLIP on labeled image/text pairs
  - `AVs/scripts/extract_clip_features.py` — compute CLIP image embeddings → `.npz`
  - `AVs/scripts/clip_reward.py` — CLI helper to score an image with saved text embeddings
  - `AVs/wrappers/clip_reward_wrapper.py` — Gym wrapper to add CLIP reward during RL

---

### 2) Local prep — quick smoke test (recommended)
These commands assume you have a Python venv or conda env. Replace `venv` or `conda` commands as needed.

1) Activate local venv (example):

```bash
# from repo root
source avs_venv/bin/activate
python --version
pip install -r requirements.txt
pip install -r requirements_watch_cars.txt  # if you need extra deps
```

2) Run extractor on a small subset (smoke):

```bash
# create output dir
mkdir -p AVs/data/clip_frames
python3 AVs/scripts/extract_frames_for_clip.py \
  --input AVs/data/ambulance_dataset_diagnose \
  --output AVs/data/clip_frames \
  --max-per-scenario 200
# Result: AVs/data/clip_frames/frames.csv and PNG images under that directory
```

3) Auto-label (balanced N samples, e.g., 500):

```bash
python3 AVs/scripts/auto_label_frames.py \
  --frames AVs/data/clip_frames/frames.csv \
  --out AVs/data/clip_frames/labels_autolabel.csv \
  --n 500 \
  --copy-to AVs/data/clip_frames/labeled_images \
  --seed 42

# Check that images were copied to AVs/data/clip_frames/labeled_images/images/
ls -lh AVs/data/clip_frames/labeled_images/images | head
```

4) Run a *small* CLIP finetune smoke (1 epoch) locally to confirm everything works (optional but recommended):

```bash
python3 AVs/scripts/finetune_clip.py \
  --labels AVs/data/clip_frames/labels_autolabel.csv \
  --image-root AVs/data/clip_frames/labeled_images \
  --out-dir AVs/models/clip_finetuned_smoke \
  --epochs 1 \
  --batch-size 8

# After this, ensure AVs/models/clip_finetuned_smoke exists and text_embeddings.json was created
ls -la AVs/models/clip_finetuned_smoke
```

5) Extract image embeddings (smoke) from the finetuned model or base CLIP:

```bash
python3 AVs/scripts/extract_clip_features.py \
  --frames-csv AVs/data/clip_frames/frames.csv \
  --model-dir AVs/models/clip_finetuned_smoke \
  --out AVs/data/clip_features_smoke.npz \
  --batch-size 32

python -c "import numpy as np; print(np.load('AVs/data/clip_features_smoke.npz').files)"
```

If the smoke steps pass, proceed to packaging and HPC transfer.

---

### 3) Package artifacts to transfer to HPC
Decide what to put in git vs. what to transfer separately. Large artifacts you typically don't commit (images, heavy model checkpoints, .npz). Options:
- Keep code in git and transfer artifacts to HPC via rsync/scp/S3.
- Use Git LFS only for model checkpoints if you want them versioned with the repo (requires LFS configured on remote).

Typical packaging commands (create tar.gz and checksums):

```bash
# create a dist dir
mkdir -p /tmp/avs_artifacts
cp -r AVs/models/clip_finetuned /tmp/avs_artifacts/ || true
cp AVs/data/clip_features.npz /tmp/avs_artifacts/ 2>/dev/null || true
cp AVs/data/clip_frames/frames.csv /tmp/avs_artifacts/
tar -czf /tmp/avs_artifacts_$(date +%Y%m%d).tar.gz -C /tmp avs_artifacts
sha256sum /tmp/avs_artifacts_$(date +%Y%m%d).tar.gz > /tmp/avs_artifacts_$(date +%Y%m%d).tar.gz.sha256
```

---

### 4) Push code to git (only code)

```bash
# from repo root
git checkout -b your_branch_name
git add .
git commit -m "Add HPC runbook and CLIP training artifacts prep"
git push origin your_branch_name
```

Note: Do not commit large binary artifacts (models, many images). Use the packaging/transfer steps above.

---

### 5) Transfer artifacts to HPC
Choose one option below depending on what your HPC allows.

Option A — rsync over SSH (recommended for large transfers, resumable):

```bash
# from your local machine
REMOTE_USER=username
REMOTE_HOST=hpc.example.edu
REMOTE_DIR=/scratch/$REMOTE_USER/avs_artifacts
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"
rsync -avh --progress /tmp/avs_artifacts_*.tar.gz $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
rsync -avh --progress AVs/scripts $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/  # optional: copy scripts too
```

Option B — scp (simple, not resumable):

```bash
scp /tmp/avs_artifacts_*.tar.gz $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
```

Option C — S3 / object store
- Upload tar.gz to S3 or cluster-provided object storage and use `aws s3 cp` or cluster CLI to pull it on the HPC.

---

### 6) On the HPC — basic environment setup
Below are typical steps; your cluster may use modules or a specific conda environment.

1) SSH to the cluster and unpack artifacts:

```bash
ssh $REMOTE_USER@$REMOTE_HOST
cd /scratch/$REMOTE_USER
tar -xzf avs_artifacts_*.tar.gz -C .
ls -la avs_artifacts
```

2) Clone the repo (code) on the HPC (if not using local copy):

```bash
cd $HOME
git clone git@github.com:your_org/avs.git
cd avs
git checkout your_branch_name
```

3) Create a Python environment (conda recommended on clusters):

```bash
# create and activate conda env
module load anaconda/2023.11  # or use environment modules if provided
conda create -n avs_hpc python=3.10 -y
conda activate avs_hpc
pip install -r requirements.txt
```

If your cluster provides GPUs, request CUDA-enabled Python packages compatible with cluster CUDA drivers (or use a prebuilt conda environment with CUDA).

---

### 7) Example SLURM job — CLIP fine-tune (1 GPU, single node)
Adjust SBATCH parameters to match your scheduler/account/project.

Save this as `run_finetune_clip.sbatch` on the HPC and `sbatch run_finetune_clip.sbatch`.

```bash
#!/bin/bash
#SBATCH --job-name=finetune-clip
#SBATCH --output=logs/finetune_clip.%j.out
#SBATCH --error=logs/finetune_clip.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e
module load anaconda
conda activate avs_hpc
mkdir -p logs

# unpack artifacts (if placed in your $SCRATCH)
cd $SCRATCH/$USER
tar -xzf avs_artifacts_*.tar.gz -C . || true

# Run full finetune (example: 15 epochs, batch=32)
python3 $HOME/avs/AVs/scripts/finetune_clip.py \
  --labels $SCRATCH/$USER/avs_artifacts/labels_autolabel.csv \
  --image-root $SCRATCH/$USER/avs_artifacts/labeled_images \
  --out-dir $SCRATCH/$USER/avs_models/clip_finetuned \
  --epochs 15 \
  --batch-size 32 \
  --lr 5e-4

echo "finetune complete"
```

Notes:
- If you want distributed multi-GPU training, the training script must support it (use accelerate or torch.distributed). Modify SBATCH `--gres` and launch using `srun`/`python -m torch.distributed.launch`/`accelerate launch` as appropriate.

---

### 8) Example SLURM job — RL training with CLIP reward
This assumes your RL trainer script exists (replace `train_rl.py` below with your actual trainer). The wrapper `AVs/wrappers/clip_reward_wrapper.py` will be used inside the training script to augment reward.

Create `run_rl_clip.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=rl-clip
#SBATCH --output=logs/rl_clip.%j.out
#SBATCH --error=logs/rl_clip.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -e
module load anaconda
conda activate avs_hpc
mkdir -p logs

# Example run: replace with your RL training entrypoint and flags
python3 $HOME/avs/AVs/rl/train_rl.py \
  --env some_env_name \
  --num-steps 10000000 \
  --clip-model-dir $SCRATCH/$USER/avs_models/clip_finetuned \
  --clip-text-embeddings $SCRATCH/$USER/avs_models/clip_finetuned/text_embeddings.json \
  --use-clip-reward True \
  --clip-reward-weight 1.2 \
  --save-dir $SCRATCH/$USER/avs_rl_runs/run_clip_$SLURM_JOB_ID

echo "rl training complete"
```

---

### 9) Monitor & debug tips
- Stream logs: `tail -f logs/finetune_clip.<jobid>.out`
- Check GPU: `srun --pty --gres=gpu:1 nvidia-smi` or `nvidia-smi` inside job
- If OOM on GPU: reduce batch size or use gradient accumulation
- If Python package mismatch: verify CUDA/CuDNN versions and reinstall compatible torch wheel or use conda package with matching CUDA

---

### 10) After training — copy back artifacts
Compress and copy results back to local machine or to persistent storage:

```bash
# on HPC
tar -czf clip_finetuned_${SLURM_JOB_ID}.tar.gz -C $SCRATCH/$USER avs_models/clip_finetuned
scp $SCRATCH/$USER/clip_finetuned_${SLURM_JOB_ID}.tar.gz $LOCAL_USER@$LOCAL_HOST:/path/to/store/

# or use rsync for large dirs
rsync -avh --progress $REMOTE_USER@$REMOTE_HOST:$SCRATCH/$USER/avs_models/clip_finetuned ./local_models/
```

Compute and verify checksums to ensure no corruption:

```bash
sha256sum clip_finetuned_${SLURM_JOB_ID}.tar.gz
# compare with the checksum you created on the cluster
```

---

### 11) Reproducibility & tips
- Fix random seeds in training scripts (`random`, `numpy`, `torch`, and determinism flags) if you need exact reproducibility.
- Save training command-line args together with the model (e.g., `train_args.json`).
- Log metrics to a persistent place (tensorboard, Weights & Biases, or plain CSVs in $SCRATCH) so you can inspect later.

---

### 12) Quick troubleshooting checklist
- If jobs fail immediately: check python path, conda env, and module loads inside the job script.
- If CUDA/driver errors: module load correct CUDA or use a conda/pip wheel matching cluster CUDA version.
- If out-of-memory: drop batch size, increase mem or request a GPU with more memory.

---

If you'd like, I can also generate ready-to-use `sbatch` files tuned to your cluster's common settings (partition name, available GPUs per node, module system). Tell me the cluster details (partition names, CUDA version, scheduler restrictions) and I'll adapt them.

---

End of runbook.
