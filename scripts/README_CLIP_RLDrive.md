CLIP-RLDrive ambulance pipeline scripts

Files:
- collect_ambulance_to_clip_dataset.py
    Consolidates parallel collector outputs into CLIP dataset layout (frames/ + texts/ CSVs).

- run_ambulance_all.sh
    Orchestration script: runs parallel ambulance collection, consolidates dataset, runs CLIP finetune smoke and launches a DQN smoke training run.

Usage:
- Small smoke run (local CPU):
    ./scripts/run_ambulance_all.sh

- Full run (use RTX5090 / CUDA):
    ./scripts/run_ambulance_all.sh --full

Notes:
- The collector writes per-batch directories in the output dir and consolidated_index.json; the consolidation script will search those paths.
- CLIP finetune expects a captions CSV. The consolidation script will auto-generate per-scenario CSVs with default captions if no metadata found.
- Edit the script to change scenario list or other params as needed.
