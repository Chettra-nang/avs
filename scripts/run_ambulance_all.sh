#!/usr/bin/env bash
set -euo pipefail

# Orchestration script for ambulance -> CLIP -> DQN smoke pipeline
# Usage: ./scripts/run_ambulance_all.sh [--full]
# By default runs a small smoke collection and training locally (CPU). Use --full to set larger jobs (recommended on RTX5090).

FULL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift;;
    -h|--help) echo "Usage: $0 [--full]"; exit 0;;
    *) echo "Unknown arg $1"; exit 1;;
  esac
done

LOGDIR=logs
mkdir -p "$LOGDIR"

# Configurable params
SCENARIOS=(highway_emergency_moderate merge_highway_entry intersection_four_way)
OUT_ROOT=data/ambulance_parallel/three_types
CLIP_DATA_ROOT=data/highway_multimodal_dataset
CLIP_OUTPUT=clip_finetune/outputs_ambulance

if [[ $FULL -eq 1 ]]; then
  EPISODES=1000
  MAX_STEPS=200
  BATCH_WORKERS=6
  EPOCHS=12
  BATCH=32
  DEVICE_ARG="--device cuda"
else
  EPISODES=5
  MAX_STEPS=50
  BATCH_WORKERS=2
  EPOCHS=2
  BATCH=8
  DEVICE_ARG="--device cpu"
fi

# 1) Parallel collection
echo "Starting parallel ambulance collection (scenarios: ${SCENARIOS[*]})"
python collecting_ambulance_data/examples/parallel_ambulance_collection.py \
  --scenarios ${SCENARIOS[*]} \
  --episodes $EPISODES \
  --max-steps $MAX_STEPS \
  --output-dir $OUT_ROOT \
  --seed 20251017 \
  --n-agents 4 \
  --max-workers $BATCH_WORKERS \
  --gpu \
  --batch-optimize \
  > "$LOGDIR/collect_ambulance.log" 2>&1

# 2) Consolidate to CLIP dataset
echo "Consolidating collector outputs to CLIP dataset layout"
python scripts/collect_ambulance_to_clip_dataset.py \
  --collector-root $OUT_ROOT \
  --out-dir $CLIP_DATA_ROOT \
  --force-copy \
  > "$LOGDIR/consolidate_to_clip.log" 2>&1

# 3) CLIP finetune (smoke or full)
mkdir -p "$CLIP_OUTPUT"
CAPTIONS_FILE="$CLIP_DATA_ROOT/texts/${SCENARIOS[0]}.csv"
# Note: finetune script currently takes a single captions file. For full runs, combine CSVs into one all_captions.csv

echo "Starting CLIP finetune (smoke)"
python clip_finetune/finetune_clip.py \
  --data-dir $CLIP_DATA_ROOT/frames \
  --captions $CAPTIONS_FILE \
  --output $CLIP_OUTPUT \
  --model roberta-ViT-B-32 \
  --epochs $EPOCHS \
  --batch-size $BATCH \
  > "$LOGDIR/finetune_clip.log" 2>&1

# 4) Quick DQN smoke run with clip shaping
echo "Starting DQN smoke training with CLIP shaping"
python scripts/train_dqn.py \
  --env-id highway-fast-v0 \
  --timesteps 2000 \
  --use-clip \
  --clip-embeds $CLIP_OUTPUT/action_embeds.npz \
  --w-clip 1.2 \
  --clip-model-name roberta-ViT-B-32 \
  --device cpu \
  > "$LOGDIR/train_dqn_ambulance_smoke.log" 2>&1 &

echo "Run complete. Logs under $LOGDIR. CLIP outputs under $CLIP_OUTPUT"
