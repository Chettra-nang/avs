<!--
Repository-wide Copilot instructions for the AVs project.
Place this file at .github/copilot-instructions.md to give Copilot/chat agents consistent context
about the project's purpose, tech stack, data formats, and developer workflows.
-->

# AVs — Copilot / Agent Instructions (project-level)

Purpose
-------
This repository collects driving scenarios and trains RL agents for ambulance-aware driving policies. The main goals are:
- Collect multi-agent scenario data (grayscale frames + metadata) using `collecting_ambulance_data`.
- Extract and label image–text pairs for CLIP-based preference learning.
- Train RL policies offline (and optionally online) using precomputed CLIP features and standard RL libs (PPO, DQN, BC).

Tech stack
----------
- Language: Python 3.10+ (venv-based development)
- ML: PyTorch (training scripts live under `offline_rl/trainers`)
- Data storage: Parquet (pyarrow), JSONL metadata
- Image processing: Pillow
- Utilities: numpy, pandas, tqdm

Repo layout (important folders)
-------------------------------
- `AVs/` — top-level project folder (this repo root)
- `collecting_ambulance_data/` — data collection utilities and examples
- `highway_datacollection/` — DatasetStorageManager, encoders and storage helpers
  - `storage/encoders.py` — BinaryArrayEncoder: encodes numpy arrays to bytes using `np.save` (and stores shape/dtype)
- `AVs/data/` — datasets produced by the collector (contains per-scenario `_transitions.parquet` and `_meta.jsonl` files)
- `AVs/scripts/` — utility scripts (including `extract_frames_for_clip.py`)
- `offline_rl/trainers/` — trainer scripts (BC, PPO, DQN) which expect precomputed CLIP feature `.npz`

Key data format notes (essential for any code-generation)
-------------------------------------------------------
- Transitions are stored in Parquet files named `*_transitions.parquet`. Each row (transition) typically includes:
  - `episode_id` (str)
  - `step` (int)
  - `scenario` (str)
  - `agent_id` (int)
  - `grayscale_blob` (binary blob containing a numpy array saved with `np.save`)
  - `grayscale_shape` (list/tuple) and `grayscale_dtype` (string) — used to reshape/decode the blob
  - `action` (may be a per-agent array/list/ndarray; ambulance agent is at index 0)
  - `reward` (float)

- BinaryArrayEncoder.decode(blob, shape, dtype) must be used to decode `grayscale_blob` when available. Accept that pyarrow/pandas may give `memoryview` or `np.ndarray(uint8)` for blob — convert to bytes before decoding.

How to run the frame extractor (important)
-----------------------------------------
Run from project root (activate venv first). Example:

```bash
# activate venv (example)
source avs_venv/bin/activate

# install runtime deps if needed
pip install -r requirements.txt

# run extractor (path under AVs/)
python3 AVs/scripts/extract_frames_for_clip.py --input AVs/data/ambulance_dataset_diagnose --output AVs/data/clip_frames --max-per-scenario 500
```

Expected output: PNG images under the `--output` dir organized by scenario, and `frames.csv` containing columns: `image_path, episode_id, step, agent_id, action, reward, scenario`.

Common pitfalls
---------------
- Running `python3 scripts/...` from repo root will fail if the scripts live under `AVs/scripts` — prefer specifying the full relative path (`AVs/scripts/...`) or cd into `AVs/`.
- Ensure `pyarrow` is installed in the active venv (Parquet reading requires it).
- Action fields may be per-agent arrays; code that expects scalar truthiness will fail (``The truth value of an array is ambiguous``) — extract agent 0 or flatten safely.
- Blobs may appear as `memoryview` or `np.ndarray(uint8)` from pyarrow — convert to bytes before calling `np.load`.

Coding & style guidelines
-------------------------
- Prefer small, well-tested helper functions when dealing with binary-encoded data (decoding, casting, reshaping).
- Use explicit type conversions and checks; avoid relying on truthiness of numpy arrays.
- Keep the public API stable for dataset storage: `BinaryArrayEncoder.encode/decode` is the contract.
- Add docstrings for top-level functions and small examples in script module-level docstrings.
- Use `black` + `isort` and simple linting; keep changes minimal and targeted when editing existing files.

Testing and QA
--------------
- Add fast unit tests for `highway_datacollection/storage/encoders.py` to verify encode/decode for typical shapes and dtypes.
- Add an integration smoke test that runs `AVs/scripts/extract_frames_for_clip.py --max-per-scenario 2` on a tiny sample dataset.

Suggested Copilot/agent behavior
--------------------------------
- Always prefer to read `highway_datacollection/storage/encoders.py` and `collecting_ambulance_data/collection/ambulance_collector.py` before changing code that touches data I/O.
- When producing code that reads Parquet binary columns, include robust guards for `memoryview`, `np.ndarray(uint8)`, empty blobs, and missing shape/dtype metadata.
- For new features that change data layout (column names or binary formats), also update dataset index creation (`highway_datacollection/storage/manager.py`) and include a backward-compatible read path.

Example prompts for this project
-------------------------------
- "Create a script that reads `_transitions.parquet`, decodes `grayscale_blob` using `BinaryArrayEncoder`, resizes to 224×224 RGB, and writes PNGs plus `frames.csv`."
- "Add a small Flask labeling UI to review images under `AVs/data/clip_frames` and export `labels.csv` with `image_path,label` columns."
- "Implement a CLIP feature extraction script that loads a fine-tuned CLIP ViT-B/32, encodes images listed in `frames.csv`, normalizes embeddings, and writes a `clip_features.npz` for offline trainers."

Maintenance notes
-----------------
- Keep this file updated when the dataset layout changes or new collection modalities are added.
- Consider adding path-specific `.instructions.md` files under directories with special rules (e.g., `offline_rl/` or `collecting_ambulance_data/`).

Contact
-------
Project maintainer: see repository README or git history for committers. For urgent dataset questions, inspect `collecting_ambulance_data` and `highway_datacollection` first.
