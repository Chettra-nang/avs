<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# write me a comprehensive document for AI IDe to create it for me .

Below is a copy-paste friendly, comprehensive specification for an AI IDE to automatically generate a complete multi‑agent HighwayEnv data-collection system that logs all modalities at once (Kinematics, OccupancyGrid, Grayscale), plus derived TTC and language summaries, aligned with the project PDF requirements for multi-agent training, curriculum scenarios, and rich per-step metadata.[^1]

### Title

HighwayEnv Multi-Agent Data Collection System: All-Modal Logging Specification[^1]

### Objective

Create a reproducible simulator and logger that runs multi-agent highway driving scenarios and records synchronized observations across Kinematics, OccupancyGrid, and Grayscale modalities at each timestep for each controlled vehicle, along with derived metrics (TTC) and language summaries, to support both RL and LLM-planner training as described in the project PDF.[^1]

### Functional Requirements

- Multi-agent simulation with configurable number of controlled vehicles using HighwayEnv and Gymnasium.[^1]
- Observation modalities:
    - Kinematics (tabular V×F: presence, x, y, vx, vy, cos_h, sin_h), normalized, ego-relative where configured.[^1]
    - OccupancyGrid (W×H×F spatial encoding around ego).[^1]
    - Grayscale image stacks for vision-based probes where supported.[^1]
- Derived features at each step:
    - Time-to-Collision (TTC) from Kinematics.[^1]
    - Per-agent language summary describing lane, speed, gaps, TTC, density, and maneuver context for LLM planning datasets.[^1]
- Curriculum scenarios:
    - Free-flow, dense commuting, stop-and-go, aggressive neighbors, lane closure, time-budget episodes; each tagged in metadata and configured for vehicles_count, lanes_count, and duration.[^1]
- Synchronized multi-modal logging:
    - Collect all three modalities “at the same time” by stepping parallel envs with identical seeds and actions, storing aligned rows keyed by episode_id, step, and agent_id.[^1]
- Storage format:
    - Transitions in Parquet (or CSV fallback) with binary blobs for large arrays (Occupancy and Grayscale), and explicit shape fields for decoding.[^1]
    - Episode metadata as JSONL; global index.json summarizing all scenario files for easy loading.[^1]


### Non-Functional Requirements

- Determinism: identical seeds per episode across modalities to guarantee synchronized trajectories.[^1]
- Scalability: configurable agents, steps per episode, and episodes per scenario; efficient binary storage for large tensors.[^1]
- Extensibility: plug-in policy hook to replace random actions with trained agents; modality toggles per scenario.[^1]


### Directory Layout

- dataset_highway/
    - free_flow/ … lane_closure/ … time_budget/ with per-episode transitions.parquet and meta.jsonl.[^1]
    - index.json listing all scenarios and their file paths.[^1]


### Installation Script

```bash
pip install highway-env gymnasium stable-baselines3 torch numpy pandas matplotlib pyarrow
```


### AI IDE Tasks (Step-by-Step)

1) Generate constants and scenario registry

- Define DATA_ROOT and SCENARIOS dict with the six curriculum scenarios and their counts/lanes/durations to match the PDF objectives of varied densities and constraints.[^1]

2) Implement multi-agent config builder

- Build make_multiagent_config with:
    - controlled_vehicles ≥ 2
    - observation: MultiAgentObservation with observation_config type selectable among Kinematics, OccupancyGrid, and GrayscaleObservation
    - action: MultiAgentAction using DiscreteMetaAction (0..4 for lane and speed meta-commands)
    - reward parameters: collision_reward, high_speed_reward, right_lane_reward, lane_change_reward, normalize_reward (logged as emitted)
This matches the multi-agent control and observation requirements in the PDF.[^1]

3) Create per-scenario env factory

- make_env_for_scenario(scenario_name, n_agents, obs_type, normalize) returns a gymnasium HighwayEnv configured for the given scenario and modality, ensuring reproducibility via seed parameter during reset.[^1]

4) Implement state summarization utilities

- estimate_lane from lateral y position, find_lead_vehicle in same lane, lane_gap for left/right lanes, TTC from relative speed and gap, traffic_density from presence flags, and summarize_text producing compact natural language strings per agent per step for LLM conditioning per the project approach.[^1]

5) Implement writers and schema

- new_episode_paths creates unique per-episode file targets; to_parquet with pyarrow fallback to CSV; append_jsonl for metadata; ensure each record includes episode_id, step, agent_id, scenario, and config pointers.[^1]

6) Implement “all-modal” synchronized collector

- collect_all_modalities(scenario_name, episodes, n_agents, seed):
    - Construct three parallel envs: obs_type in {"Kinematics","OccupancyGrid","GrayscaleObservation"} with identical base config and seed for sync.[^1]
    - For each step:
        - Sample a joint multi-agent DiscreteMetaAction (replace with a policy if available)
        - Step all three envs with the same action tuple
        - Record per-agent:
            - Kinematics ego row fields (presence, x, y, vx, vy, cos_h, sin_h)
            - Derived TTC and LLM summary text (compact)
            - OccupancyGrid and Grayscale arrays as binary blobs with dtype and shape metadata (e.g., occ_blob, occ_shape; gray_blob, gray_shape)
        - Stop when any env signals done or truncated; also cap by max_steps if needed
    - Write transitions parquet and metadata JSONL for the episode batch and return file paths.[^1]

7) Implement per-scenario and full-curriculum runners

- run_full_collection(episodes_per_scenario, n_agents, seed) loops over SCENARIOS, optionally sets time_budget for the time_budget scenario, and calls collect_all_modalities, then writes index.json listing all artifacts for downstream loaders.[^1]

8) Optional: modality-specific or mixed-by-scenario runner

- Provide run_mixed_collection mapping each scenario to a specific obs_type for single-modality runs, then concatenate later if needed, ensuring provenance via obs_type column or separate folders per scenario.[^1]


### Complete Reference Code (single file)

Paste the following into main.py and run to produce the dataset (edit episodes_per_scenario/n_agents as desired).[^1]

```python
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from pathlib import Path
import json, time, uuid
from typing import Dict, Any, List, Tuple

# ---- Storage root ----
DATA_ROOT = Path("./dataset_highway")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Scenarios (curriculum) ----
SCENARIOS = {
    "free_flow": dict(vehicles_count=30, lanes_count=4, duration=35),
    "dense_commuting": dict(vehicles_count=60, lanes_count=4, duration=45),
    "stop_and_go": dict(vehicles_count=50, lanes_count=4, duration=45),
    "aggressive_neighbors": dict(vehicles_count=45, lanes_count=4, duration=45),
    "lane_closure": dict(vehicles_count=45, lanes_count=3, duration=50),
    "time_budget": dict(vehicles_count=40, lanes_count=4, duration=40),
}

# ---- Multi-agent config ----
def make_multiagent_config(
    n_agents:int=3,
    vehicles_count:int=50,
    lanes_count:int=4,
    duration:int=40,
    obs_type:str="Kinematics",
    normalize:bool=True
) -> Dict[str, Any]:
    return {
        "controlled_vehicles": n_agents,
        "lanes_count": lanes_count,
        "vehicles_count": vehicles_count,
        "duration": duration,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": obs_type,
                "vehicles_count": 15,
                "features": ["presence","x","y","vx","vy","cos_h","sin_h"] if obs_type=="Kinematics" else None,
                "absolute": False,
                "normalize": normalize
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"}
        },
        "collision_reward": -1.0,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
        "lane_change_reward": 0.0,
        "normalize_reward": True
    }

def episode_writer_root(scenario_name:str) -> Path:
    root = DATA_ROOT / scenario_name
    root.mkdir(parents=True, exist_ok=True)
    return root

def new_episode_paths(scenario_name:str) -> Dict[str, Path]:
    ep_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    root = episode_writer_root(scenario_name)
    return {
        "transitions": root / f"{ep_id}_transitions.parquet",
        "metadata":    root / f"{ep_id}_meta.jsonl"
    }

def to_parquet(df: pd.DataFrame, path: Path):
    try:
        import pyarrow  # noqa
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)

def append_jsonl(records: List[Dict[str, Any]], path: Path):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def make_env_for_scenario(scenario_name:str, n_agents:int=3, obs_type:str="Kinematics", normalize:bool=True):
    sc = SCENARIOS[scenario_name]
    cfg = make_multiagent_config(
        n_agents=n_agents,
        vehicles_count=sc["vehicles_count"],
        lanes_count=sc["lanes_count"],
        duration=sc["duration"],
        obs_type=obs_type,
        normalize=normalize
    )
    env = gym.make("highway-v0", config=cfg)
    return env, cfg

# ---- Summarizer utilities ----
def estimate_lane(y_pos: float, lane_width: float=4.0, lanes:int=4) -> int:
    lane = int((y_pos + lane_width*lanes/2) // lane_width) + 1
    return max(1, min(lanes, lane))

def find_lead_vehicle(ego, others) -> Tuple[np.ndarray, float, float]:
    ego_lane = estimate_lane(ego[^2])
    lead, min_dx = None, np.inf
    for v in others:
        if v[^0] < 0.5:
            continue
        if estimate_lane(v[^2]) == ego_lane and (v[^1] > ego[^1]):
            dx = v[^1] - ego[^1]
            if dx < min_dx:
                min_dx, lead = dx, v
    if lead is None:
        return None, 1e3, 0.0
    rel_speed_kmh = (lead[^3]-ego[^3]) * 3.6
    return lead, float(min_dx), float(rel_speed_kmh)

def lane_gap(ego, others, direction:str, lanes:int=4) -> float:
    ego_lane = estimate_lane(ego[^2], lanes=lanes)
    target = ego_lane + (1 if direction=="left" else -1)
    if target < 1 or target > lanes:
        return 0.0
    front_gap, back_gap = 1e3, 1e3
    for v in others:
        if v[^0] < 0.5:
            continue
        if estimate_lane(v[^2], lanes=lanes) == target:
            dx = v[^1] - ego[^1]
            if dx >= 0: front_gap = min(front_gap, dx)
            else: back_gap = min(back_gap, abs(dx))
    return float(min(front_gap, back_gap))

def time_to_collision(ego, lead_gap_m: float, lead_rel_speed_kmh: float) -> float:
    rel_speed_ms = (lead_rel_speed_kmh)/3.6
    if rel_speed_ms >= 0: return 10.0
    if lead_gap_m <= 0.1: return 0.1
    return float(max(0.1, min(10.0, lead_gap_m/(-rel_speed_ms))))

def traffic_density(others) -> float:
    present = np.sum(others[:,0] > 0.5) if len(others)>0 else 0
    return float(present / max(1, len(others))) if len(others)>0 else 0.0

def summarize_text(ego, others, lanes:int=4, speed_limit_kmh:int=120) -> str:
    ego_speed = np.sqrt(ego[^3]**2 + ego[^4]**2) * 3.6
    lane_idx = estimate_lane(ego[^2], lanes=lanes)
    _, gap_m, rel_kmh = find_lead_vehicle(ego, others)
    ttc = time_to_collision(ego, gap_m, rel_kmh)
    left_g = lane_gap(ego, others, "left", lanes=lanes)
    right_g = lane_gap(ego, others, "right", lanes=lanes)
    dens = traffic_density(others)
    dens_txt = "high" if dens>0.7 else "medium" if dens>0.3 else "low"
    right_status = "available" if right_g>0 else "not available"
    return (f"Ego lane={lane_idx}/{lanes}, speed={ego_speed:.0f} km/h (limit={speed_limit_kmh}). "
            f"Lead gap={gap_m:.0f} m, rel_speed={rel_kmh:.0f} km/h, TTC={ttc:.1f} s. "
            f"Left gap={left_g:.0f} m. Right lane {right_status}. Traffic density={dens_txt}.")

# ---- All-modal synchronized collector ----
def make_env_with_obs(obs_type:str, base_cfg:dict):
    cfg = dict(base_cfg)
    cfg["observation"] = {
        "type": "MultiAgentObservation",
        "observation_config": {"type": obs_type,
                               "vehicles_count": 15,
                               "absolute": False,
                               "normalize": True}
    }
    return gym.make("highway-v0", config=cfg)

def collect_all_modalities(
    scenario_name:str,
    episodes:int=5,
    n_agents:int=3,
    seed:int=0,
    max_steps:int=1500
):
    env_base, base_cfg = make_env_for_scenario(scenario_name, n_agents=n_agents, obs_type="Kinematics", normalize=True)
    base_cfg = dict(base_cfg)

    env_kin = make_env_with_obs("Kinematics", base_cfg)
    env_occ = make_env_with_obs("OccupancyGrid", base_cfg)
    env_gray = make_env_with_obs("GrayscaleObservation", base_cfg)

    paths = new_episode_paths(scenario_name)
    all_rows, meta = [], []

    for ep in range(episodes):
        s = seed + ep
        obs_k, _ = env_kin.reset(seed=s)
        obs_o, _ = env_occ.reset(seed=s)
        obs_g, _ = env_gray.reset(seed=s)
        done = truncated = False
        step = 0
        ep_id = uuid.uuid4().hex

        meta.append({
            "episode_id": ep_id,
            "scenario": scenario_name,
            "config": base_cfg,
            "modalities": ["Kinematics","OccupancyGrid","Grayscale"]
        })

        while not (done or truncated):
            actions = tuple(np.random.randint(0, 5) for _ in range(n_agents))
            nobs_k, rew_k, d1, t1, info_k = env_kin.step(actions)
            nobs_o, rew_o, d2, t2, info_o = env_occ.step(actions)
            nobs_g, rew_g, d3, t3, info_g = env_gray.step(actions)

            done = d1 or d2 or d3
            truncated = t1 or t2 or t3

            for i in range(n_agents):
                obk = obs_k[i]
                ego = obk[^0] if obk.ndim==2 else obk
                others = obk[1:] if obk.ndim==2 else np.zeros((0, len(ego)))

                row = {
                    "episode_id": ep_id,
                    "step": step,
                    "agent_id": i,
                    "action": int(actions[i]),
                    "reward": float(rew_k[i] if isinstance(rew_k,(list,tuple,np.ndarray)) else rew_k),
                    # Kinematics (ego)
                    "kin_presence": float(ego[^0]),
                    "kin_x": float(ego[^1]), "kin_y": float(ego[^2]),
                    "kin_vx": float(ego[^3]), "kin_vy": float(ego[^4]),
                    "kin_cos_h": float(ego[^5]), "kin_sin_h": float(ego[^6]),
                    # Derived language
                    "summary_text": summarize_text(ego, others, lanes=base_cfg["lanes_count"]),
                }

                # OccupancyGrid
                occ_i = np.array(obs_o[i])
                row["occ_shape"] = list(occ_i.shape)
                row["occ_dtype"] = str(occ_i.dtype)
                row["occ_blob"] = occ_i.astype(np.float32).tobytes()

                # Grayscale
                gray_i = np.array(obs_g[i])
                row["gray_shape"] = list(gray_i.shape)
                row["gray_dtype"] = str(gray_i.dtype)
                row["gray_blob"] = gray_i.astype(np.uint8).tobytes()

                all_rows.append(row)

            obs_k, obs_o, obs_g = nobs_k, nobs_o, nobs_g
            step += 1
            if step >= max_steps: truncated = True

    df = pd.DataFrame(all_rows)
    to_parquet(df, paths["transitions"])
    append_jsonl(meta, paths["metadata"])
    return paths

def run_full_collection(episodes_per_scenario:int=10, n_agents:int=3, seed:int=0):
    index = []
    for name in SCENARIOS.keys():
        p = collect_all_modalities(
            scenario_name=name,
            episodes=episodes_per_scenario,
            n_agents=n_agents,
            seed=seed,
            max_steps=1500
        )
        index.append({
            "scenario": name,
            "transitions_path": str(p["transitions"]),
            "metadata_path": str(p["metadata"]),
            "modalities": ["Kinematics","OccupancyGrid","Grayscale"]
        })
    (DATA_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index

if __name__ == "__main__":
    idx = run_full_collection(episodes_per_scenario=5, n_agents=3, seed=0)
    print("Dataset index:", json.dumps(idx, indent=2))
```


### Usage Notes

- To reduce file size, consider downsampling Grayscale or storing only every Nth frame while keeping Kinematics at every step for control fidelity.[^1]
- To switch to single-modality collection, call collect_episodes from the earlier script instead, or modify collect_all_modalities to include/exclude modalities via flags.[^1]
- Replace the random action sampler with a trained policy to bias the dataset toward higher-value trajectories for imitation or offline RL.[^1]

This document gives an AI IDE all instructions and code needed to generate a synchronized multi-modal dataset across the curriculum scenarios, fully aligned with the project PDF’s data collection requirements for multi-agent autonomous driving research.[^1]

<div style="text-align: center">⁂</div>

[^1]: https://highway-env.farama.org/multi_agent/

