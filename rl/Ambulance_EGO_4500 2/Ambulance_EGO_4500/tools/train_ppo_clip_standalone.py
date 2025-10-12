# tools/train_ppo_clip_standalone.py
from __future__ import annotations
import os, sys, random, argparse, json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

# ---- import root so rl_langvision/* is found ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gymnasium as gym
import highway_env  # registers "highway-v0"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
)
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# project modules
from rl_langvision.clip_embedder import CLIPImageEncoder
from rl_langvision.cached_embedder import CachedLLMEmbedder
from rl_langvision.language_embedder import FrozenTextEmbedder
from rl_langvision.amb_highway_wrapper_clip import AmbulanceHighwayCLIPWrapper
from rl_langvision.features_extractor_clip import CLIPLangExtractor
from rl_langvision.reward_wrappers import SafetySpeedRewardWrapper

# registers "rl_langvision.yielding_traffic.YieldingIDM"
import rl_langvision.yielding_traffic  # noqa: F401


# ---------- utils ----------
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("high")  # M2 stability
        return "mps"
    return "cpu"

def set_global_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- TB logger for extra info ----------
class InfoKeysLogger(BaseCallback):
    """Logs selected info[] keys (emitted by SafetySpeedRewardWrapper) to TensorBoard."""
    def __init__(self, keys=("blockers_ahead","ttc_min","r_speed","r_clear","r_block","r_ttc_pen"), verbose=0):
        super().__init__(verbose)
        self.keys = keys
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        agg: Dict[str, list] = {k: [] for k in self.keys}
        for d in infos:
            for k in self.keys:
                if k in d:
                    agg[k].append(d[k])
        for k, vals in agg.items():
            if vals:
                self.logger.record_mean(f"info/{k}", float(np.mean(vals)))
        return True


# ---------- environment builders ----------
def _choose_scenario(cfg_env: dict, mode: str) -> dict:
    """Pick a scenario from split lists (train/val/test) and return a shallow-copied cfg."""
    cfg = dict(cfg_env)
    key = {
        "train": "scenarios_train",
        "eval":  "scenarios_val",
        "test":  "scenarios_test",
    }[mode]
    scenarios: Optional[List[str]] = cfg.get(key)
    if scenarios:
        cfg["scenario"] = random.choice(scenarios)
    return cfg

def _make_env_thunk(config: dict, seed: int, mode: str):
    """
    Builds highway env, swaps in YieldingIDM when emergency mode is on,
    wraps with CLIP + text encoders, then (optionally) applies SafetySpeedRewardWrapper.
    """
    assert mode in {"train","eval","test"}
    def _init():
        # --- base config (pick scenario according to split) ---
        cfg = _choose_scenario(config.get("env", {}) or {}, mode)

        # Make other vehicles yield to the ambulance (scripted, deterministic).
        if cfg.get("is_emergency", False):
            cfg["other_vehicles_type"] = "rl_langvision.yielding_traffic.YieldingIDM"
            if "yield_radius" in cfg:      cfg["yield_radius"] = float(cfg["yield_radius"])
            if "yield_right_bias" in cfg:  cfg["yield_right_bias"] = float(cfg["yield_right_bias"])

        base_env = gym.make("highway-v0", render_mode="rgb_array", config=cfg)

        # reproducible seeding
        try: base_env.reset(seed=seed)
        except TypeError: pass
        try: base_env.action_space.seed(seed); base_env.observation_space.seed(seed)
        except Exception: pass

        # ---- encoders (vision + text) ----
        vcfg = config.get("vision", {}) or {}
        clip_enc = CLIPImageEncoder(
            vcfg.get("clip_model", "openai/clip-vit-base-patch32"),
            device=vcfg.get("device") or _pick_device(),
        )
        tcfg = config.get("text", {}) or {}
        if tcfg.get("cached_llm_dir"):
            cached_llm = CachedLLMEmbedder(tcfg["cached_llm_dir"], dim=int(tcfg.get("cached_dim", 384)))
            text_emb = None
        else:
            text_emb = FrozenTextEmbedder(tcfg.get("local_model", "sentence-transformers/all-MiniLM-L6-v2"))
            cached_llm = None

        env = AmbulanceHighwayCLIPWrapper(
            base_env,
            clip_enc,
            text_embedder=text_emb,
            cached_llm=cached_llm,
            clip_stride=int(config.get("wrapper", {}).get("clip_stride", 4)),
        )

        # ---- reward choice ----
        reward_mode = (config.get("reward") or {}).get("mode", "custom")  # "custom" or "stock"
        if reward_mode == "custom":
            rs_cfg = (config.get("reward") or {}).get("custom_cfg", {})
            env = SafetySpeedRewardWrapper(env, cfg=rs_cfg)
        # else: use stock highway-env reward weights already in cfg

        return env
    return _init

def make_vec_envs(config: dict, n_envs: int, base_seed: int, mode: str):
    thunks = [_make_env_thunk(config, base_seed + i, mode) for i in range(max(1, n_envs))]
    venv = SubprocVecEnv(thunks) if (n_envs > 1 and mode=="train") else DummyVecEnv(thunks)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, training=(mode=="train"))
    return venv

def _sync_vecnorm_stats(train_env: VecNormalize, eval_env: VecNormalize):
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.training = False
    eval_env.norm_reward = False


# ---------- evaluation ----------
def evaluate_model(model: PPO, env: VecNormalize, episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
    """Run evaluation and compute simple metrics."""
    returns = []
    collisions = 0
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False; truncated = False
        total_r = 0.0; steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, truncated, info = env.step(action)
            total_r += float(r); steps += 1
            # collision info may be in 'crashed' or in wrapper info
            if isinstance(info, (list, tuple)) and info:
                inf = info[0]
            else:
                inf = info
            if inf.get("crashed", False):
                collisions += 1
        returns.append(total_r); lengths.append(steps)

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "len_mean": float(np.mean(lengths)),
        "collisions": int(collisions),
    }


# ---------- training ----------
def train_one_seed(config: dict, seed: int, total_timesteps: int, model_prefix: str):
    set_global_seeds(seed)
    n_envs = int((config.get("algo") or {}).get("n_envs", 1))

    # directories
    tb_root = Path(config.get("log_dir", ROOT / "runs" / "ppo_ambulance"))
    run_dir = tb_root / f"seed_{seed}"
    ckpt_dir = run_dir / "ckpt"
    best_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # envs
    env = make_vec_envs(config, n_envs, base_seed=seed, mode="train")
    eval_env = make_vec_envs(config, 1, base_seed=seed + 10_000, mode="eval")
    _sync_vecnorm_stats(env, eval_env)

    # policy (CLIP features extractor)
    policy_kwargs = dict(
        features_extractor_class=CLIPLangExtractor,
        features_extractor_kwargs=dict(features_dim=(config.get("policy") or {}).get("feat_dim", 512)),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    a = (config.get("algo") or {}).get("ppo", {}) or {}
    model = PPO(
        "MultiInputPolicy",
        env,
        seed=seed,
        policy_kwargs=policy_kwargs,
        n_steps=a.get("n_steps", 2048),
        batch_size=a.get("batch_size", 512),
        learning_rate=a.get("lr", 2.5e-4),
        gamma=a.get("gamma", 0.99),
        clip_range=a.get("clip_range", 0.2),
        gae_lambda=a.get("gae_lambda", 0.95),
        ent_coef=a.get("ent_coef", 0.005),
        target_kl=a.get("target_kl", 0.02),
        verbose=1,
        tensorboard_log=str(tb_root),
    )

    # callbacks
    info_cb = InfoKeysLogger()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(1, (10_000 // max(1, n_envs))),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, (50_000 // max(1, n_envs))),
        save_path=str(ckpt_dir),
        name_prefix="ppo",
    )

    print(f"[seed {seed}] starting learn() for {total_timesteps} steps …")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"seed_{seed}",
        callback=[info_cb, eval_cb, ckpt_cb],
    )

    # save policy + VecNormalize stats
    model_path = Path(config.get("output_dir", ROOT)) / f"{model_prefix}_seed{seed}.zip"
    model.save(str(model_path))
    env.save(str(run_dir / "vecnorm.pkl"))
    print(f"[seed {seed}] saved model -> {model_path}")

    # final VAL evaluation (uses scenarios_val)
    val_metrics = evaluate_model(model, eval_env, episodes=10, deterministic=True)
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"[seed {seed}] VAL metrics:", val_metrics)

    # TEST evaluation (held-out; uses scenarios_test)
    test_env = make_vec_envs(config, 1, base_seed=seed + 20_000, mode="test")
    _sync_vecnorm_stats(env, test_env)
    test_metrics = evaluate_model(model, test_env, episodes=10, deterministic=True)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"[seed {seed}] TEST metrics:", test_metrics)

    # cleanup
    try:
        env.close(); eval_env.close(); test_env.close()
    except Exception:
        pass


def train(config: dict, seeds: Iterable[int] | None = None,
          total_timesteps: int = 1_000_000, model_prefix: str = "ppo_clip_vlm"):
    seeds = list(seeds) if seeds is not None else [0]
    for s in seeds:
        train_one_seed(config, seed=s, total_timesteps=total_timesteps, model_prefix=model_prefix)


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--logdir", type=str, default=str(ROOT / "runs" / "ppo_ambulance"))
    parser.add_argument("--outdir", type=str, default=str(ROOT))
    parser.add_argument("--reward", choices=["custom","stock"], default="custom",
                        help="custom=SafetySpeedRewardWrapper; stock=HighwayEnv reward weights")
    args = parser.parse_args()

    CACHED_DIR = str(ROOT / "cached_llm")

    # ------- CONFIG -------
    CONFIG = {
        "log_dir": args.logdir,
        "output_dir": args.outdir,

        # Which reward to use
        "reward": {
            "mode": args.reward,  # "custom" or "stock"
            "custom_cfg": {       # ONLY used when mode == "custom"
                "speed_min": 20.0, "speed_max": 30.0,
                "ahead_dist": 35.0, "same_lane_only": True,
                "ttc_threshold": 2.0,
                "ttc_w": 0.30, "speed_w": 1.00,
                "clear_bonus": 0.30, "block_penalty_w": 0.15,
                "collision_penalty": 1.0, "clip_abs": 1.0,
            }
        },

        "env": {
            # core sim
            "lanes_count": 4, "vehicles_count": 40, "duration": 50,
            "simulation_frequency": 15, "policy_frequency": 1,
            "offscreen_rendering": True, "render_agent": True, "show_trajectories": False,

            # STOCK reward weights (used if reward.mode == "stock")
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "lane_change_reward": 0.0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,

            "centering_position": [0.3, 0.5],
            "scaling": 5.5,

            # Splits (we choose one per episode from lists below)
            "scenarios_train": [
                "highway_emergency_dense", "highway_time_pressure",
                "highway_merge_heavy", "highway_rush_hour"
            ],
            "scenarios_val": [
                "highway_emergency_moderate", "highway_stop_and_go"
            ],
            "scenarios_test": [
                "highway_construction", "highway_lane_closure", "highway_accident_scene"
            ],

            # Scripted yielding (non-ego cars open the road)
            "is_emergency": True,
            "yield_radius": 60.0,
            "yield_right_bias": 0.7,
        },

        "vision": {"clip_model": "openai/clip-vit-base-patch32", "device": None},  # auto-pick
        "text": {"cached_llm_dir": CACHED_DIR, "cached_dim": 384, "local_model": None},
        "policy": {"feat_dim": 512},
        "wrapper": {"clip_stride": 4},

        "algo": {
            "n_envs": 1,
            "ppo": {
                "n_steps": 2048, "batch_size": 512, "lr": 2.5e-4,
                "gamma": 0.99, "clip_range": 0.2, "gae_lambda": 0.95,
                "ent_coef": 0.005, "target_kl": 0.02
            },
        },
    }

    # If you insist on the wrapper-only reward, disable stock weights’ influence:
    if CONFIG["reward"]["mode"] == "custom":
        CONFIG["env"].update({
            "high_speed_reward": 0.0,
            "right_lane_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
        })

    # ------- GO -------
    train(CONFIG, seeds=args.seeds, total_timesteps=args.steps, model_prefix="ppo_clip_vlm")
