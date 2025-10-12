# tools/eval_ppo_clip.py
from __future__ import annotations
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# from your package
from rl_langvision.amb_highway_wrapper_clip import AmbulanceHighwayCLIPWrapper
from rl_langvision.clip_embedder import CLIPImageEncoder
from rl_langvision.cached_embedder import CachedLLMEmbedder
from rl_langvision.language_embedder import FrozenTextEmbedder


def _pick_device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _make_eval_env(config: dict, seed: int):
    def _init():
        base_env = gym.make("highway-v0", render_mode="rgb_array", config=config.get("env", {}))
        try: base_env.reset(seed=seed)
        except TypeError: pass

        # vision encoder
        vcfg = config.get("vision", {}) or {}
        clip_enc = CLIPImageEncoder(vcfg.get("clip_model", "openai/clip-vit-base-patch32"),
                                    device=vcfg.get("device") or _pick_device())

        # text side: prefer cached embeddings
        tcfg = config.get("text", {}) or {}
        if tcfg.get("cached_llm_dir"):
            cached_llm = CachedLLMEmbedder(tcfg["cached_llm_dir"], dim=int(tcfg.get("cached_dim", 384)))
            text_emb = None
        else:
            text_emb = FrozenTextEmbedder(tcfg.get("local_model", "sentence-transformers/all-MiniLM-L6-v2"))
            cached_llm = None

        return AmbulanceHighwayCLIPWrapper(
            base_env, clip_enc, text_embedder=text_emb, cached_llm=cached_llm,
            clip_stride=int(config.get("wrapper", {}).get("clip_stride", 4)),
        )
    return DummyVecEnv([_init])


def evaluate(model_path: Path, vecnorm_path: Path, config: dict,
             n_episodes: int = 10, seed: int = 12345, make_video: bool = False, video_path: Path | None = None):
    eval_env_raw = _make_eval_env(config, seed)
    # restore normalization stats saved during training
    eval_env = VecNormalize.load(str(vecnorm_path), eval_env_raw)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(str(model_path), env=eval_env, device="auto")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=n_episodes, deterministic=True, render=False)
    print(f"[EVAL] episodes={n_episodes}  mean_reward={mean_r:.3f} ± {std_r:.3f}")

    if make_video:
        import imageio
        frames = []
        # run a single episode and record frames
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            frame = eval_env.get_attr("render", 0)()  # call underlying env.render()
            frames.append(frame)
        out = str(video_path or (model_path.parent / "eval_episode.mp4"))
        imageio.mimsave(out, frames, fps=15)
        print(f"[EVAL] saved video -> {out}")


if __name__ == "__main__":
    # ==== paths you want to evaluate ====
    seed = 0
    model_path  = ROOT / f"ppo_clip_vlm_seed{seed}.zip"                 # final model from training script
    vecnorm_path = ROOT / "runs" / "ppo_ambulance" / f"seed_{seed}" / "vecnorm.pkl"
    # You can also evaluate the best checkpoint:
    # model_path = ROOT / "runs" / "ppo_ambulance" / f"seed_{seed}" / "best" / "best_model.zip"

    CACHED_DIR = str(ROOT / "cached_llm")

    CONFIG = {
        "env": {
            "lanes_count": 4, "vehicles_count": 40, "duration": 50,
            "simulation_frequency": 15, "policy_frequency": 1,
            "offscreen_rendering": True, "render_agent": True, "show_trajectories": False,
            "high_speed_reward": 0.4, "right_lane_reward": 0.1, "lane_change_reward": 0.0,
            "reward_speed_range": [20, 30], "normalize_reward": True,
            "centering_position": [0.3, 0.5], "scaling": 5.5,
            "scenario": "highway_emergency_dense",
        },
        "vision": {"clip_model": "openai/clip-vit-base-patch32", "device": None},
        "text": {"cached_llm_dir": CACHED_DIR, "cached_dim": 384, "local_model": None},
        "policy": {"feat_dim": 512},
        "wrapper": {"clip_stride": 4},
    }

    evaluate(model_path, vecnorm_path, CONFIG, n_episodes=10, seed=12345,
             make_video=False, video_path=None)
