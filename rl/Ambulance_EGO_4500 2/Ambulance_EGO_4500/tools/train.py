# tools/train_ppo_clip_standalone.py
from __future__ import annotations
import os, sys, random, argparse, json, time
from pathlib import Path
from typing import Iterable, Dict, List, Optional

# ---- import root so rl_langvision/* is found ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Mac / headless / MPS stability ----
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import gymnasium as gym
import highway_env  # registers "highway-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# project modules
from rl_langvision.clip_embedder import CLIPImageEncoder
from rl_langvision.cached_embedder import CachedLLMEmbedder
from rl_langvision.language_embedder import FrozenTextEmbedder
from rl_langvision.amb_highway_wrapper_clip import AmbulanceHighwayCLIPWrapper
from rl_langvision.features_extractor_clip import CLIPLangExtractor
from rl_langvision.reward_wrappers import SafetySpeedRewardWrapper
import rl_langvision.yielding_traffic  # noqa: F401


# ---------- device & seeds ----------
def _pick_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        return "mps"
    return "cpu"

def set_global_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ---------- console episode printer ----------
class ConsolePrinter(BaseCallback):
    """Prints per-episode lines and rolling (avg30) metrics, saves best model."""
    def __init__(self, eval_env, save_dir: Path, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.history = []
        self.best_score = -float("inf")

    def _on_step(self):
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []
        for i, info in enumerate(infos):
            if i < len(dones) and dones[i]:
                ep_info = info.get("episode")
                if not ep_info: continue
                r, l = ep_info["r"], ep_info["l"]
                crashed = int(info.get("crashed", False))
                speed = float(info.get("speed_kph_mean", info.get("speed_kph", 0)))
                print(f"[EP] return={r:7.2f} | len={l:4d} | crash={crashed} | mean_speed={speed:6.1f}kph")
                self.history.append(dict(r=r, l=l, crash=crashed, speed=speed))
                if len(self.history) > 30: self.history.pop(0)

                if len(self.history) >= 5:
                    avg_r = np.mean([x["r"] for x in self.history])
                    avg_c = np.mean([x["crash"] for x in self.history])
                    avg_s = np.mean([x["speed"] for x in self.history])
                    print(f"[AVG-30] reward={avg_r:7.2f} | crash={avg_c*100:5.1f}% | speed={avg_s:6.1f}kph")
        return True

    def _on_rollout_end(self):
        val = evaluate_model(self.model, self.eval_env, episodes=10)
        score = val["return_mean"] - 3.0 * (val["collisions"] / 10)
        print(f"[EVAL] ret={val['return_mean']:.2f}, coll={val['collisions']}, speed={val['speed_kph_mean']:.1f}, score={score:.2f}")
        if score > self.best_score:
            self.best_score = score
            best_path = self.save_dir / "best_model.zip"
            self.model.save(best_path)
            with open(self.save_dir / "best_val.json", "w") as f: json.dump(val, f, indent=2)
            print(f"🌟 New best model saved → {best_path}")


# ---------- environment builders ----------
def _choose_scenario(cfg_env: dict, mode: str) -> dict:
    cfg = dict(cfg_env)
    key = {"train": "scenarios_train", "eval": "scenarios_val", "test": "scenarios_test"}[mode]
    if key in cfg and cfg[key]: cfg["scenario"] = random.choice(cfg[key])
    return cfg

def _make_env_thunk(config: dict, seed: int, mode: str):
    assert mode in {"train","eval","test"}
    def _init():
        cfg = _choose_scenario(config.get("env", {}), mode)
        if cfg.get("is_emergency", False):
            cfg["other_vehicles_type"] = "rl_langvision.yielding_traffic.YieldingIDM"
        base_env = gym.make("highway-v0", render_mode="rgb_array", config=cfg)
        try: base_env.reset(seed=seed)
        except TypeError: pass

        vcfg, tcfg = config.get("vision", {}), config.get("text", {})
        clip_enc = CLIPImageEncoder(vcfg.get("clip_model","openai/clip-vit-base-patch32"),
                                    device=vcfg.get("device") or _pick_device())
        text_emb, cached_llm = None, None
        if tcfg.get("cached_llm_dir"):
            cached_llm = CachedLLMEmbedder(tcfg["cached_llm_dir"], dim=int(tcfg.get("cached_dim",384)))
        else:
            text_emb = FrozenTextEmbedder(tcfg.get("local_model","sentence-transformers/all-MiniLM-L6-v2"))

        env = AmbulanceHighwayCLIPWrapper(base_env, clip_enc, text_embedder=text_emb, cached_llm=cached_llm)
        if (config.get("reward") or {}).get("mode","custom") == "custom":
            env = SafetySpeedRewardWrapper(env, cfg=config["reward"]["custom_cfg"])
        return env
    return _init

def make_vec_envs(config, n_envs, base_seed, mode):
    thunks = [_make_env_thunk(config, base_seed+i, mode) for i in range(max(1,n_envs))]
    venv = DummyVecEnv(thunks)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=(mode=="train"))
    return venv

def _sync_vecnorm_stats(train_env, eval_env):
    eval_env.obs_rms, eval_env.ret_rms = train_env.obs_rms, train_env.ret_rms
    eval_env.training, eval_env.norm_reward = False, False


# ---------- evaluation ----------
def evaluate_model(model, env, episodes=10, deterministic=True) -> Dict[str,float]:
    rets, speeds, coll = [], [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False; trunc = False; ret = 0.0
        while not (bool(done) or bool(trunc)):
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, trunc, infos = env.step(act)
            ret += float(np.asarray(r).mean())
            info0 = (infos[0] if isinstance(infos, (list, tuple)) and infos else infos) or {}
            if info0.get("crashed", False): coll += 1
            if "speed_kph" in info0: speeds.append(float(info0["speed_kph"]))
        rets.append(ret)
    return dict(
        return_mean=float(np.mean(rets)) if rets else 0.0,
        collisions=int(coll),
        speed_kph_mean=float(np.mean(speeds)) if speeds else 0.0,
    )



# ---------- training ----------
def train_one_seed(config, seed, total_timesteps, prefix):
    set_global_seeds(seed)
    device = _pick_device()
    run_dir = Path(config["log_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_envs(config, config["algo"]["n_envs"], seed, "train")
    eval_env = make_vec_envs(config, 1, seed+9999, "eval")
    _sync_vecnorm_stats(env, eval_env)

    policy_kwargs = dict(
        features_extractor_class=CLIPLangExtractor,
        features_extractor_kwargs=dict(features_dim=config["policy"]["feat_dim"]),
        net_arch=dict(pi=[256,256], vf=[256,256])
    )
    a = config["algo"]["ppo"]
    model = PPO("MultiInputPolicy", env, seed=seed, policy_kwargs=policy_kwargs,
                n_steps=a["n_steps"], batch_size=a["batch_size"], learning_rate=a["lr"],
                gamma=a["gamma"], clip_range=a["clip_range"], gae_lambda=a["gae_lambda"],
                ent_coef=a["ent_coef"], target_kl=a["target_kl"],
                tensorboard_log=config["log_dir"], device=device, verbose=0)

    cb = ConsolePrinter(eval_env=eval_env, save_dir=run_dir)
    print(f"\n🚑 [seed {seed}] Training on {device} for {total_timesteps:,} steps…")
    t0=time.time()
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"seed_{seed}", callback=cb)
    print(f"⏱ Done in {(time.time()-t0)/60:.1f} min")

    val = evaluate_model(model, eval_env, 10)
    json.dump(val, open(run_dir/"final_val.json","w"), indent=2)
    print("VAL:", val)
    env.close(); eval_env.close()


def train(cfg, seeds, steps, prefix):
    for s in seeds: train_one_seed(cfg, s, steps, prefix)


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seeds", nargs="*", type=int, default=[33])
    args = parser.parse_args()

    CACHED_DIR = str(ROOT / "cached_llm")
    CONFIG = {
        "log_dir": str(ROOT / "runs" / "ppo_ambulance_safety_speed"),
        "reward": {
            "mode": "custom",
            "custom_cfg": {
                "speed_min":22.0,"speed_max":32.0,"speed_w":1.2,
                "ttc_threshold":2.5,"ttc_w":0.6,
                "ahead_dist":40.0,"same_lane_only":True,
                "block_penalty_w":0.2,"clear_bonus":0.4,
                "collision_penalty":2.0,"clip_abs":1.0
            }
        },
        "env": {
            "lanes_count":4,"vehicles_count":30,"duration":60,
            "simulation_frequency":15,"policy_frequency":5,
            "offscreen_rendering":True,"render_agent":True,
            "scenarios_train":["highway_emergency_dense","highway_time_pressure"],
            "scenarios_val":["highway_emergency_moderate"],
            "scenarios_test":["highway_construction"],
            "is_emergency":False
        },
        "vision":{"clip_model":"openai/clip-vit-base-patch32","device":None},
        "text":{"cached_llm_dir":CACHED_DIR,"cached_dim":384,"local_model":None},
        "policy":{"feat_dim":512},
        "algo":{
            "n_envs":1,
            "ppo":{
                "n_steps":1536,"batch_size":512,"lr":1e-4,
                "gamma":0.99,"clip_range":0.2,"gae_lambda":0.95,
                "ent_coef":0.003,"target_kl":0.02
            }
        }
    }
    print("Using", _pick_device(), "device")
    train(CONFIG, seeds=args.seeds, steps=args.steps, prefix="ppo_clip_vlm")
    print("🏁 All done.")
