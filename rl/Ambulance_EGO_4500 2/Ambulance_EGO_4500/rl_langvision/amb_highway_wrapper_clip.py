# rl_langvision/amb_highway_wrapper_clip.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any
from gymnasium import spaces

from .clip_embedder import CLIPImageEncoder
from .cached_embedder import CachedLLMEmbedder
from .language_embedder import FrozenTextEmbedder

class AmbulanceHighwayCLIPWrapper(gym.Wrapper):
    """Always emits {'clip': (512,), 'text': (D_text,)}."""
    def __init__(
        self,
        env: gym.Env,
        clip_enc: CLIPImageEncoder,
        text_embedder: Optional[FrozenTextEmbedder] = None,
        cached_llm: Optional[CachedLLMEmbedder] = None,
        clip_stride: int = 4,
    ):
        super().__init__(env)
        self.clip_enc = clip_enc
        self.text_embedder = text_embedder
        self.cached_llm = cached_llm
        self.clip_stride = max(1, int(clip_stride))
        self._t = 0

        clip_dim = 512
        text_dim = int(getattr(cached_llm, "dim", 384))
        self.observation_space = spaces.Dict({
            "clip": spaces.Box(low=-np.inf, high=np.inf, shape=(clip_dim,), dtype=np.float32),
            "text": spaces.Box(low=-np.inf, high=np.inf, shape=(text_dim,), dtype=np.float32),
        })

    def reset(self, **kwargs):
        self._t = 0
        _, info = self.env.reset(**kwargs)
        out = self._obs_from_env(info)
        assert isinstance(out, dict) and "clip" in out and "text" in out
        return out, info

    def step(self, action):
        _, r, terminated, truncated, info = self.env.step(action)
        self._t += 1
        out = self._obs_from_env(info)
        assert isinstance(out, dict) and "clip" in out and "text" in out
        return out, r, terminated, truncated, info

    def _obs_from_env(self, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        rgb = self.env.render()                          # 'rgb_array' mode
        clip_vec = self.clip_enc.encode_np_rgb(rgb)      # (512,)

        if self.cached_llm is not None:
            text_vec = self.cached_llm.get_constant()
        else:
            speed_mps = float(getattr(getattr(self.env, "vehicle", None), "speed", 0.0) or 0.0)
            scen = (getattr(self.env, "config", {}) or {}).get("scenario", "highway")
            cap = f"Ambulance with right-of-way in {scen}. speed={speed_mps:.1f} m/s. Keep corridor open."
            assert self.text_embedder is not None, "text_embedder required when cached_llm is None"
            text_vec = self.text_embedder.encode_one(cap)

        return {"clip": np.asarray(clip_vec, np.float32),
                "text": np.asarray(text_vec, np.float32)}
