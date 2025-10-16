"""Gym wrapper that adds a CLIP-based cosine-similarity reward term per step.

Usage: Wrap individual (non-vectorized) envs. For SubprocVecEnv each worker
will import and instantiate the wrapper separately, which keeps models local
to each subprocess.
"""
from typing import Any, Dict, Optional
import numpy as np
import torch
import gym
from PIL import Image


class ClipRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, text_embeds: Any, clip_model: Optional[Any] = None,
                 preprocess: Optional[Any] = None, w_clip: float = 1.2, device: str = 'cpu',
                 model_name: Optional[str] = None, use_last_frame: bool = True):
        super().__init__(env)
        self.w_clip = float(w_clip)
        self.device = torch.device(device)
        self.use_last_frame = use_last_frame

        # Load text embeddings: accept path to npz, dict, or numpy array
        if isinstance(text_embeds, (str,)):
            data = np.load(text_embeds)
            # keys may be strings '0','1',... or action names
            embeds = {}
            for k in data.files:
                try:
                    embeds[int(k)] = torch.tensor(data[k], dtype=torch.float32, device=self.device)
                except Exception:
                    # non-integer keys -> keep as raw
                    embeds[k] = torch.tensor(data[k], dtype=torch.float32, device=self.device)
            self.text_embeds = {k: v / v.norm() for k, v in embeds.items()}
        elif isinstance(text_embeds, dict):
            self.text_embeds = {}
            for k, v in text_embeds.items():
                tk = int(k) if not isinstance(k, int) else k
                arr = np.asarray(v)
                t = torch.tensor(arr, dtype=torch.float32, device=self.device)
                self.text_embeds[tk] = t / t.norm()
        else:
            # assume numpy array of shape (n_actions, dim)
            arr = np.asarray(text_embeds)
            self.text_embeds = {i: torch.tensor(arr[i], dtype=torch.float32, device=self.device) / np.linalg.norm(arr[i]) for i in range(arr.shape[0])}

        # Load CLIP model if not provided
        self.clip_model = clip_model
        self.preprocess = preprocess
        if self.clip_model is None and model_name is not None:
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
                self.clip_model = model.to(self.device)
                self.preprocess = preprocess
            except Exception as e:
                raise RuntimeError(f'Failed to load open_clip model {model_name}: {e}')

        if self.clip_model is None or self.preprocess is None:
            raise ValueError('ClipRewardWrapper requires clip_model+preprocess or a model_name to load one')

        # Ensure model in eval mode
        try:
            self.clip_model.eval()
        except Exception:
            pass

    def step(self, action):
        obs, r_basic, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Extract an image from the observation
        img = self._obs_to_pil(obs)
        r_clip = 0.0
        try:
            with torch.no_grad():
                img_in = self.preprocess(img).unsqueeze(0).to(self.device)
                img_feat = self.clip_model.encode_image(img_in)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                # choose text embedding for action
                key = int(action) if isinstance(action, (int, float)) or (hasattr(action, '__int__')) else action
                if key not in self.text_embeds:
                    # fallback to modulo 3 mapping
                    key = int(key) % max(1, len(self.text_embeds))
                txt = self.text_embeds[key]
                # ensure normalized
                txt = txt / txt.norm()
                r_clip = float((img_feat @ txt.unsqueeze(-1)).squeeze().cpu().item())
        except Exception:
            # on any failure, treat r_clip as 0 to avoid breaking training
            r_clip = 0.0

        r = r_basic + self.w_clip * r_clip
        # expose r_clip for logging
        if isinstance(info, dict):
            info = dict(info)
            info['r_clip'] = float(r_clip)
        else:
            info = {'r_clip': float(r_clip)}

        return obs, r, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _obs_to_pil(self, obs):
        # obs can be dict or array; try common keys
        arr = None
        if isinstance(obs, dict):
            for key in ('rgb', 'image', 'screen'):
                if key in obs:
                    arr = obs[key]
                    break
            # fall back to raw observation
            if arr is None:
                # take 'observation' or first value
                arr = obs.get('observation') or next(iter(obs.values()))
        else:
            arr = obs

        a = np.asarray(arr)
        # if channel-first (C,H,W) move to H,W,C
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        # if stacked frames (T,H,W) -> pick last
        if a.ndim == 3 and a.shape[2] not in (1, 3) and (a.shape[0] > 3):
            # shape maybe (T,H,W)
            a = a[-1]
        # convert to uint8 image
        if a.dtype != np.uint8:
            # try scaling
            a = (255 * (a - a.min()) / max(1e-8, (a.max() - a.min()))).astype('uint8')
        try:
            if a.ndim == 2:
                pil = Image.fromarray(a, mode='L').convert('RGB')
            else:
                pil = Image.fromarray(a)
        except Exception:
            # fallback: create blank image
            pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
        return pil
