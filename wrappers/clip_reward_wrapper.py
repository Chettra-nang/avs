"""
ClipRewardWrapper

Gym wrapper that augments environment rewards with a CLIP-based similarity
between the current rendered image and a set of text embeddings (one per
meta-action). It expects a fine-tuned CLIP model directory (Hugging Face
format) and a JSON file with text->embedding vectors (saved by
`finetune_clip.py`).

Usage:
    from wrappers.clip_reward_wrapper import ClipRewardWrapper
    env = ClipRewardWrapper(env, model_dir='AVs/models/clip_finetuned', text_emb='AVs/models/clip_finetuned/text_embeddings.json', w_clip=1.2)

Notes:
- The wrapper uses env.render(mode='rgb_array') to obtain an RGB image. If
  your environment doesn't support rendering, adapt the wrapper to read
  observation frames and convert them to RGB before encoding.
"""

from typing import Optional, Dict
import json
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class ClipRewardWrapper:
    """Minimal compatibility wrapper for Gym-like environments.

    Methods:
    - reset()
    - step(action)

    It assumes the wrapped env exposes `render(mode='rgb_array')`.
    """

    def __init__(self, env, model_dir: str, text_emb: str, w_clip: float = 1.2, device: Optional[str] = None):
        self.env = env
        self.model_dir = Path(model_dir)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(self.model_dir)
        self.model = CLIPModel.from_pretrained(self.model_dir).to(self.device)
        # load text embeddings JSON {text: [floats]}
        p = Path(text_emb)
        if not p.exists():
            raise FileNotFoundError(text_emb)
        with open(p, 'r') as f:
            data = json.load(f)
        self.texts = list(data.keys())
        self.text_embeddings = np.stack([np.asarray(data[t], dtype=np.float32) for t in self.texts])
        # ensure normalized
        norms = np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        self.text_embeddings = (self.text_embeddings / (norms + 1e-12)).astype(np.float32)
        self.w_clip = float(w_clip)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _render_image(self):
        # Try to get rgb_array from env.render
        try:
            img = self.env.render(mode='rgb_array')
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img)
            else:
                pil = Image.fromarray(np.asarray(img))
            return pil.convert('RGB')
        except Exception:
            # fallback: no render available
            return None

    def _compute_clip_reward(self, pil_img: Image.Image, target_text_idx: Optional[int] = None):
        if pil_img is None:
            return 0.0
        inputs = self.processor(images=pil_img, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1).cpu().numpy()[0]
        # Compute dot product with all text embeddings => cosine similarity
        sims = float((self.text_embeddings @ img_feat).max()) if target_text_idx is None else float(self.text_embeddings[int(target_text_idx)] @ img_feat)
        return sims

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        pil = self._render_image()
        # If env supplies an 'intended_meta_action' in info, use it to pick text; otherwise take max similarity
        target_idx = None
        if isinstance(info, dict) and 'intended_meta_action' in info:
            target_idx = info['intended_meta_action']
        r_clip = self._compute_clip_reward(pil, target_idx)
        reward = reward + self.w_clip * r_clip
        info = dict(info) if info is not None else {}
        info['r_clip'] = r_clip
        return obs, reward, done, info

    # convenience attribute access
    def __getattr__(self, name):
        return getattr(self.env, name)
