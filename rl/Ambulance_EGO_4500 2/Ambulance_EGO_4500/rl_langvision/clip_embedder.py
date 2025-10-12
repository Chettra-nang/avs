# rl_langvision/clip_embedder.py
from __future__ import annotations
import numpy as np
import torch
import open_clip
from PIL import Image

def _pick_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        return "mps"
    return "cpu"

class CLIPImageEncoder:
    """
    Lightweight image → CLIP ViT embedding (L2-normalized).
    Default: ViT-B/32 -> 512-d.
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
        self.device = device or _pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def encode_np_rgb(self, rgb: np.ndarray) -> np.ndarray:
        # rgb: (H,W,3) uint8
        img = Image.fromarray(rgb)
        x = self.preprocess(img).unsqueeze(0).to(self.device)  # 1x3x224x224
        z = self.model.encode_image(x)                         # 1x512
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (512,)





#
# 
# 
#  from __future__ import annotations
# import numpy as np
# import torch
# from PIL import Image

# # We try open_clip first (fast, common in RL repos); fall back to transformers if missing.
# try:
#     import open_clip
#     _HAVE_OPENCLIP = True
# except Exception:
#     _HAVE_OPENCLIP = False
#     from transformers import CLIPModel, CLIPProcessor  # type: ignore


# class CLIPImageEncoder:
#     """
#     Tiny wrapper around CLIP that exposes:
#         - .dim          -> embedding dimension (int)
#         - .device       -> torch device string
#         - .encode(np_rgb) -> (dim,) float32 L2-normalized vector

#     Accepts numpy HxWx3 uint8 arrays or PIL.Image.
#     """
#     def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else
#                                  "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
#                                  else "cpu")

#         if _HAVE_OPENCLIP:
#             # Map some common HF names to open_clip naming
#             name_map = {
#                 "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
#                 "openai/clip-vit-base-patch16": ("ViT-B-16", "openai"),
#             }
#             oc_name, oc_pre = name_map.get(model_name, ("ViT-B-32", "openai"))
#             self.model, _, self.preprocess = open_clip.create_model_and_transforms(
#                 oc_name, pretrained=oc_pre, device=self.device
#             )
#             self.model.eval()
#             # open_clip uses 512 for ViT-B-32/16
#             self.dim = int(self.model.text_projection.shape[1]) if hasattr(self.model, "text_projection") else 512
#             self._backend = "open_clip"
#         else:
#             self.proc = CLIPProcessor.from_pretrained(model_name)
#             self.model = CLIPModel.from_pretrained(model_name)
#             self.model.to(self.device).eval()
#             self.dim = int(self.model.vision_model.config.hidden_size)  # 768 for B/16; final proj->512 in projection
#             # We’ll read the final image_embeds (already projected)
#             self._backend = "hf"

#         # one tensor to reuse dtype/device
#         self._no_grad = torch.inference_mode()

#     def _to_image(self, x) -> Image.Image:
#         if isinstance(x, Image.Image):
#             return x.convert("RGB")
#         # assume numpy
#         arr = np.asarray(x)
#         if arr.dtype != np.uint8:
#             arr = arr.clip(0, 255).astype(np.uint8)
#         return Image.fromarray(arr, mode="RGB")

#     def encode(self, img) -> np.ndarray:
#         pil = self._to_image(img)
#         with torch.no_grad():
#             if self._backend == "open_clip":
#                 tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
#                 feats = self.model.encode_image(tensor)
#                 feats = torch.nn.functional.normalize(feats, dim=-1)
#                 vec = feats[0].detach().cpu().float().numpy()
#             else:
#                 inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
#                 out = self.model.get_image_features(**inputs)  # already projected to 512 and normalized in practice
#                 out = torch.nn.functional.normalize(out, dim=-1)
#                 vec = out[0].detach().cpu().float().numpy()
#         # Ensure float32 and 1D
#         return np.asarray(vec, dtype=np.float32)
