# rl_langvision/text_embedder.py
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional
import numpy as np

# Option A: cached MiniLM (384-d) — your current fast path
class CachedLLMEmbedder:
    """
    Reads parquet shards you created (columns: episode_id, step, embedding).
    For RL, we typically don't have matching episode_ids; so we expose:
      - get_by_step(step): cycling through a single shard as a simple curriculum
      - OR get_constant(): e.g., average embedding (corridor guidance)
    """
    def __init__(self, folder: str | Path, dim: int = 384):
        import pandas as pd
        self.folder = Path(folder)
        self.dim = dim
        self.vectors = []
        for p in sorted(self.folder.glob("*.parquet")):
            try:
                df = pd.read_parquet(p, columns=["embedding"])
                self.vectors.extend(df["embedding"].tolist())
            except Exception:
                continue
        if not self.vectors:
            raise FileNotFoundError(f"No embeddings found in {self.folder}")
        self.n = len(self.vectors)
        self.mean_vec = np.asarray(self.vectors, dtype=np.float32).mean(axis=0)
        # L2 norm
        nrm = np.linalg.norm(self.mean_vec) + 1e-12
        self.mean_vec = (self.mean_vec / nrm).astype(np.float32)

    def get_by_step(self, step: int) -> np.ndarray:
        v = np.asarray(self.vectors[step % self.n], dtype=np.float32)
        nrm = np.linalg.norm(v) + 1e-12
        return (v / nrm).astype(np.float32)

    def get_constant(self) -> np.ndarray:
        return self.mean_vec.copy()

# Option B: on-the-fly MiniLM (fallback if you’d rather not preload)
class FrozenTextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import torch
        if device is None:
            if torch.cuda.is_available(): device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    @lru_cache(maxsize=4096)
    def encode_one(self, text: str) -> np.ndarray:
        v = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v.astype(np.float32)




# from __future__ import annotations
# from pathlib import Path
# from typing import Dict, Tuple
# import numpy as np
# import pandas as pd
# import json

# class CachedLLMEmbedder:
#     """
#     Loads your parquet shards produced by tools/make_text_embeddings.py
#     and serves vectors by (episode_id, step) if available, or falls back to a
#     default vector. Use this when you *don't* want to run a text model online.
#     """
#     def __init__(self, root: str | Path, dim: int | None = None):
#         root = Path(root)
#         self.root = root
#         self.dim = int(dim) if dim is not None else self._read_dim()
#         self._table: Dict[Tuple[str, int], np.ndarray] = {}
#         self._load_all()

#     def _read_dim(self) -> int:
#         info = self.root / "embed_info.json"
#         if info.exists():
#             try:
#                 return int(json.loads(info.read_text()).get("dim", 384))
#             except Exception:
#                 pass
#         return 384

#     def _load_all(self):
#         files = sorted(self.root.glob("embeddings_*.parquet"))
#         for f in files:
#             try:
#                 df = pd.read_parquet(f, columns=["episode_id", "step", "embedding"])
#             except Exception:
#                 continue
#             for eid, step, emb in zip(df["episode_id"], df["step"], df["embedding"]):
#                 vec = np.asarray(emb, dtype=np.float32)
#                 if vec.shape[0] != self.dim:
#                     # pad or trim for robustness
#                     if vec.shape[0] > self.dim:
#                         vec = vec[: self.dim]
#                     else:
#                         tmp = np.zeros(self.dim, dtype=np.float32)
#                         tmp[: vec.shape[0]] = vec
#                         vec = tmp
#                 self._table[(str(eid), int(step))] = vec

#         if not self._table:
#             # keep at least a dummy vector
#             self._default = np.zeros(self.dim, dtype=np.float32)
#         else:
#             self._default = np.mean(np.stack(list(self._table.values())), axis=0).astype(np.float32)

#     def get(self, episode_id: str, step: int) -> np.ndarray:
#         return self._table.get((episode_id, int(step)), self._default).astype(np.float32)

#     def get_default(self) -> np.ndarray:
#         return self._default.copy()
