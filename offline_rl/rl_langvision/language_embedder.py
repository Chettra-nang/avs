from __future__ import annotations
from typing import List
import numpy as np

class FrozenTextEmbedder:
    """
    Sentence-Transformers (local, no API key). Keeps the model in memory
    and returns L2-normalized float32 vectors.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        try:
            self.dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self.dim = len(self.model.encode(["probe"], normalize_embeddings=True)[0])

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
