# language_embedder.py
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List


class FrozenTextEmbedder(nn.Module):
    """Frozen text embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.encoder.eval()
        
        # Freeze parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Get embedding dimension
        test_text = "test"
        with torch.no_grad():
            test_embedding = self.encoder.encode([test_text])
            self.embedding_dim = test_embedding.shape[-1]
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            embeddings = self.encoder.encode(texts)
            embeddings = torch.from_numpy(embeddings).float()
            
        return embeddings
    
    def forward(self, texts):
        return self.encode(texts)
    
    def __call__(self, texts):
        return self.encode(texts)