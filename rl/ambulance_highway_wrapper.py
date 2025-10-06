# ambulance_highway_wrapper.py
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from typing import Optional, Dict as TypeDict, Any
import cv2


class AmbulanceHighwayCLIPWrapper(gym.Wrapper):
    """
    Wrapper that adds CLIP vision encoding and text context to highway-env 
    for ambulance scenarios.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        clip_encoder,
        text_embedder=None,
        cached_llm=None,
        clip_stride: int = 4,
        text_context: str = None
    ):
        super().__init__(env)
        self.clip_encoder = clip_encoder
        self.text_embedder = text_embedder
        self.cached_llm = cached_llm
        self.clip_stride = clip_stride
        self.step_count = 0
        
        # Default text context for ambulance scenarios
        if text_context is None:
            text_context = ("Emergency ambulance navigating through dense "
                          "highway traffic to reach hospital quickly while "
                          "maintaining safety")
        self.text_context = text_context
        
        # Encode text context
        if self.text_embedder is not None:
            self.text_features = self.text_embedder.encode(text_context)
        elif self.cached_llm is not None:
            self.text_features = torch.from_numpy(
                self.cached_llm.embed(text_context)
            ).float()
        else:
            # Use a default embedding if no text encoder available
            self.text_features = torch.zeros(384)
        
        # Ensure text features are always 1D
        if len(self.text_features.shape) > 1:
            self.text_features = self.text_features.flatten()
        
        # Get dimensions
        clip_dim = self.clip_encoder.feature_dim
        text_dim = self.text_features.shape[-1]
        
        # Define new observation space
        self.observation_space = Dict({
            'image_features': Box(
                low=-np.inf, high=np.inf, shape=(clip_dim,), dtype=np.float32
            ),
            'text_features': Box(
                low=-np.inf, high=np.inf, shape=(text_dim,), dtype=np.float32
            ),
            'vector': Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )  # Placeholder for additional features
        })
    
    def _get_visual_obs(self):
        """Get visual observation from environment."""
        rgb_array = self.env.render()
        return rgb_array
    
    def _encode_observation(self, obs):
        """Encode observation using CLIP and text embedder."""
        # Get visual features (only every clip_stride steps to save computation)
        if self.step_count % self.clip_stride == 0:
            rgb = self._get_visual_obs()
            if rgb is not None:
                self.cached_visual_features = self.clip_encoder.encode(rgb)
                if len(self.cached_visual_features.shape) > 1:
                    self.cached_visual_features = self.cached_visual_features[0]
        
        # Create vector features from original observation
        if hasattr(obs, 'flatten'):
            vector_features = obs.flatten()[:10]  # Take first 10 elements
        else:
            vector_features = np.zeros(10)
        
        # Pad or truncate to exactly 10 dimensions
        if len(vector_features) < 10:
            vector_features = np.pad(
                vector_features, (0, 10 - len(vector_features))
            )
        else:
            vector_features = vector_features[:10]
        
        # Ensure text features are properly flattened
        text_features_np = self.text_features.cpu().numpy().astype(np.float32)
        if len(text_features_np.shape) > 1:
            text_features_np = text_features_np.flatten()
        
        return {
            'image_features': self.cached_visual_features.cpu().numpy().astype(np.float32),
            'text_features': text_features_np,
            'vector': vector_features.astype(np.float32)
        }
    
    def reset(self, **kwargs):
        """Reset environment and initialize features."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        
        # Initialize visual features
        rgb = self._get_visual_obs()
        if rgb is not None:
            self.cached_visual_features = self.clip_encoder.encode(rgb)
            if len(self.cached_visual_features.shape) > 1:
                self.cached_visual_features = self.cached_visual_features[0]
        else:
            self.cached_visual_features = torch.zeros(self.clip_encoder.feature_dim)
        
        encoded_obs = self._encode_observation(obs)
        return encoded_obs, info
    
    def step(self, action):
        """Step environment and encode observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        encoded_obs = self._encode_observation(obs)
        return encoded_obs, reward, terminated, truncated, info


class CachedLLMEmbedder:
    """Simple cached embedder interface."""
    
    def __init__(self, cache_dir: str, dim: int = 1536):
        self.cache_dir = cache_dir
        self.dim = dim
        self.cache = {}
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text (with caching)."""
        if text in self.cache:
            return self.cache[text]
        
        # Fallback: return random embedding
        embedding = np.random.randn(self.dim).astype(np.float32)
        self.cache[text] = embedding
        return embedding