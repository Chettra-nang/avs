# features_extractor_clip.py
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CLIPLangExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that combines CLIP image features with text features.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # Calculate total input dimension
        image_dim = observation_space['image_features'].shape[0]
        text_dim = observation_space['text_features'].shape[0]  
        vector_dim = observation_space['vector'].shape[0]
        
        total_input_dim = image_dim + text_dim + vector_dim
        
        super().__init__(observation_space, features_dim)
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract individual components
        image_features = observations['image_features']
        text_features = observations['text_features'] 
        vector_features = observations['vector']
        
        # Ensure proper shapes and types
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)
        if len(vector_features.shape) == 1:
            vector_features = vector_features.unsqueeze(0)
            
        # Concatenate all features
        combined_features = torch.cat([
            image_features.float(),
            text_features.float(), 
            vector_features.float()
        ], dim=-1)
        
        # Apply feature fusion
        output = self.feature_fusion(combined_features)
        
        return output