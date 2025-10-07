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
        # Calculate total input dimension with safer access
        try:
            image_dim = observation_space['image_features'].shape[0]
            text_dim = observation_space['text_features'].shape[0]  
            vector_dim = observation_space['vector'].shape[0]
        except (KeyError, AttributeError):
            # Fallback dimensions if space not properly initialized
            image_dim = 512  # CLIP image features
            text_dim = 384   # Text embedding features  
            vector_dim = 10  # Highway env vector features
        
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
        
    def forward(self, observations) -> torch.Tensor:
        # Extract individual components
        image_features = observations['image_features']
        text_features = observations['text_features']
        vector_features = observations['vector']
        
        # Get batch size from any of the tensors
        batch_size = None
        for feat in [image_features, text_features, vector_features]:
            if len(feat.shape) > 0:
                batch_size = feat.shape[0]
                break
        
        if batch_size is None:
            batch_size = 1
        
        # Reshape all features to (batch_size, -1) format
        def reshape_feature(feat):
            if len(feat.shape) == 1:
                # Single sample, add batch dimension
                return feat.unsqueeze(0)
            elif len(feat.shape) > 2:
                # Multi-dimensional, flatten to 2D
                return feat.view(batch_size, -1)
            else:
                # Already 2D
                return feat
        
        image_features = reshape_feature(image_features)
        text_features = reshape_feature(text_features)
        vector_features = reshape_feature(vector_features)
        
        # Concatenate all features
        combined_features = torch.cat([
            image_features.float(),
            text_features.float(),
            vector_features.float()
        ], dim=-1)
        
        # Apply feature fusion
        output = self.feature_fusion(combined_features)
        
        return output