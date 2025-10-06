# clip_embedder.py - Fixed version for open-clip
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import numpy as np
from typing import Union, List


class CLIPImageEncoder(nn.Module):
    """CLIP vision encoder for highway observations."""
    
    def __init__(self, model_name: str = "ViT-B-32", device: str = "auto"):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Use a known working model
        try:
            # Try ViT-B-32 with OpenAI weights first
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai', device=device
            )
        except Exception as e1:
            try:
                # Fallback to LAION weights
                print(f"OpenAI weights failed ({e1}), trying LAION...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
                )
            except Exception as e2:
                # Last resort - try RN50
                print(f"ViT-B-32 failed ({e2}), trying RN50...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'RN50', pretrained='openai', device=device
                )
        
        self.model.eval()
        
        # Get feature dimension
        with torch.no_grad():
            # Create a proper dummy image tensor
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            features = self.model.encode_image(dummy_input)
            self.feature_dim = features.shape[-1]
        
        print(f"CLIP model loaded successfully. Feature dim: {self.feature_dim}")
    
    def encode(self, images: Union[np.ndarray, List[np.ndarray], torch.Tensor]) -> torch.Tensor:
        """Encode image(s) to CLIP features."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image (H, W, C)
                images = [images]
            elif images.ndim == 4:  # Batch of images (B, H, W, C)
                images = [images[i] for i in range(images.shape[0])]
        
        # Convert to PIL and preprocess
        processed_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Ensure uint8 format
                if img.dtype != np.uint8:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                
                # Convert to PIL Image
                if len(img.shape) == 3 and img.shape[-1] == 3:  # RGB
                    pil_img = Image.fromarray(img, 'RGB')
                else:  # Grayscale or other
                    if len(img.shape) == 2:  # Grayscale
                        pil_img = Image.fromarray(img, 'L').convert('RGB')
                    else:
                        pil_img = Image.fromarray(img).convert('RGB')
            else:
                pil_img = img
                
            # Apply preprocessing
            processed_tensor = self.preprocess(pil_img)
            processed_tensors.append(processed_tensor)
        
        # Stack and move to device
        if len(processed_tensors) == 1:
            batch = processed_tensors[0].unsqueeze(0).to(self.device)
        else:
            batch = torch.stack(processed_tensors).to(self.device)
        
        # Encode
        with torch.no_grad():
            features = self.model.encode_image(batch)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def __call__(self, images):
        return self.encode(images)