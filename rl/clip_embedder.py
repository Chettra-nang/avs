# clip_embedder.py
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import numpy as np
from typing import Union, List


class CLIPImageEncoder(nn.Module):
    """CLIP vision encoder for highway observations."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        if "openai" in model_name.lower():
            # Use OpenAI CLIP models
            model_path = model_name.replace("openai/", "")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_path, pretrained="openai", device=device
            )
        else:
            # Use other open_clip models
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="laion2b_s34b_b79k", device=device
            )
        
        self.model.eval()
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            features = self.model.encode_image(dummy_input)
            self.feature_dim = features.shape[-1]
    
    def encode(self, images: Union[np.ndarray, List[np.ndarray], torch.Tensor]) -> torch.Tensor:
        """Encode image(s) to CLIP features."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image
                images = [images]
            elif images.ndim == 4:  # Batch of images
                images = [images[i] for i in range(images.shape[0])]
        
        # Convert to PIL and preprocess
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.shape[-1] == 3:  # RGB
                    pil_img = Image.fromarray(img)
                else:  # Assume grayscale or other format
                    pil_img = Image.fromarray(img).convert('RGB')
            else:
                pil_img = img
            pil_images.append(pil_img)
        
        # Preprocess and batch
        processed = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        # Encode
        with torch.no_grad():
            features = self.model.encode_image(processed)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features
    
    def __call__(self, images):
        return self.encode(images)