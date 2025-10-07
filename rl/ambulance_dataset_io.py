# ambulance_dataset_io.py - Dataset loading utilities
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from PIL import Image
import os


class AmbulanceDatasetLoader:
    """Load and process ambulance dataset for training."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.manifests_dir = self.data_dir / "manifests"
        self.npz_dir = self.data_dir / "npz"
        
    def load_manifests(self) -> Dict[str, list]:
        """Load train/val/test manifests."""
        manifests = {}
        for split in ["train", "val", "test"]:
            manifest_path = self.manifests_dir / f"{split}.jsonl"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifests[split] = [json.loads(line) for line in f]
            else:
                manifests[split] = []
        return manifests
    
    def build_npz_from_manifests(
        self, 
        clip_encoder,
        text_embedder,
        output_dir: str = "npz_processed"
    ):
        """Convert JSONL manifests to NPZ format for training."""
        os.makedirs(output_dir, exist_ok=True)
        manifests = self.load_manifests()
        
        for split, episodes in manifests.items():
            if not episodes:
                continue
                
            print(f"Processing {split} split: {len(episodes)} episodes")
            
            # Collect all data
            obs_state, obs_clip, obs_text = [], [], []
            next_obs_state, next_obs_clip, next_obs_text = [], [], []
            actions, rewards, dones = [], [], []
            
            for episode in episodes:
                episode_id = episode["episode_id"]
                scenario = episode["scenario"]
                
                # Load episode data (you'll need to implement this based on your data format)
                episode_data = self._load_episode_data(episode_id, scenario)
                if episode_data is None:
                    continue
                    
                # Process each step in the episode
                for step_data in episode_data:
                    # State features
                    state = step_data.get("state", np.zeros(10))
                    next_state = step_data.get("next_state", np.zeros(10))
                    
                    # Visual features (CLIP encoding)
                    if "image" in step_data:
                        clip_feat = clip_encoder.encode(step_data["image"])
                        if len(clip_feat.shape) > 1:
                            clip_feat = clip_feat[0]
                        clip_feat = clip_feat.cpu().numpy()
                    else:
                        clip_feat = np.zeros(512)  # Default CLIP dim
                    
                    if "next_image" in step_data:
                        next_clip_feat = clip_encoder.encode(step_data["next_image"])
                        if len(next_clip_feat.shape) > 1:
                            next_clip_feat = next_clip_feat[0]
                        next_clip_feat = next_clip_feat.cpu().numpy()
                    else:
                        next_clip_feat = clip_feat
                    
                    # Text features
                    text_context = f"Emergency ambulance in {scenario} scenario"
                    text_feat = text_embedder.encode(text_context)
                    if len(text_feat.shape) > 1:
                        text_feat = text_feat[0]
                    text_feat = text_feat.cpu().numpy()
                    
                    # Actions and rewards
                    action = step_data.get("action", 0)
                    reward = step_data.get("reward", 0.0)
                    done = step_data.get("done", False)
                    
                    # Append to lists
                    obs_state.append(state)
                    obs_clip.append(clip_feat)
                    obs_text.append(text_feat)
                    next_obs_state.append(next_state)
                    next_obs_clip.append(next_clip_feat)
                    next_obs_text.append(text_feat)  # Same text for next obs
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
            
            if len(actions) == 0:
                print(f"No data found for {split} split")
                continue
            
            # Convert to numpy arrays
            data = {
                "obs_state": np.array(obs_state, dtype=np.float32),
                "obs_clip": np.array(obs_clip, dtype=np.float32),
                "obs_text": np.array(obs_text, dtype=np.float32),
                "next_obs_state": np.array(next_obs_state, dtype=np.float32),
                "next_obs_clip": np.array(next_obs_clip, dtype=np.float32),
                "next_obs_text": np.array(next_obs_text, dtype=np.float32),
                "acts": np.array(actions, dtype=np.int64),
                "rews": np.array(rewards, dtype=np.float32),
                "dones": np.array(dones, dtype=bool),
                # Metadata
                "k_state": obs_state[0].shape[0],
                "d_clip": obs_clip[0].shape[0],
                "d_text": obs_text[0].shape[0],
                "action_space_n": 5  # Highway-env action space
            }
            
            # Save NPZ
            output_path = f"{output_dir}/{split}_clip_vlm.npz"
            np.savez_compressed(output_path, **data)
            print(f"Saved {len(actions)} samples to {output_path}")
    
    def _load_episode_data(self, episode_id: str, scenario: str) -> Optional[list]:
        """Load episode data from your dataset format."""
        # This is a placeholder - you'll need to implement based on your actual data format
        # For now, return None to indicate no data found
        print(f"Warning: _load_episode_data not implemented for {episode_id}")
        return None
    
    def load_npz_data(self, split: str = "train") -> Optional[Dict[str, np.ndarray]]:
        """Load processed NPZ data."""
        npz_path = self.npz_dir / f"{split}_clip_vlm.npz"
        if npz_path.exists():
            return dict(np.load(npz_path, allow_pickle=True))
        return None