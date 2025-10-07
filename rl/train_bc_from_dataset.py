# train_bc_from_dataset.py - Behavior Cloning from your ambulance dataset
#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
from gymnasium import spaces
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from clip_embedder import CLIPImageEncoder
from language_embedder import FrozenTextEmbedder
from features_extractor_clip import CLIPLangExtractor
from ambulance_dataset_io import AmbulanceDatasetLoader


def create_observation_space(state_dim: int, clip_dim: int, text_dim: int):
    """Create observation space for BC training."""
    return spaces.Dict({
        "image_features": spaces.Box(
            low=-np.inf, high=np.inf, shape=(clip_dim,), dtype=np.float32
        ),
        "text_features": spaces.Box(
            low=-np.inf, high=np.inf, shape=(text_dim,), dtype=np.float32
        ),
        "vector": spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
    })


def load_transitions_from_npz(npz_path: str) -> Transitions:
    """Load transitions from NPZ file."""
    print(f"Loading transitions from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Prepare observations
    obs = {
        "image_features": data["obs_clip"].astype(np.float32),
        "text_features": data["obs_text"].astype(np.float32),
        "vector": data["obs_state"].astype(np.float32)
    }
    
    next_obs = {
        "image_features": data["next_obs_clip"].astype(np.float32),
        "text_features": data["next_obs_text"].astype(np.float32),
        "vector": data["next_obs_state"].astype(np.float32)
    }
    
    actions = data["acts"].astype(np.int64)
    dones = data["dones"].astype(bool)
    
    print(f"Loaded {len(actions)} transitions")
    print(f"Observation shapes: {[v.shape for v in obs.values()]}")
    print(f"Action shape: {actions.shape}, unique actions: {np.unique(actions)}")
    
    return Transitions(
        obs=obs,
        acts=actions,
        next_obs=next_obs,
        dones=dones,
        infos=None
    )


def train_behavior_cloning(
    data_dir: str,
    output_dir: str = "bc_models",
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "auto"
):
    """Train behavior cloning model from ambulance dataset."""
    
    # Setup device - use explicit device parameter
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Force device if explicitly specified
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    print(f"Training Behavior Cloning on {device}")
    print(f"Data directory: {data_dir}")
    
    # Validate CUDA availability if requested
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup data loader
    dataset_loader = AmbulanceDatasetLoader(data_dir)
    
    # Check if NPZ data exists, if not build it
    npz_train_path = f"{data_dir}/npz/train_clip_vlm.npz"
    npz_val_path = f"{data_dir}/npz/val_clip_vlm.npz"
    
    if not os.path.exists(npz_train_path):
        print("NPZ data not found, building from manifests...")
        
        # Initialize encoders
        clip_encoder = CLIPImageEncoder("ViT-B-32", device=device)
        text_embedder = FrozenTextEmbedder()
        
        # Build NPZ data
        dataset_loader.build_npz_from_manifests(
            clip_encoder, text_embedder, 
            output_dir=f"{data_dir}/npz"
        )
    
    # Load training data
    if not os.path.exists(npz_train_path):
        print("No training data found. Please check your dataset format.")
        return
    
    train_transitions = load_transitions_from_npz(npz_train_path)
    
    # Load validation data (optional)
    val_transitions = None
    if os.path.exists(npz_val_path):
        val_transitions = load_transitions_from_npz(npz_val_path)
    
    # Get dimensions from data
    sample_obs = {k: v[0] for k, v in train_transitions.obs.items()}
    clip_dim = sample_obs["image_features"].shape[0]
    text_dim = sample_obs["text_features"].shape[0]
    state_dim = sample_obs["vector"].shape[0]
    
    print(f"Feature dimensions - CLIP: {clip_dim}, Text: {text_dim}, State: {state_dim}")
    
    # Create observation and action spaces
    observation_space = create_observation_space(state_dim, clip_dim, text_dim)
    action_space = spaces.Discrete(5)  # Highway-env action space
    
    # Setup policy
    policy_kwargs = dict(
        features_extractor_class=CLIPLangExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    # Create BC algorithm
    bc_policy = MultiInputActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        **policy_kwargs
    ).to(device)
    
    bc_trainer = BC(
        observation_space=observation_space,
        action_space=action_space,
        policy=bc_policy,
        demonstrations=train_transitions,
        device=device,
    )
    
    # Train
    print(f"Starting BC training for {epochs} epochs...")
    bc_trainer.train(
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_interval=10
    )
    
    # Save model
    policy_path = f"{output_dir}/bc_ambulance_policy.pt"
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"Saved BC policy to: {policy_path}")
    
    # Save full model
    bc_trainer.save_policy(f"{output_dir}/bc_ambulance_full.zip")
    print(f"Saved full BC model to: {output_dir}/bc_ambulance_full.zip")
    
    # Evaluate on validation set
    if val_transitions is not None:
        print("Evaluating on validation set...")
        # Simple accuracy evaluation
        val_obs = val_transitions.obs
        val_acts = val_transitions.acts
        
        with torch.no_grad():
            pred_acts = bc_trainer.policy.predict(val_obs, deterministic=True)[0]
            accuracy = (pred_acts == val_acts).mean()
            print(f"Validation accuracy: {accuracy:.3f}")
    
    return policy_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train BC from ambulance dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to ambulance dataset directory")
    parser.add_argument("--output_dir", type=str, default="bc_models",
                       help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        sys.exit(1)
    
    # Train BC
    policy_path = train_behavior_cloning(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    print(f"\n=== BC Training Complete! ===")
    print(f"Policy saved to: {policy_path}")
    print(f"Use this for warm-starting RL training")


if __name__ == "__main__":
    main()