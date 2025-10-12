#!/usr/bin/env python3
"""
Verify offline RL pipeline works on RTX 5090.

Tests:
1. Dataset export from parquet
2. CLIP encoder initialization on CUDA
3. Offline DQN training (1 epoch)
4. BC training (1 epoch)
5. GPU memory usage and throughput

Usage:
    python scripts/verify_offline_pipeline.py \
        --data-dir data/ambulance_dataset_diagnose \
        --test-samples 100
"""
import argparse
import time
from pathlib import Path
import tempfile
import sys

import numpy as np
import torch


def test_cuda_available():
    """Test 1: CUDA availability."""
    print("\n" + "="*60)
    print("TEST 1: CUDA Availability")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return True


def test_dataset_export(data_dir: Path, output_dir: Path):
    """Test 2: Dataset export."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Export")
    print("="*60)
    
    sys.path.insert(0, str(Path(__file__).parent))
    from export_offline_dataset import export_dataset
    
    try:
        export_dataset(data_dir, output_dir, format='npz')
        
        # Verify output
        dataset_path = output_dir / 'offline_dataset.npz'
        if not dataset_path.exists():
            print(f"❌ Dataset file not created: {dataset_path}")
            return False
        
        data = np.load(dataset_path)
        print(f"✅ Dataset exported successfully")
        print(f"   Shape: obs={data['obs'].shape}, actions={data['action'].shape}")
        print(f"   File size: {dataset_path.stat().st_size / 1e6:.2f} MB")
        return True
    
    except Exception as e:
        print(f"❌ Dataset export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_encoder():
    """Test 3: CLIP encoder initialization."""
    print("\n" + "="*60)
    print("TEST 3: CLIP Encoder")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "rl/Ambulance_EGO_4500 2/Ambulance_EGO_4500"))
        from rl_langvision.clip_embedder import CLIPImageEncoder
        
        encoder = CLIPImageEncoder(device='cuda')
        
        # Test forward pass
        dummy_img = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        
        start = time.time()
        embedding = encoder.encode_np_rgb(dummy_img)
        elapsed = time.time() - start
        
        print(f"✅ CLIP encoder initialized on CUDA")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Inference time: {elapsed*1000:.2f} ms")
        print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        return True
    
    except Exception as e:
        print(f"❌ CLIP encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_training(dataset_path: Path, output_dir: Path):
    """Test 4: DQN training (1 epoch)."""
    print("\n" + "="*60)
    print("TEST 4: Offline DQN Training")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from offline_rl.trainers.train_offline_dqn import OfflineRLDataset, OfflineDQNTrainer
        
        # Load dataset
        dataset = OfflineRLDataset(dataset_path)
        
        # Create trainer
        trainer = OfflineDQNTrainer(
            dataset=dataset,
            batch_size=32,  # Small batch for test
            device='cuda',
        )
        
        # Train 1 epoch
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        metrics = trainer.train_epoch()
        elapsed = time.time() - start
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ DQN training successful")
        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   Q mean: {metrics['q_mean']:.3f}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Peak GPU memory: {peak_mem:.2f} GB")
        
        # Save checkpoint
        trainer.save_checkpoint(output_dir / 'test_dqn_checkpoint.pt')
        return True
    
    except Exception as e:
        print(f"❌ DQN training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bc_training(dataset_path: Path, output_dir: Path):
    """Test 5: BC training (1 epoch)."""
    print("\n" + "="*60)
    print("TEST 5: Behavior Cloning Training")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from offline_rl.trainers.train_bc import OfflineRLDataset, BCTrainer
        from torch.utils.data import random_split
        
        # Load and split dataset
        full_dataset = OfflineRLDataset(dataset_path)
        train_size = int(len(full_dataset) * 0.9)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create trainer
        trainer = BCTrainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=32,
            device='cuda',
        )
        
        # Train 1 epoch
        start = time.time()
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate()
        elapsed = time.time() - start
        
        print(f"✅ BC training successful")
        print(f"   Train accuracy: {train_metrics['accuracy']:.3f}")
        print(f"   Val accuracy: {val_metrics['val_accuracy']:.3f}")
        print(f"   Time: {elapsed:.2f}s")
        
        # Save checkpoint
        trainer.save_checkpoint(output_dir / 'test_bc_checkpoint.pt')
        return True
    
    except Exception as e:
        print(f"❌ BC training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify offline RL pipeline")
    parser.add_argument('--data-dir', type=str, default='data/ambulance_dataset_diagnose',
                        help='Directory with parquet files')
    parser.add_argument('--test-samples', type=int, default=100,
                        help='Number of samples to test with')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset_dir = tmpdir / 'dataset'
        checkpoint_dir = tmpdir / 'checkpoints'
        dataset_dir.mkdir()
        checkpoint_dir.mkdir()
        
        print("\n" + "="*60)
        print("OFFLINE RL PIPELINE VERIFICATION")
        print("="*60)
        print(f"Data directory: {data_dir}")
        print(f"Temp output: {tmpdir}")
        
        results = {}
        
        # Run tests
        results['cuda'] = test_cuda_available()
        
        if results['cuda']:
            results['export'] = test_dataset_export(data_dir, dataset_dir)
            
            if results['export']:
                dataset_path = dataset_dir / 'offline_dataset.npz'
                results['clip'] = test_clip_encoder()
                results['dqn'] = test_dqn_training(dataset_path, checkpoint_dir)
                results['bc'] = test_bc_training(dataset_path, checkpoint_dir)
        
        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.upper():15s} {status}")
        
        all_passed = all(results.values())
        print("="*60)
        if all_passed:
            print("🎉 All tests passed! Pipeline ready for RTX 5090.")
        else:
            print("⚠️  Some tests failed. Check errors above.")
        print("="*60)
        
        return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
