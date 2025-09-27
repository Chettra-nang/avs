#!/usr/bin/env python3
"""
GPU STRESS TEST - Focused GPU utilization for ambulance data collection

This version focuses purely on intensive GPU computation that will be clearly
visible in Task Manager GPU usage statistics.
"""

import logging
import time
import argparse
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def gpu_stress_test(duration_seconds: int = 30):
    """
    Run intensive GPU stress test that will be clearly visible in Task Manager.
    
    Args:
        duration_seconds: How long to run the stress test
    """
    try:
        import torch
        import torch.nn.functional as F
        
        if not torch.cuda.is_available():
            logger.error("❌ CUDA not available! Cannot run GPU stress test.")
            return False
        
        device = torch.device('cuda')
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info("🚀 STARTING GPU STRESS TEST")
        logger.info("=" * 60)
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Memory: {gpu_memory:.1f} GB")
        logger.info(f"Duration: {duration_seconds} seconds")
        logger.info("=" * 60)
        logger.info("💡 Open Task Manager > Performance > GPU to see utilization!")
        logger.info("=" * 60)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        start_time = time.time()
        operation_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # STRESS TEST 1: Large matrix multiplications
                size = 1024
                a = torch.randn(size, size, device=device, dtype=torch.float32)
                b = torch.randn(size, size, device=device, dtype=torch.float32)
                
                for i in range(10):
                    c = torch.matmul(a, b)
                    c = torch.matmul(c, a.T)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # STRESS TEST 2: Convolution operations
                batch_size = 32
                channels = 64
                height, width = 256, 256
                
                input_tensor = torch.randn(batch_size, channels, height, width, device=device)
                conv_weight = torch.randn(128, channels, 3, 3, device=device)
                
                for i in range(5):
                    conv_result = F.conv2d(input_tensor, conv_weight, padding=1)
                    conv_result = F.relu(conv_result)
                    conv_result = F.max_pool2d(conv_result, 2)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # STRESS TEST 3: Neural network operations
                batch_size = 256
                input_size = 1024
                hidden_sizes = [2048, 1024, 512, 256]
                
                x = torch.randn(batch_size, input_size, device=device)
                
                for hidden_size in hidden_sizes:
                    weight = torch.randn(hidden_size, x.shape[1], device=device)
                    bias = torch.randn(hidden_size, device=device)
                    x = torch.matmul(x, weight.T) + bias
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=True)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # STRESS TEST 4: Mathematical operations
                large_tensor = torch.randn(10000, 1000, device=device)
                for i in range(3):
                    result = torch.sin(large_tensor) + torch.cos(large_tensor)
                    result = torch.exp(torch.tanh(result * 0.01))
                    result = torch.softmax(result, dim=1)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # Memory cleanup
                del a, b, c, input_tensor, conv_result, x, large_tensor, result
                
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0:  # Log every 5 seconds
                    memory_used = torch.cuda.memory_allocated() / 1e6
                    logger.info(f"🔥 {elapsed:.0f}s elapsed - {operation_count} operations - {memory_used:.0f}MB GPU memory")
                
            except Exception as e:
                logger.error(f"GPU operation failed: {e}")
                continue
        
        total_time = time.time() - start_time
        memory_used = torch.cuda.max_memory_allocated() / 1e6
        
        # Cleanup
        torch.cuda.empty_cache()
        
        logger.info("🏁 GPU STRESS TEST COMPLETED")
        logger.info("=" * 50)
        logger.info(f"✅ Duration: {total_time:.2f} seconds")
        logger.info(f"✅ Operations: {operation_count}")
        logger.info(f"✅ Ops/second: {operation_count/total_time:.1f}")
        logger.info(f"✅ Peak GPU memory: {memory_used:.0f}MB")
        logger.info("=" * 50)
        logger.info("📊 Check Task Manager - GPU should have shown high utilization!")
        
        return True
        
    except ImportError:
        logger.error("❌ PyTorch not available!")
        return False
    except Exception as e:
        logger.error(f"❌ GPU stress test failed: {e}")
        return False


def main():
    """Main function for GPU stress test."""
    parser = argparse.ArgumentParser(description='GPU Stress Test for RTX 3050')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds (default: 30)')
    parser.add_argument('--no-gpu', action='store_true', help='Skip GPU test')
    
    args = parser.parse_args()
    
    if args.no_gpu:
        logger.info("GPU test disabled")
        return 0
    
    logger.info("🎯 GOAL: Maximum GPU utilization visible in Task Manager")
    logger.info("📋 This will stress-test your RTX 3050 GPU with intensive computations")
    logger.info("💡 Open Task Manager > Performance > GPU before starting!")
    logger.info("")
    
    # Countdown
    for i in range(5, 0, -1):
        logger.info(f"Starting in {i}...")
        time.sleep(1)
    
    success = gpu_stress_test(args.duration)
    
    if success:
        logger.info("🚀 GPU stress test completed successfully!")
        logger.info("🎯 Your RTX 3050 should have shown high GPU utilization in Task Manager")
        return 0
    else:
        logger.error("❌ GPU stress test failed")
        return 1


if __name__ == "__main__":
    exit(main())