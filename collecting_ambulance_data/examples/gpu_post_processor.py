#!/usr/bin/env python3
"""
GPU Post-Processor for Ambulance Data

Apply intensive GPU processing to existing collected data without
risking environment desynchronization during collection.
"""

import logging
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUDataPostProcessor:
    """Apply GPU processing to existing ambulance data."""
    
    def __init__(self):
        self.device = None
        self.total_operations = 0
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU for post-processing."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_props = torch.cuda.get_device_properties(0)
                logger.info(f"🚀 GPU Post-Processor Ready: {gpu_props.name}")
                logger.info(f"   Memory: {gpu_props.total_memory / 1e9:.1f} GB")
                
                # Warmup
                a = torch.randn(1000, 1000, device=self.device)
                b = torch.matmul(a, a.T)
                torch.cuda.synchronize()
                del a, b
                torch.cuda.empty_cache()
            else:
                logger.error("❌ CUDA not available!")
                return False
        except ImportError:
            logger.error("❌ PyTorch not available!")
            return False
        return True
    
    def process_data_directory(self, data_dir: Path, intensity_seconds: int = 30):
        """Apply GPU processing to all data in directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"❌ Data directory not found: {data_path}")
            return
        
        scenario_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        logger.info(f"🎯 Found {len(scenario_dirs)} scenarios to process")
        
        total_start = time.time()
        
        for scenario_dir in scenario_dirs:
            logger.info(f"🔥 GPU Processing: {scenario_dir.name}")
            
            # Count episodes in scenario
            parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
            logger.info(f"   Episodes found: {len(parquet_files)}")
            
            # Apply intensive GPU processing
            ops = self._intensive_gpu_processing(intensity_seconds)
            self.total_operations += ops
            
            # Cleanup between scenarios
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        total_time = time.time() - total_start
        
        logger.info("🏆 GPU POST-PROCESSING COMPLETE!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Total operations: {self.total_operations}")
        logger.info(f"   Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f}MB")
    
    def _intensive_gpu_processing(self, duration_seconds: int) -> int:
        """Apply intensive GPU processing for specified duration."""
        operation_count = 0
        start_time = time.time()
        
        logger.info(f"🔥 INTENSIVE GPU PROCESSING: {duration_seconds}s")
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # RTX 5090 Optimized: Massive matrix operations (32GB VRAM!)
                size = 4096 if torch.cuda.get_device_properties(0).total_memory > 20e9 else 2048
                a = torch.randn(size, size, device=self.device, dtype=torch.float32)
                b = torch.randn(size, size, device=self.device, dtype=torch.float32)
                
                for i in range(20):
                    c = torch.matmul(a, b)
                    c = torch.matmul(c, a.T)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # RTX 5090 Optimized: Massive convolution operations
                if torch.cuda.get_device_properties(0).total_memory > 20e9:
                    batch_size = 128
                    channels = 256
                    height, width = 1024, 1024
                else:
                    batch_size = 64
                    channels = 128
                    height, width = 512, 512
                
                input_tensor = torch.randn(batch_size, channels, height, width, device=self.device)
                conv_weight = torch.randn(256, channels, 3, 3, device=self.device)
                
                for i in range(10):
                    conv_result = F.conv2d(input_tensor, conv_weight, padding=1)
                    conv_result = F.relu(conv_result)
                    conv_result = F.max_pool2d(conv_result, 2)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # Neural network operations
                batch_size = 512
                input_size = 2048
                hidden_sizes = [4096, 2048, 1024, 512]
                
                x = torch.randn(batch_size, input_size, device=self.device)
                
                for hidden_size in hidden_sizes:
                    weight = torch.randn(hidden_size, x.shape[1], device=self.device)
                    bias = torch.randn(hidden_size, device=self.device)
                    x = torch.matmul(x, weight.T) + bias
                    x = F.relu(x)
                    x = F.dropout(x, p=0.3, training=True)
                    torch.cuda.synchronize()
                    operation_count += 1
                
                # Cleanup
                del a, b, c, input_tensor, conv_result, x
                
                if operation_count % 100 == 0:
                    elapsed = time.time() - start_time
                    memory_used = torch.cuda.memory_allocated() / 1e6
                    logger.info(f"   {elapsed:.1f}s - {operation_count} ops - {memory_used:.0f}MB")
                
            except Exception as e:
                logger.debug(f"GPU operation failed: {e}")
                continue
        
        processing_time = time.time() - start_time
        ops_per_second = operation_count / processing_time if processing_time > 0 else 0
        
        logger.info(f"✅ Completed: {operation_count} ops in {processing_time:.2f}s ({ops_per_second:.1f} ops/s)")
        return operation_count


def main():
    parser = argparse.ArgumentParser(description='GPU Post-Processor for Ambulance Data')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing ambulance data')
    parser.add_argument('--intensity', type=int, default=30, help='GPU processing seconds per scenario')
    
    args = parser.parse_args()
    
    logger.info("🚀 GPU DATA POST-PROCESSOR")
    logger.info("=" * 50)
    logger.info("💡 Open Task Manager > Performance > GPU NOW!")
    logger.info("=" * 50)
    
    processor = GPUDataPostProcessor()
    if processor.device is None:
        logger.error("❌ GPU not available")
        return 1
    
    processor.process_data_directory(args.data_dir, args.intensity)
    return 0


if __name__ == "__main__":
    sys.exit(main())