#!/usr/bin/env python3
from image_stack_visualizer import ImageStackVisualizer

viz = ImageStackVisualizer()
file_path = '/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_151946-f35b18e8_transitions.parquet'
if viz.load_parquet_file(file_path):
    viz.debug_data_format(num_rows=2)
    print("\nAttempting to print dataset summary...")
    viz.print_dataset_summary()