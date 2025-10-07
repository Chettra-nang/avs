#!/usr/bin/env python3
"""
Quick Parquet Data Explorer

This script examines the structure of ambulance dataset parquet files
to understand the data format before creating comprehensive analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_parquet_structure():
    """Explore the structure of parquet files in the dataset."""
    
    # Path to a sample parquet file
    sample_file = Path(r"d:\Research_ITC\avs_folder\avs\data\ambulance_dataset_150_espisode_cpu_30_senario\ambulance_dataset_150_espisode_cpu_30_senario\batch_653831\highway_merge_heavy\20251004_121200-32e17d2d_transitions.parquet")
    
    logger.info(f"Examining: {sample_file.name}")
    
    if not sample_file.exists():
        logger.error(f"File not found: {sample_file}")
        return
    
    try:
        # Load the parquet file
        df = pd.read_parquet(sample_file)
        
        print("\n" + "="*60)
        print("PARQUET FILE STRUCTURE ANALYSIS")
        print("="*60)
        print(f"File: {sample_file.name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print()
        
        print("COLUMN NAMES:")
        print("-" * 40)
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. {col}")
        print()
        
        print("DATA TYPES:")
        print("-" * 40)
        print(df.dtypes)
        print()
        
        print("FIRST FEW ROWS:")
        print("-" * 40)
        print(df.head(3))
        print()
        
        # Look for speed/velocity related columns
        speed_cols = [col for col in df.columns if any(word in col.lower() 
                     for word in ['speed', 'velocity', 'vel', 'v_'])]
        
        if speed_cols:
            print("SPEED-RELATED COLUMNS:")
            print("-" * 40)
            for col in speed_cols:
                print(f"  - {col}")
                print(f"    Sample values: {df[col].head(3).tolist()}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"    Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print()
        
        # Look for position related columns
        pos_cols = [col for col in df.columns if any(word in col.lower() 
                   for word in ['pos', 'x', 'y', 'position'])]
        
        if pos_cols:
            print("POSITION-RELATED COLUMNS:")
            print("-" * 40)
            for col in pos_cols:
                print(f"  - {col}")
                print(f"    Sample values: {df[col].head(3).tolist()}")
        print()
        
        # Look for observation columns
        obs_cols = [col for col in df.columns if 'obs' in col.lower()]
        
        if obs_cols:
            print("OBSERVATION COLUMNS:")
            print("-" * 40)
            for col in obs_cols:
                print(f"  - {col}")
                sample_val = df[col].iloc[0]
                print(f"    Type: {type(sample_val)}")
                if hasattr(sample_val, 'shape'):
                    print(f"    Shape: {sample_val.shape}")
                elif isinstance(sample_val, (list, tuple)):
                    print(f"    Length: {len(sample_val)}")
                else:
                    print(f"    Sample: {sample_val}")
        print()
        
        # Check for any array/list columns
        array_cols = []
        for col in df.columns:
            sample_val = df[col].iloc[0] if len(df) > 0 else None
            if isinstance(sample_val, (list, tuple, np.ndarray)):
                array_cols.append(col)
        
        if array_cols:
            print("ARRAY/LIST COLUMNS:")
            print("-" * 40)
            for col in array_cols:
                sample_val = df[col].iloc[0]
                print(f"  - {col}")
                print(f"    Type: {type(sample_val)}")
                if hasattr(sample_val, 'shape'):
                    print(f"    Shape: {sample_val.shape}")
                    if len(sample_val.shape) == 1 and sample_val.shape[0] <= 10:
                        print(f"    Values: {sample_val}")
                elif isinstance(sample_val, (list, tuple)):
                    print(f"    Length: {len(sample_val)}")
                    if len(sample_val) <= 10:
                        print(f"    Values: {sample_val}")
        print()
        
        print("SUMMARY STATISTICS:")
        print("-" * 40)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("No numeric columns found")
        
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    explore_parquet_structure()