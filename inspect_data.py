#!/usr/bin/env python3
"""
Quick data inspection script to understand the collected data format.
"""

import pandas as pd
import json
from pathlib import Path

def inspect_collected_data():
    """Inspect the structure of collected ambulance data."""
    
    data_path = Path("data/test_fast_ambulance")
    
    # Load index
    index_file = data_path / "index.json"
    print(f"Reading index: {index_file}")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    print("Index content:")
    print(json.dumps(index, indent=2))
    
    # Check first scenario
    first_scenario = list(index['scenarios'].keys())[0]
    scenario_dir = data_path / first_scenario
    
    print(f"\nInspecting scenario: {first_scenario}")
    print(f"Scenario directory: {scenario_dir}")
    
    # Find and inspect parquet file
    parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
    if parquet_files:
        parquet_file = parquet_files[0]
        print(f"Loading parquet file: {parquet_file}")
        
        df = pd.read_parquet(parquet_file)
        print(f"\nDataFrame shape: {df.shape}")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        
        # Check if we have agent_id column
        if 'agent_id' in df.columns:
            print(f"\nUnique agent IDs: {df['agent_id'].unique()}")
            
            # Check ambulance data (agent 0)
            ambulance_data = df[df['agent_id'] == 0]
            print(f"Ambulance data points: {len(ambulance_data)}")
            if len(ambulance_data) > 0:
                print("\nAmbulance data sample:")
                print(ambulance_data.head())
    
    # Check metadata file
    metadata_files = list(scenario_dir.glob("*_meta.jsonl"))
    if metadata_files:
        metadata_file = metadata_files[0]
        print(f"\nLoading metadata file: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 lines
                    metadata = json.loads(line.strip())
                    print(f"Metadata line {i+1}: {json.dumps(metadata, indent=2)}")
                else:
                    break

if __name__ == "__main__":
    inspect_collected_data()