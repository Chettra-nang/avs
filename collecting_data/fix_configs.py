#!/usr/bin/env python3
"""Fix all scenario files by removing invalid parameters"""

import re
from pathlib import Path

def fix_scenario_config(file_path):
    """Fix collection_config in a scenario file"""
    print(f"Fixing {file_path.name}...")
    
    content = file_path.read_text()
    
    # Find and replace the collection_config
    pattern = r'collection_config = \{[^}]+\}'
    
    def replacement(match):
        config_str = match.group(0)
        
        # Extract valid parameters only
        valid_params = {}
        
        # Extract base_storage_path
        base_path_match = re.search(r'"base_storage_path":\s*([^,\n]+)', config_str)
        if base_path_match:
            valid_params['base_storage_path'] = base_path_match.group(1).strip()
        
        # Extract episodes_per_scenario  
        episodes_match = re.search(r'"episodes_per_scenario":\s*(\d+)', config_str)
        if episodes_match:
            valid_params['episodes_per_scenario'] = episodes_match.group(1)
            
        # Extract n_agents
        agents_match = re.search(r'"n_agents":\s*(\d+)', config_str)
        if agents_match:
            valid_params['n_agents'] = agents_match.group(1)
            
        # Extract max_steps_per_episode
        steps_match = re.search(r'"max_steps_per_episode":\s*(\d+)', config_str)
        if steps_match:
            valid_params['max_steps_per_episode'] = steps_match.group(1)
            
        # Extract scenarios
        scenarios_match = re.search(r'"scenarios":\s*(\[[^\]]+\])', config_str)
        if scenarios_match:
            valid_params['scenarios'] = scenarios_match.group(1)
            
        # Extract base_seed
        seed_match = re.search(r'"base_seed":\s*(\d+)', config_str)
        if seed_match:
            valid_params['base_seed'] = seed_match.group(1)
            
        # Extract batch_size
        batch_match = re.search(r'"batch_size":\s*(\d+)', config_str)
        if batch_match:
            valid_params['batch_size'] = batch_match.group(1)
        
        # Build the new config
        new_config = "collection_config = {\n"
        for key, value in valid_params.items():
            new_config += f'        "{key}": {value},\n'
        new_config = new_config.rstrip(',\n') + '\n    }'
        
        return new_config
    
    # Replace the config
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    file_path.write_text(new_content)
    print(f"✅ Fixed {file_path.name}")

def main():
    """Fix all scenario files"""
    collecting_data_dir = Path("/home/chettra/ITC/Research/AVs/collecting_data")
    
    scenario_files = list(collecting_data_dir.glob("scenario_*.py"))
    
    for file_path in scenario_files:
        try:
            fix_scenario_config(file_path)
        except Exception as e:
            print(f"❌ Error fixing {file_path.name}: {e}")
    
    print(f"\n🎉 Fixed {len(scenario_files)} scenario configuration files!")

if __name__ == "__main__":
    main()