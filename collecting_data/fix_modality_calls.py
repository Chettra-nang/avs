#!/usr/bin/env python3
"""Fix all scenario files to pass modality_config_manager"""

from pathlib import Path

def fix_scenario_modality(file_path):
    """Fix run_full_collection call to include modality_config_manager"""
    print(f"Fixing {file_path.name}...")
    
    content = file_path.read_text()
    
    # Replace the function call
    old_call = 'result = run_full_collection(**collection_config)'
    new_call = 'result = run_full_collection(modality_config_manager=modality_manager, **collection_config)'
    
    if old_call in content:
        content = content.replace(old_call, new_call)
        file_path.write_text(content)
        print(f"✅ Fixed {file_path.name}")
    else:
        print(f"⚠️  {file_path.name} - pattern not found")

def main():
    """Fix all scenario files"""
    collecting_data_dir = Path("/home/chettra/ITC/Research/AVs/collecting_data")
    
    scenario_files = list(collecting_data_dir.glob("scenario_*.py"))
    
    for file_path in scenario_files:
        try:
            fix_scenario_modality(file_path)
        except Exception as e:
            print(f"❌ Error fixing {file_path.name}: {e}")
    
    print(f"\n🎉 Updated {len(scenario_files)} scenario files!")

if __name__ == "__main__":
    main()