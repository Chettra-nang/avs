#!/usr/bin/env python3
"""Clean up unused collector initialization from scenario files"""

from pathlib import Path

def clean_scenario_file(file_path):
    """Remove unused collector initialization"""
    print(f"Cleaning {file_path.name}...")
    
    content = file_path.read_text()
    
    # Remove the unused collector creation block
    lines_to_remove = [
        "# Configure complete dataset collection",
        "logger.info(\"Configuring complete multimodal data collection\")",
        "# Configure complete dataset collection",
        "modality_manager = ModalityConfigManager()",
        "collector = SynchronizedCollector(",
        "    n_agents=collection_config[\"n_agents\"],", 
        "    modality_config_manager=modality_manager",
        ")"
    ]
    
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_blank = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip the collector initialization block
        if any(remove_line in line for remove_line in lines_to_remove):
            continue
            
        cleaned_lines.append(line)
    
    # Join back and clean up extra blank lines
    new_content = '\n'.join(cleaned_lines)
    new_content = '\n'.join(line for line in new_content.split('\n') if line.strip() or not line.strip())
    
    file_path.write_text(new_content)
    print(f"✅ Cleaned {file_path.name}")

def main():
    """Clean all scenario files"""
    collecting_data_dir = Path("/home/chettra/ITC/Research/AVs/collecting_data")
    
    scenario_files = list(collecting_data_dir.glob("scenario_*.py"))
    
    for file_path in scenario_files:
        try:
            clean_scenario_file(file_path)
        except Exception as e:
            print(f"❌ Error cleaning {file_path.name}: {e}")
    
    print(f"\n🎉 Cleaned {len(scenario_files)} scenario files!")

if __name__ == "__main__":
    main()