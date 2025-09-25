#!/usr/bin/env python3
"""
Show Natural Language Data Structure

This script shows exactly how natural language is stored and used in your dataset.
"""

import pandas as pd
import json

def show_parquet_structure():
    """Show the structure of the Parquet file with natural language."""
    print("📊 Natural Language Data Structure in Parquet")
    print("=" * 50)
    
    # Load the demo data
    df = pd.read_parquet("data/demo_with_natural_language.parquet")
    
    print(f"📋 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📝 Columns: {list(df.columns)}")
    
    print("\n🔍 Sample Data Structure:")
    print("-" * 30)
    
    # Show first few rows
    for i in range(3):
        row = df.iloc[i]
        print(f"\nRow {i}:")
        print(f"  episode_id: {row['episode_id']}")
        print(f"  step: {row['step']}")
        print(f"  speed: {row['speed']:.1f} m/s")
        print(f"  traffic_density: {row['traffic_density']:.2f}")
        print(f"  scenario_type: {row['scenario_type']}")
        print(f"  summary_text: '{row['summary_text']}'")
    
    print("\n📊 Data Types:")
    print("-" * 15)
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    return df

def show_how_text_is_processed():
    """Show how natural language text is processed in the plotting code."""
    print("\n🔧 How Natural Language is Processed in Your Code")
    print("=" * 55)
    
    df = pd.read_parquet("data/demo_with_natural_language.parquet")
    
    print("1. 📝 Text Extraction (from plot_text_summaries function):")
    print("-" * 55)
    
    # Simulate the text extraction process from your plotter
    sample_indices = [0, len(df)//3, 2*len(df)//3, len(df)-1]
    summaries = []
    
    print("   Selected sample indices:", sample_indices)
    
    for idx in sample_indices:
        if idx < len(df) and 'summary_text' in df.columns:
            summary = df.iloc[idx]['summary_text']
            if pd.notna(summary) and summary.strip():
                step = df.iloc[idx]['step']
                formatted_summary = f"Step {step}: {summary[:100]}..."
                summaries.append(formatted_summary)
                print(f"   Index {idx} -> {formatted_summary}")
    
    print("\n2. 📊 Text Display (how it appears in plots):")
    print("-" * 45)
    
    if summaries:
        combined_text = "\n\n".join(summaries)
        print("   Combined text for plot display:")
        print(f"   '{combined_text}'")
    
    print("\n3. 🔍 Text Analysis (word frequency, patterns):")
    print("-" * 50)
    
    all_summaries = df['summary_text'].tolist()
    
    # Word frequency
    all_words = []
    for summary in all_summaries:
        words = summary.lower().split()
        all_words.extend(words)
    
    from collections import Counter
    word_freq = Counter(all_words)
    
    print("   Top 10 most common words:")
    for word, count in word_freq.most_common(10):
        print(f"     '{word}': {count} times")
    
    # Pattern analysis
    safety_words = ['safe', 'collision', 'distance', 'risk', 'following']
    speed_words = ['speed', 'fast', 'slow', 'accelerating', 'reducing']
    traffic_words = ['traffic', 'congestion', 'vehicles', 'density']
    
    safety_count = sum(1 for s in all_summaries 
                      for word in safety_words 
                      if word in s.lower())
    
    speed_count = sum(1 for s in all_summaries 
                     for word in speed_words 
                     if word in s.lower())
    
    traffic_count = sum(1 for s in all_summaries 
                       for word in traffic_words 
                       if word in s.lower())
    
    print(f"\n   Pattern Analysis:")
    print(f"     Safety-related mentions: {safety_count}")
    print(f"     Speed-related mentions: {speed_count}")
    print(f"     Traffic-related mentions: {traffic_count}")

def show_integration_with_other_data():
    """Show how natural language integrates with numerical and image data."""
    print("\n🔗 Integration with Other Multimodal Data")
    print("=" * 45)
    
    df = pd.read_parquet("data/demo_with_natural_language.parquet")
    
    print("📊 How Text Relates to Numerical Data:")
    print("-" * 40)
    
    for i in range(0, len(df), 5):  # Every 5th row
        row = df.iloc[i]
        print(f"\nStep {row['step']}:")
        print(f"  📈 Numerical: Speed={row['speed']:.1f}, Traffic={row['traffic_density']:.2f}")
        print(f"  📝 Text: '{row['summary_text']}'")
        print(f"  🎭 Scenario: {row['scenario_type']}")
        
        # Show correlation
        if 'heavy traffic' in row['summary_text'].lower():
            print(f"  ✅ Text matches data: High traffic density ({row['traffic_density']:.2f})")
        elif 'light traffic' in row['summary_text'].lower():
            print(f"  ✅ Text matches data: Low traffic density ({row['traffic_density']:.2f})")

def show_real_world_usage():
    """Show how this would work with real highway data."""
    print("\n🚗 Real-World Usage in Highway Dataset")
    print("=" * 40)
    
    print("""
🎯 In your actual highway dataset, natural language summaries would:

1. 📝 DESCRIBE DRIVING CONTEXT:
   • "Vehicle merging onto highway from on-ramp"
   • "Approaching construction zone, reducing speed"
   • "Lane blocked ahead, vehicles changing lanes"

2. 🛡️  EXPLAIN SAFETY DECISIONS:
   • "Emergency braking due to sudden obstacle"
   • "Increasing following distance in wet conditions"
   • "Avoiding aggressive driver in adjacent lane"

3. 🚦 PROVIDE TRAFFIC CONTEXT:
   • "Rush hour congestion, stop-and-go traffic"
   • "School zone, reduced speed limit in effect"
   • "Highway patrol vehicle spotted, traffic slowing"

4. 🎭 SCENARIO-SPECIFIC DETAILS:
   • Free Flow: "Cruise control engaged, maintaining 65 mph"
   • Dense Commuting: "Frequent lane changes due to varying speeds"
   • Stop-and-Go: "Traffic jam, 15-minute delay expected"

5. 🔗 BRIDGE DATA AND UNDERSTANDING:
   • Links numerical data (speed, TTC) to human-readable context
   • Explains WHY certain actions were taken
   • Provides context that sensors alone cannot capture
    """)

def show_code_integration():
    """Show exactly where natural language fits in your plotting code."""
    print("\n💻 Code Integration Points")
    print("=" * 30)
    
    print("""
🔧 In your multimodal_parquet_plotter.py:

1. 📊 DATA LOADING:
   ```python
   df = pd.read_parquet('episode_transitions.parquet')
   # df['summary_text'] contains natural language summaries
   ```

2. 📝 TEXT EXTRACTION (plot_text_summaries function):
   ```python
   def plot_text_summaries(self, df, ax):
       # Get sample summaries from different time points
       sample_indices = [0, len(df)//3, 2*len(df)//3, len(df)-1]
       summaries = []
       
       for idx in sample_indices:
           summary = df.iloc[idx]['summary_text']
           step = df.iloc[idx]['step']
           summaries.append(f"Step {step}: {summary[:100]}...")
       
       # Display in plot
       summary_text = "\\n\\n".join(summaries)
       ax.text(0.05, 0.95, summary_text, ...)
   ```

3. 🔍 TEXT ANALYSIS:
   ```python
   # Word frequency analysis
   all_words = []
   for summary in df['summary_text']:
       words = summary.lower().split()
       all_words.extend(words)
   
   word_freq = Counter(all_words)
   ```

4. 📊 VISUALIZATION:
   • Text appears in dedicated subplot
   • Shows context for numerical data
   • Helps interpret driving behavior
    """)

def main():
    """Main function to demonstrate natural language processing."""
    df = show_parquet_structure()
    show_how_text_is_processed()
    show_integration_with_other_data()
    show_real_world_usage()
    show_code_integration()
    
    print("\n✅ Natural Language Summary:")
    print("=" * 30)
    print("• Natural language is stored as 'summary_text' column in Parquet")
    print("• Text provides human-readable context for each driving step")
    print("• Your plotting code extracts and displays sample summaries")
    print("• Text analysis reveals patterns in driving behavior")
    print("• Integrates seamlessly with numerical and image data")
    print("• Essential for understanding WHY actions were taken")

if __name__ == "__main__":
    main()