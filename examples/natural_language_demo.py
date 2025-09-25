#!/usr/bin/env python3
"""
Natural Language Processing Demo for Multimodal Dataset

This demonstrates exactly how natural language summaries work in your 
multimodal highway dataset, including generation, storage, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter
import re

class NaturalLanguageProcessor:
    """Handles natural language processing for highway driving scenarios."""
    
    def __init__(self):
        # Define driving context templates
        self.driving_contexts = {
            'speed_changes': [
                "Vehicle accelerating to {speed:.1f} m/s due to clear road ahead",
                "Reducing speed to {speed:.1f} m/s due to traffic congestion",
                "Maintaining steady speed of {speed:.1f} m/s in current traffic",
                "Gradual deceleration to {speed:.1f} m/s approaching intersection"
            ],
            'lane_changes': [
                "Considering lane change due to slower vehicle ahead",
                "Executing lane change to overtake slower traffic",
                "Staying in current lane, maintaining safe following distance",
                "Lane change completed, now in faster moving lane"
            ],
            'traffic_conditions': [
                "Light traffic detected, optimal driving conditions",
                "Moderate traffic density, adjusting speed accordingly", 
                "Heavy traffic congestion, frequent stop-and-go movement",
                "Dense traffic with {vehicle_count} vehicles in vicinity"
            ],
            'safety_events': [
                "Time-to-collision at {ttc:.1f}s, maintaining safe distance",
                "Close following detected, increasing gap to lead vehicle",
                "Emergency braking avoided, safe distance restored",
                "Potential collision risk detected, taking evasive action"
            ],
            'scenario_specific': {
                'free_flow': [
                    "Free-flowing traffic, maintaining highway speed",
                    "Optimal conditions for cruise control engagement",
                    "Clear visibility, no obstacles detected ahead"
                ],
                'dense_commuting': [
                    "Rush hour traffic, frequent speed adjustments needed",
                    "Multiple lane changes required due to congestion",
                    "Stop-and-go pattern typical of commuter traffic"
                ],
                'stop_and_go': [
                    "Traffic jam conditions, frequent complete stops",
                    "Crawling speed due to bottleneck ahead",
                    "Patience required, maintaining minimal following distance"
                ]
            }
        }
    
    def generate_contextual_summary(self, row, scenario_type='free_flow'):
        """Generate a contextual natural language summary for a driving step."""
        
        # Extract key metrics
        speed = row.get('speed', 20.0)
        ttc = row.get('ttc', 5.0)
        traffic_density = row.get('traffic_density', 0.5)
        vehicle_count = row.get('vehicle_count', 5)
        step = row.get('step', 0)
        
        # Choose summary type based on conditions
        summaries = []
        
        # Speed-based context
        if speed > 25:
            template = np.random.choice(self.driving_contexts['speed_changes'][:2])
        elif speed < 10:
            template = np.random.choice(self.driving_contexts['speed_changes'][1:3])
        else:
            template = np.random.choice(self.driving_contexts['speed_changes'])
        
        try:
            summaries.append(template.format(speed=speed, ttc=ttc, vehicle_count=vehicle_count))
        except:
            summaries.append(template)
        
        # Traffic-based context
        if traffic_density > 0.7:
            template = self.driving_contexts['traffic_conditions'][2]
        elif traffic_density > 0.4:
            template = self.driving_contexts['traffic_conditions'][1]
        else:
            template = self.driving_contexts['traffic_conditions'][0]
        
        try:
            summaries.append(template.format(vehicle_count=vehicle_count))
        except:
            summaries.append(template)
        
        # Safety-based context
        if ttc < 3.0:
            template = np.random.choice(self.driving_contexts['safety_events'][:2])
            try:
                summaries.append(template.format(ttc=ttc))
            except:
                summaries.append(template)
        
        # Scenario-specific context
        if scenario_type in self.driving_contexts['scenario_specific']:
            scenario_template = np.random.choice(
                self.driving_contexts['scenario_specific'][scenario_type]
            )
            summaries.append(scenario_template)
        
        # Combine and return
        if summaries:
            return np.random.choice(summaries)
        else:
            return f"Vehicle operating at {speed:.1f} m/s in {scenario_type.replace('_', ' ')} conditions"
    
    def analyze_text_patterns(self, summaries):
        """Analyze patterns in natural language summaries."""
        analysis = {
            'total_summaries': len(summaries),
            'word_frequency': {},
            'common_phrases': {},
            'sentiment_indicators': {},
            'driving_actions': {},
            'safety_mentions': 0,
            'speed_mentions': 0,
            'traffic_mentions': 0
        }
        
        # Word frequency analysis
        all_words = []
        for summary in summaries:
            words = re.findall(r'\b\w+\b', summary.lower())
            all_words.extend(words)
        
        analysis['word_frequency'] = dict(Counter(all_words).most_common(20))
        
        # Phrase analysis
        all_phrases = []
        for summary in summaries:
            # Extract 2-word phrases
            words = summary.lower().split()
            phrases = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_phrases.extend(phrases)
        
        analysis['common_phrases'] = dict(Counter(all_phrases).most_common(10))
        
        # Driving action analysis
        action_keywords = {
            'acceleration': ['accelerating', 'speeding up', 'increasing speed'],
            'deceleration': ['reducing speed', 'slowing down', 'braking'],
            'lane_change': ['lane change', 'changing lanes', 'overtake'],
            'following': ['following', 'maintaining distance', 'behind vehicle']
        }
        
        for action, keywords in action_keywords.items():
            count = sum(1 for summary in summaries 
                       for keyword in keywords 
                       if keyword in summary.lower())
            analysis['driving_actions'][action] = count
        
        # Safety, speed, and traffic mentions
        analysis['safety_mentions'] = sum(1 for s in summaries 
                                        if any(word in s.lower() 
                                             for word in ['safe', 'collision', 'distance', 'risk']))
        
        analysis['speed_mentions'] = sum(1 for s in summaries 
                                       if any(word in s.lower() 
                                            for word in ['speed', 'fast', 'slow', 'm/s']))
        
        analysis['traffic_mentions'] = sum(1 for s in summaries 
                                         if any(word in s.lower() 
                                              for word in ['traffic', 'congestion', 'vehicles']))
        
        return analysis

def demonstrate_natural_language_generation():
    """Show how natural language summaries are generated."""
    print("🗣️  Natural Language Generation Demo")
    print("=" * 40)
    
    nlp = NaturalLanguageProcessor()
    
    # Create sample driving scenarios
    scenarios = {
        'free_flow': {
            'speed': 28.5,
            'ttc': 8.2,
            'traffic_density': 0.2,
            'vehicle_count': 3,
            'step': 45
        },
        'dense_commuting': {
            'speed': 15.3,
            'ttc': 2.8,
            'traffic_density': 0.8,
            'vehicle_count': 12,
            'step': 78
        },
        'stop_and_go': {
            'speed': 5.1,
            'ttc': 1.5,
            'traffic_density': 0.9,
            'vehicle_count': 18,
            'step': 120
        }
    }
    
    print("\n📝 Generated Natural Language Summaries:")
    print("-" * 45)
    
    for scenario_name, data in scenarios.items():
        print(f"\n🎭 {scenario_name.replace('_', ' ').title()} Scenario:")
        print(f"   📊 Data: Speed={data['speed']:.1f} m/s, TTC={data['ttc']:.1f}s, "
              f"Traffic={data['traffic_density']:.1f}, Vehicles={data['vehicle_count']}")
        
        # Generate multiple summaries for this scenario
        for i in range(3):
            summary = nlp.generate_contextual_summary(data, scenario_name)
            print(f"   {i+1}. {summary}")

def demonstrate_text_storage_in_parquet():
    """Show how text is stored in Parquet format."""
    print("\n💾 Text Storage in Parquet Demo")
    print("=" * 35)
    
    nlp = NaturalLanguageProcessor()
    
    # Create sample episode data with natural language
    episode_data = []
    
    for step in range(20):
        # Simulate changing driving conditions
        if step < 7:
            scenario = 'free_flow'
            speed = 25 + np.random.normal(0, 2)
            traffic_density = 0.2 + np.random.uniform(0, 0.2)
        elif step < 14:
            scenario = 'dense_commuting'
            speed = 15 + np.random.normal(0, 3)
            traffic_density = 0.6 + np.random.uniform(0, 0.3)
        else:
            scenario = 'stop_and_go'
            speed = 8 + np.random.normal(0, 2)
            traffic_density = 0.8 + np.random.uniform(0, 0.2)
        
        row_data = {
            'episode_id': 'demo_episode_001',
            'step': step,
            'speed': max(0, speed),
            'ttc': np.random.exponential(4),
            'traffic_density': min(1.0, max(0.0, traffic_density)),
            'vehicle_count': int(np.random.randint(2, 20)),
            'ego_x': 100 + step * 10,
            'ego_y': 50 + np.sin(step * 0.2) * 3
        }
        
        # Generate natural language summary
        summary = nlp.generate_contextual_summary(row_data, scenario)
        row_data['summary_text'] = summary
        row_data['scenario_type'] = scenario
        
        episode_data.append(row_data)
    
    # Create DataFrame and save as Parquet
    df = pd.DataFrame(episode_data)
    
    # Save to demonstrate Parquet storage
    output_path = Path("data/demo_with_natural_language.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Created demo Parquet file: {output_path}")
    print(f"📊 Contains {len(df)} steps with natural language summaries")
    
    # Show sample data
    print("\n📋 Sample Data with Natural Language:")
    print("-" * 40)
    
    for i in range(0, len(df), 5):  # Show every 5th step
        row = df.iloc[i]
        print(f"\nStep {row['step']} ({row['scenario_type']}):")
        print(f"   Speed: {row['speed']:.1f} m/s, Traffic: {row['traffic_density']:.2f}")
        print(f"   Summary: {row['summary_text']}")
    
    return df

def demonstrate_text_analysis():
    """Show how to analyze natural language summaries."""
    print("\n🔍 Natural Language Analysis Demo")
    print("=" * 35)
    
    # Load the demo data
    df = pd.read_parquet("data/demo_with_natural_language.parquet")
    
    nlp = NaturalLanguageProcessor()
    summaries = df['summary_text'].tolist()
    
    # Perform analysis
    analysis = nlp.analyze_text_patterns(summaries)
    
    print(f"📊 Analysis Results for {analysis['total_summaries']} summaries:")
    print("-" * 50)
    
    print("\n🔤 Most Common Words:")
    for word, count in list(analysis['word_frequency'].items())[:10]:
        print(f"   {word}: {count}")
    
    print("\n📝 Common Phrases:")
    for phrase, count in analysis['common_phrases'].items():
        print(f"   '{phrase}': {count}")
    
    print("\n🚗 Driving Actions Mentioned:")
    for action, count in analysis['driving_actions'].items():
        print(f"   {action.replace('_', ' ').title()}: {count}")
    
    print(f"\n🛡️  Safety mentions: {analysis['safety_mentions']}")
    print(f"⚡ Speed mentions: {analysis['speed_mentions']}")
    print(f"🚦 Traffic mentions: {analysis['traffic_mentions']}")

def demonstrate_text_visualization():
    """Show how natural language is visualized."""
    print("\n📊 Natural Language Visualization Demo")
    print("=" * 40)
    
    # Load the demo data
    df = pd.read_parquet("data/demo_with_natural_language.parquet")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Natural Language Analysis Visualization', fontsize=14, fontweight='bold')
    
    # 1. Text summaries over time
    ax = axes[0, 0]
    ax.set_title("Driving Summaries Timeline")
    
    # Color code by scenario
    scenario_colors = {'free_flow': 'green', 'dense_commuting': 'orange', 'stop_and_go': 'red'}
    
    for i, row in df.iterrows():
        color = scenario_colors.get(row['scenario_type'], 'blue')
        ax.scatter(row['step'], i, c=color, s=50, alpha=0.7)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Summary Index")
    ax.grid(True, alpha=0.3)
    
    # Add legend
    for scenario, color in scenario_colors.items():
        ax.scatter([], [], c=color, label=scenario.replace('_', ' ').title())
    ax.legend()
    
    # 2. Word frequency
    ax = axes[0, 1]
    nlp = NaturalLanguageProcessor()
    analysis = nlp.analyze_text_patterns(df['summary_text'].tolist())
    
    words = list(analysis['word_frequency'].keys())[:8]
    counts = list(analysis['word_frequency'].values())[:8]
    
    bars = ax.bar(words, counts, color='skyblue', alpha=0.8)
    ax.set_title("Most Common Words")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(count), ha='center', va='bottom')
    
    # 3. Driving actions
    ax = axes[1, 0]
    actions = list(analysis['driving_actions'].keys())
    action_counts = list(analysis['driving_actions'].values())
    
    bars = ax.bar(actions, action_counts, color='lightcoral', alpha=0.8)
    ax.set_title("Driving Actions Mentioned")
    ax.set_ylabel("Mentions")
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Sample text display
    ax = axes[1, 1]
    ax.set_title("Sample Natural Language Summaries")
    
    # Show sample summaries
    sample_summaries = df['summary_text'].iloc[::4].tolist()[:5]  # Every 4th summary
    sample_text = "\n\n".join([f"{i+1}. {summary}" for i, summary in enumerate(sample_summaries)])
    
    ax.text(0.05, 0.95, sample_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', wrap=True,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"natural_language_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Natural language analysis saved as: {filename}")
    
    plt.show()

def main():
    """Main demonstration of natural language processing."""
    print("🗣️  Natural Language Processing in Multimodal Dataset")
    print("=" * 55)
    
    print("\n🎯 This demo shows exactly how natural language works in your code:")
    print("1. How summaries are generated from driving data")
    print("2. How text is stored in Parquet format")
    print("3. How to analyze and visualize text data")
    print("4. How text integrates with other multimodal data")
    
    # Run demonstrations
    demonstrate_natural_language_generation()
    df = demonstrate_text_storage_in_parquet()
    demonstrate_text_analysis()
    demonstrate_text_visualization()
    
    print("\n✅ Natural Language Processing Demo Complete!")
    print("\n🔑 Key Takeaways:")
    print("• Natural language summaries provide rich context for each driving step")
    print("• Text is stored efficiently in Parquet alongside numerical data")
    print("• Text analysis reveals patterns in driving behavior descriptions")
    print("• Natural language bridges the gap between raw data and human understanding")
    print("• Summaries can be generated automatically or provided by human annotators")
    
    print(f"\n📁 Files created:")
    print("• data/demo_with_natural_language.parquet - Demo dataset with text")
    print("• natural_language_analysis_[timestamp].png - Visualization")

if __name__ == "__main__":
    main()