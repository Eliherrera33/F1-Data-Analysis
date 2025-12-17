# -*- coding: utf-8 -*-
"""
F1 Circuit & Track Analysis
Visualizations comparing circuit characteristics, lap records, and more.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "circuit_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_circuits():
    """Load circuit data."""
    return pd.read_csv(DATA_DIR / "reference" / "circuits.csv")


def load_race_results():
    """Load all race results to get circuit-specific data."""
    all_data = []
    for csv_file in sorted((DATA_DIR / "f1db_csv").glob("*_race_results.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_fastest_laps():
    """Load fastest lap data."""
    all_data = []
    for csv_file in sorted((DATA_DIR / "f1db_csv").glob("*_fastest_laps.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def plot_circuit_lengths(circuits):
    """Bar chart of circuit lengths."""
    # Sort by length
    circuits_sorted = circuits.sort_values('length', ascending=True)
    
    fig, ax = plt.subplots(figsize=(16, 20))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(circuits_sorted)))
    
    bars = ax.barh(circuits_sorted['name'], circuits_sorted['length'], color=colors)
    
    ax.set_xlabel('Circuit Length (km)', fontsize=12)
    ax.set_ylabel('Circuit', fontsize=12)
    ax.set_title('F1 Circuit Lengths Comparison', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, length in zip(bars, circuits_sorted['length']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{length:.2f} km', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuit_lengths.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuit_lengths.png")


def plot_circuit_turns(circuits):
    """Bar chart of number of turns per circuit."""
    circuits_sorted = circuits.sort_values('turns', ascending=True)
    
    fig, ax = plt.subplots(figsize=(16, 20))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(circuits_sorted)))
    
    bars = ax.barh(circuits_sorted['name'], circuits_sorted['turns'], color=colors)
    
    ax.set_xlabel('Number of Turns', fontsize=12)
    ax.set_ylabel('Circuit', fontsize=12)
    ax.set_title('F1 Circuits by Number of Turns', fontsize=16, fontweight='bold')
    
    for bar, turns in zip(bars, circuits_sorted['turns']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{int(turns)}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuit_turns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuit_turns.png")


def plot_circuit_types_pie(circuits):
    """Pie chart of circuit types."""
    type_counts = circuits['type'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    explode = [0.05 if t == type_counts.index[0] else 0 for t in type_counts.index]
    
    ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
           colors=colors[:len(type_counts)], explode=explode, shadow=True, startangle=90)
    
    ax.set_title('F1 Circuit Types Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuit_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuit_types.png")


def plot_circuits_by_country(circuits):
    """Bar chart of circuits per country."""
    country_counts = circuits['countryId'].value_counts().head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(country_counts)))
    
    bars = ax.bar(country_counts.index, country_counts.values, color=colors)
    
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Number of Circuits', fontsize=12)
    ax.set_title('F1 Circuits by Country (Top 20)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuits_by_country.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuits_by_country.png")


def plot_circuit_direction(circuits):
    """Pie chart of circuit direction (clockwise vs anti-clockwise)."""
    direction_counts = circuits['direction'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#00ff00', '#ff0000']
    
    ax.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%',
           colors=colors, shadow=True, startangle=90)
    
    ax.set_title('F1 Circuit Directions', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuit_directions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuit_directions.png")


def plot_length_vs_turns(circuits):
    """Scatter plot of circuit length vs number of turns."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by type
    types = circuits['type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = dict(zip(types, colors))
    
    for circuit_type in types:
        subset = circuits[circuits['type'] == circuit_type]
        ax.scatter(subset['length'], subset['turns'], 
                  c=[type_colors[circuit_type]], label=circuit_type, s=100, alpha=0.7)
    
    # Add annotations for some notable circuits
    notable = ['monaco', 'spa-francorchamps', 'monza', 'silverstone', 'suzuka']
    for _, row in circuits[circuits['id'].isin(notable)].iterrows():
        ax.annotate(row['name'], (row['length'], row['turns']), 
                   fontsize=9, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Circuit Length (km)', fontsize=12)
    ax.set_ylabel('Number of Turns', fontsize=12)
    ax.set_title('Circuit Length vs Number of Turns', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'length_vs_turns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: length_vs_turns.png")


def plot_circuit_world_map(circuits):
    """Plot circuits on a simple world map (scatter plot approximation)."""
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Color by type
    types = circuits['type'].unique()
    colors = {'RACE': '#00ff00', 'STREET': '#ff0000', 'ROAD': '#ffff00'}
    
    for circuit_type in types:
        subset = circuits[circuits['type'] == circuit_type]
        color = colors.get(circuit_type, '#0000ff')
        ax.scatter(subset['longitude'], subset['latitude'], 
                  c=color, label=circuit_type, s=80, alpha=0.7, edgecolors='white')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('F1 Circuit Locations Around the World', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-130, 160)
    ax.set_ylim(-50, 70)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circuit_world_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: circuit_world_map.png")


def plot_current_calendar_circuits(circuits, race_results):
    """Show circuits used in recent seasons."""
    # Get circuits used in 2024
    recent_races = race_results[race_results['year'] == 2024]['race'].unique()
    
    # Extract circuit names from race names
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # For simplicity, show circuit lengths for modern F1 circuits
    modern_circuits = ['monaco', 'spa-francorchamps', 'monza', 'silverstone', 'suzuka',
                      'austin', 'interlagos', 'melbourne', 'bahrain', 'jeddah',
                      'miami', 'las-vegas', 'baku', 'shanghai', 'marina-bay',
                      'hungaroring', 'zandvoort', 'lusail', 'yas-marina', 'montreal']
    
    modern_data = circuits[circuits['id'].isin(modern_circuits)].sort_values('length', ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(modern_data)))
    
    bars = ax.barh(modern_data['name'], modern_data['length'], color=colors)
    
    ax.set_xlabel('Circuit Length (km)', fontsize=12)
    ax.set_ylabel('Circuit', fontsize=12)
    ax.set_title('Modern F1 Calendar Circuit Lengths', fontsize=16, fontweight='bold')
    
    for bar, length in zip(bars, modern_data['length']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{length:.2f} km', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'modern_circuit_lengths.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: modern_circuit_lengths.png")


def plot_street_vs_permanent(circuits):
    """Compare street circuits vs permanent circuits."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Length comparison
    street = circuits[circuits['type'] == 'STREET']['length']
    permanent = circuits[circuits['type'] == 'RACE']['length']
    
    data = [street.dropna(), permanent.dropna()]
    labels = ['Street Circuits', 'Permanent Circuits']
    
    bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#4ecdc4')
    axes[0].set_ylabel('Circuit Length (km)', fontsize=12)
    axes[0].set_title('Circuit Length: Street vs Permanent', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Turns comparison
    street_turns = circuits[circuits['type'] == 'STREET']['turns']
    permanent_turns = circuits[circuits['type'] == 'RACE']['turns']
    
    data = [street_turns.dropna(), permanent_turns.dropna()]
    
    bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#4ecdc4')
    axes[1].set_ylabel('Number of Turns', fontsize=12)
    axes[1].set_title('Number of Turns: Street vs Permanent', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'street_vs_permanent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: street_vs_permanent.png")


def main():
    print("=" * 60)
    print("F1 CIRCUIT ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\nLoading circuit data...")
    circuits = load_circuits()
    race_results = load_race_results()
    
    print(f"Loaded {len(circuits)} circuits")
    
    # Generate visualizations
    print("\nGenerating circuit visualizations...")
    
    plot_circuit_lengths(circuits)
    plot_circuit_turns(circuits)
    plot_circuit_types_pie(circuits)
    plot_circuits_by_country(circuits)
    plot_circuit_direction(circuits)
    plot_length_vs_turns(circuits)
    plot_circuit_world_map(circuits)
    plot_current_calendar_circuits(circuits, race_results)
    plot_street_vs_permanent(circuits)
    
    print("\n" + "=" * 60)
    print("CIRCUIT ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
