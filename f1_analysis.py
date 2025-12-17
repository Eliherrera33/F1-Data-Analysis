# -*- coding: utf-8 -*-
"""
F1 Data Analysis & Visualization
Comprehensive analysis of F1 data from 2012-2025 with various charts and comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Directories
BASE_DIR = Path(__file__).parent
F1DB_CSV_DIR = BASE_DIR / "data" / "f1db_csv"
FORMULA1_DATASETS_DIR = BASE_DIR / "formula1-datasets"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_all_race_results():
    """Load all race results from f1db CSVs."""
    all_data = []
    for csv_file in sorted(F1DB_CSV_DIR.glob("*_race_results.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_all_driver_standings():
    """Load all driver standings from f1db CSVs."""
    all_data = []
    for csv_file in sorted(F1DB_CSV_DIR.glob("*_driver_standings.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_all_constructor_standings():
    """Load all constructor standings from f1db CSVs."""
    all_data = []
    for csv_file in sorted(F1DB_CSV_DIR.glob("*_constructor_standings.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_formula1_datasets_results():
    """Load race results from formula1-datasets."""
    all_data = []
    
    # Current seasons
    for csv_file in FORMULA1_DATASETS_DIR.glob("*_raceResults.csv"):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    for csv_file in FORMULA1_DATASETS_DIR.glob("*_RaceResults.csv"):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    # Previous seasons
    prev_dir = FORMULA1_DATASETS_DIR / "PreviousSeasons"
    if prev_dir.exists():
        for csv_file in prev_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_championship_winners(df_standings):
    """Extract championship winners per year."""
    if df_standings.empty:
        return pd.DataFrame()
    
    # Get final race standings for each year
    final_standings = df_standings.groupby('year').apply(
        lambda x: x[x['race'] == x['race'].max()]
    ).reset_index(drop=True)
    
    # Get position 1
    winners = final_standings[final_standings['position'] == 1]
    return winners


def calculate_points_per_season(df_results):
    """Calculate total points per driver per season."""
    if df_results.empty or 'points' not in df_results.columns:
        return pd.DataFrame()
    
    points_df = df_results.groupby(['year', 'driverId'])['points'].sum().reset_index()
    points_df = points_df.sort_values(['year', 'points'], ascending=[True, False])
    return points_df


def calculate_constructor_points(df_results):
    """Calculate constructor points per season."""
    if df_results.empty or 'points' not in df_results.columns:
        return pd.DataFrame()
    
    points_df = df_results.groupby(['year', 'constructorId'])['points'].sum().reset_index()
    points_df = points_df.sort_values(['year', 'points'], ascending=[True, False])
    return points_df


def count_wins_per_driver(df_results):
    """Count wins per driver across all seasons."""
    if df_results.empty:
        return pd.DataFrame()
    
    # Handle position that might be string or numeric
    df_copy = df_results.copy()
    df_copy['pos_numeric'] = pd.to_numeric(df_copy['position'], errors='coerce')
    wins = df_copy[df_copy['pos_numeric'] == 1].groupby('driverId').size().reset_index(name='wins')
    wins = wins.sort_values('wins', ascending=False)
    return wins


def count_podiums_per_driver(df_results):
    """Count podiums per driver across all seasons."""
    if df_results.empty:
        return pd.DataFrame()
    
    # Handle position that might be string or numeric
    df_copy = df_results.copy()
    df_copy['pos_numeric'] = pd.to_numeric(df_copy['position'], errors='coerce')
    podiums = df_copy[df_copy['pos_numeric'] <= 3].groupby('driverId').size().reset_index(name='podiums')
    podiums = podiums.sort_values('podiums', ascending=False)
    return podiums


def count_wins_per_constructor(df_results):
    """Count wins per constructor across all seasons."""
    if df_results.empty:
        return pd.DataFrame()
    
    # Handle position that might be string or numeric
    df_copy = df_results.copy()
    df_copy['pos_numeric'] = pd.to_numeric(df_copy['position'], errors='coerce')
    wins = df_copy[df_copy['pos_numeric'] == 1].groupby('constructorId').size().reset_index(name='wins')
    wins = wins.sort_values('wins', ascending=False)
    return wins


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_driver_wins_bar(df_results, top_n=20):
    """Bar chart of drivers with most wins."""
    wins = count_wins_per_driver(df_results)
    if wins.empty:
        return
    
    top_winners = wins.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_winners)))
    
    bars = ax.barh(top_winners['driverId'], top_winners['wins'], color=colors)
    ax.set_xlabel('Number of Wins', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'Top {top_n} F1 Drivers by Race Wins (2012-2025)', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, wins_count in zip(bars, top_winners['wins']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{wins_count}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'driver_wins_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: driver_wins_bar.png")


def plot_constructor_wins_bar(df_results, top_n=15):
    """Bar chart of constructors with most wins."""
    wins = count_wins_per_constructor(df_results)
    if wins.empty:
        return
    
    top_winners = wins.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(top_winners)))
    
    bars = ax.barh(top_winners['constructorId'], top_winners['wins'], color=colors)
    ax.set_xlabel('Number of Wins', fontsize=12)
    ax.set_ylabel('Constructor', fontsize=12)
    ax.set_title(f'Top {top_n} F1 Constructors by Race Wins (2012-2025)', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, wins_count in zip(bars, top_winners['wins']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{wins_count}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'constructor_wins_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: constructor_wins_bar.png")


def plot_driver_podiums_bar(df_results, top_n=20):
    """Bar chart of drivers with most podiums."""
    podiums = count_podiums_per_driver(df_results)
    if podiums.empty:
        return
    
    top_podiums = podiums.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.cool(np.linspace(0, 1, len(top_podiums)))
    
    bars = ax.barh(top_podiums['driverId'], top_podiums['podiums'], color=colors)
    ax.set_xlabel('Number of Podiums', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'Top {top_n} F1 Drivers by Podium Finishes (2012-2025)', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, pod_count in zip(bars, top_podiums['podiums']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{pod_count}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'driver_podiums_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: driver_podiums_bar.png")


def plot_points_by_year_line(df_results, top_n=10):
    """Line chart of top drivers' points progression over years."""
    points = calculate_points_per_season(df_results)
    if points.empty:
        return
    
    # Get top drivers overall
    top_drivers = points.groupby('driverId')['points'].sum().nlargest(top_n).index.tolist()
    
    # Filter for top drivers
    filtered = points[points['driverId'].isin(top_drivers)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for driver in top_drivers:
        driver_data = filtered[filtered['driverId'] == driver]
        ax.plot(driver_data['year'], driver_data['points'], marker='o', linewidth=2, 
                markersize=6, label=driver.replace('-', ' ').title())
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Points', fontsize=12)
    ax.set_title(f'Top {top_n} Drivers Points Progression (2012-2025)', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'driver_points_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: driver_points_progression.png")


def plot_constructor_points_by_year(df_results, top_n=8):
    """Stacked area chart of constructor points over years."""
    points = calculate_constructor_points(df_results)
    if points.empty:
        return
    
    # Get top constructors overall
    top_constructors = points.groupby('constructorId')['points'].sum().nlargest(top_n).index.tolist()
    
    # Pivot for stacked area
    filtered = points[points['constructorId'].isin(top_constructors)]
    pivot = filtered.pivot(index='year', columns='constructorId', values='points').fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Points', fontsize=12)
    ax.set_title(f'Constructor Points Distribution by Year (2012-2025)', fontsize=16, fontweight='bold')
    ax.legend(title='Constructor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'constructor_points_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: constructor_points_stacked.png")


def plot_wins_heatmap(df_results):
    """Heatmap of wins by driver and year."""
    if df_results.empty:
        return
    
    df_copy = df_results.copy()
    df_copy['pos_numeric'] = pd.to_numeric(df_copy['position'], errors='coerce')
    wins = df_copy[df_copy['pos_numeric'] == 1].groupby(['year', 'driverId']).size().reset_index(name='wins')
    
    if wins.empty:
        print("  No wins data for heatmap")
        return
    
    # Get top 15 drivers by total wins
    top_drivers = wins.groupby('driverId')['wins'].sum().nlargest(15).index.tolist()
    filtered = wins[wins['driverId'].isin(top_drivers)]
    
    if filtered.empty:
        print("  No filtered wins data for heatmap")
        return
    
    pivot = filtered.pivot(index='driverId', columns='year', values='wins').fillna(0)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'Wins'})
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title('Driver Wins Heatmap by Year (2012-2025)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'driver_wins_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: driver_wins_heatmap.png")


def plot_constructor_wins_heatmap(df_results):
    """Heatmap of wins by constructor and year."""
    if df_results.empty:
        return
    
    df_copy = df_results.copy()
    df_copy['pos_numeric'] = pd.to_numeric(df_copy['position'], errors='coerce')
    wins = df_copy[df_copy['pos_numeric'] == 1].groupby(['year', 'constructorId']).size().reset_index(name='wins')
    
    if wins.empty:
        print("  No constructor wins data for heatmap")
        return
    
    pivot = wins.pivot(index='constructorId', columns='year', values='wins').fillna(0)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, annot=True, fmt='g', cmap='Blues', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'Wins'})
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Constructor', fontsize=12)
    ax.set_title('Constructor Wins Heatmap by Year (2012-2025)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'constructor_wins_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: constructor_wins_heatmap.png")


def plot_position_distribution(df_results, driver_id):
    """Pie chart of finishing positions for a specific driver."""
    if df_results.empty:
        return
    
    driver_data = df_results[df_results['driverId'] == driver_id]
    if driver_data.empty:
        print(f"  No data for driver: {driver_id}")
        return
    
    driver_data['pos_numeric'] = pd.to_numeric(driver_data['position'], errors='coerce')
    
    # Categorize positions
    def categorize_position(pos):
        if pd.isna(pos):
            return 'DNF/DNS'
        elif pos == 1:
            return 'Win'
        elif pos <= 3:
            return 'Podium (2-3)'
        elif pos <= 10:
            return 'Points (4-10)'
        else:
            return 'Outside Points'
    
    driver_data['category'] = driver_data['pos_numeric'].apply(categorize_position)
    category_counts = driver_data['category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['gold', 'silver', 'limegreen', 'dodgerblue', 'gray']
    explode = [0.05 if cat == 'Win' else 0 for cat in category_counts.index]
    
    ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           colors=colors[:len(category_counts)], explode=explode, shadow=True, startangle=90)
    ax.set_title(f"Race Finish Distribution: {driver_id.replace('-', ' ').title()} (2012-2025)", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    safe_name = driver_id.replace('-', '_')
    plt.savefig(OUTPUT_DIR / f'position_dist_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: position_dist_{safe_name}.png")


def plot_races_per_year(df_results):
    """Bar chart showing number of races per year."""
    if df_results.empty:
        return
    
    races_per_year = df_results.groupby('year')['race'].nunique().reset_index(name='races')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(races_per_year)))
    
    bars = ax.bar(races_per_year['year'], races_per_year['races'], color=colors)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Races', fontsize=12)
    ax.set_title('F1 Races Per Season (2012-2025)', fontsize=16, fontweight='bold')
    ax.set_xticks(races_per_year['year'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'races_per_year.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: races_per_year.png")


def plot_dnf_rate_by_constructor(df_results):
    """Bar chart of DNF rates by constructor."""
    if df_results.empty or 'reasonRetired' not in df_results.columns:
        return
    
    # Count total races and DNFs per constructor
    total_races = df_results.groupby('constructorId').size()
    dnfs = df_results[df_results['reasonRetired'].notna() & (df_results['reasonRetired'] != '')].groupby('constructorId').size()
    
    dnf_rate = (dnfs / total_races * 100).fillna(0).sort_values(ascending=False)
    dnf_rate = dnf_rate.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(dnf_rate)))
    
    bars = ax.barh(dnf_rate.index, dnf_rate.values, color=colors)
    ax.set_xlabel('DNF Rate (%)', fontsize=12)
    ax.set_ylabel('Constructor', fontsize=12)
    ax.set_title('DNF Rate by Constructor (2012-2025)', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, rate in zip(bars, dnf_rate.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dnf_rate_by_constructor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: dnf_rate_by_constructor.png")


def plot_championship_dominance(df_results):
    """Show constructor championship dominance by percentage of total points."""
    if df_results.empty:
        return
    
    points = df_results.groupby(['year', 'constructorId'])['points'].sum().reset_index()
    total_per_year = points.groupby('year')['points'].sum()
    
    points['percentage'] = points.apply(
        lambda x: x['points'] / total_per_year[x['year']] * 100 if total_per_year[x['year']] > 0 else 0, 
        axis=1
    )
    
    # Get top 6 constructors
    top_constructors = points.groupby('constructorId')['points'].sum().nlargest(6).index.tolist()
    filtered = points[points['constructorId'].isin(top_constructors)]
    
    pivot = filtered.pivot(index='year', columns='constructorId', values='percentage').fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentage of Total Points', fontsize=12)
    ax.set_title('Constructor Championship Dominance (2012-2025)', fontsize=16, fontweight='bold')
    ax.legend(title='Constructor', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(pivot.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'championship_dominance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: championship_dominance.png")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary_stats(df_results):
    """Generate and save summary statistics."""
    if df_results.empty:
        return
    
    stats = []
    
    # Total races
    total_races = df_results.groupby(['year', 'race']).ngroup().max() + 1
    stats.append(f"Total Races Analyzed: {total_races}")
    
    # Unique drivers
    unique_drivers = df_results['driverId'].nunique()
    stats.append(f"Unique Drivers: {unique_drivers}")
    
    # Unique constructors
    unique_constructors = df_results['constructorId'].nunique()
    stats.append(f"Unique Constructors: {unique_constructors}")
    
    # Most successful driver
    wins = count_wins_per_driver(df_results)
    if not wins.empty:
        top_driver = wins.iloc[0]
        stats.append(f"Most Wins: {top_driver['driverId']} ({top_driver['wins']} wins)")
    
    # Most successful constructor
    constructor_wins = count_wins_per_constructor(df_results)
    if not constructor_wins.empty:
        top_constructor = constructor_wins.iloc[0]
        stats.append(f"Most Constructor Wins: {top_constructor['constructorId']} ({top_constructor['wins']} wins)")
    
    # Save stats
    with open(OUTPUT_DIR / 'summary_stats.txt', 'w') as f:
        f.write("F1 DATA ANALYSIS SUMMARY (2012-2025)\n")
        f.write("=" * 50 + "\n\n")
        for stat in stats:
            f.write(stat + "\n")
    
    print("  Saved: summary_stats.txt")
    return stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("F1 DATA ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df_results = load_all_race_results()
    
    if df_results.empty:
        print("No data found! Please run convert_yaml_to_csv.py first.")
        return
    
    print(f"Loaded {len(df_results)} race result entries")
    print(f"Years: {df_results['year'].min()} - {df_results['year'].max()}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_driver_wins_bar(df_results)
    plot_constructor_wins_bar(df_results)
    plot_driver_podiums_bar(df_results)
    plot_points_by_year_line(df_results)
    plot_constructor_points_by_year(df_results)
    plot_wins_heatmap(df_results)
    plot_constructor_wins_heatmap(df_results)
    plot_races_per_year(df_results)
    plot_dnf_rate_by_constructor(df_results)
    plot_championship_dominance(df_results)
    
    # Position distributions for top drivers
    print("\nGenerating individual driver analyses...")
    top_drivers = ['max-verstappen', 'lewis-hamilton', 'sebastian-vettel', 
                   'charles-leclerc', 'lando-norris', 'fernando-alonso']
    for driver in top_drivers:
        plot_position_distribution(df_results, driver)
    
    # Generate summary
    print("\nGenerating summary statistics...")
    stats = generate_summary_stats(df_results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    if stats:
        print("\nQUICK STATS:")
        for stat in stats:
            print(f"  {stat}")


if __name__ == "__main__":
    main()
