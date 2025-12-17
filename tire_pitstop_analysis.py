# -*- coding: utf-8 -*-
"""
F1 Tire & Pit Stop Analysis
Visualizations for tire compounds, pit stop times, strategies, and degradation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
import fastf1

warnings.filterwarnings('ignore')

# Setup FastF1
fastf1.Cache.enable_cache(str(Path(__file__).parent / "cache"))

# Dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "tire_pitstop_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# PIT STOP ANALYSIS (from f1db data)
# =============================================================================

def load_pit_stop_data():
    """Load all pit stop data from f1db CSVs."""
    all_data = []
    for csv_file in sorted((DATA_DIR / "f1db_csv").glob("*_pit_stops.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def parse_pit_time(time_str):
    """Parse pit stop time string to seconds."""
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str)
    
    # Handle various formats
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            mins, secs = parts
            try:
                return float(mins) * 60 + float(secs)
            except:
                return np.nan
        elif len(parts) == 3:
            hours, mins, secs = parts
            try:
                return float(hours) * 3600 + float(mins) * 60 + float(secs)
            except:
                return np.nan
    else:
        try:
            return float(time_str)
        except:
            return np.nan


def plot_fastest_pit_stops(pit_data, year=2024):
    """Bar chart of fastest pit stops in a season."""
    df = pit_data[pit_data['year'] == year].copy()
    df['time_seconds'] = df['time'].apply(parse_pit_time)
    
    # Filter reasonable pit stop times (1.5 to 40 seconds)
    df = df[(df['time_seconds'] >= 1.5) & (df['time_seconds'] <= 40)]
    
    # Get fastest per team
    fastest = df.loc[df.groupby('constructorId')['time_seconds'].idxmin()]
    fastest = fastest.sort_values('time_seconds')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(fastest)))
    
    bars = ax.barh(fastest['constructorId'], fastest['time_seconds'], color=colors)
    
    ax.set_xlabel('Pit Stop Time (seconds)', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title(f'{year} Season - Fastest Pit Stop per Team', fontsize=16, fontweight='bold')
    
    for bar, time in zip(bars, fastest['time_seconds']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{time:.3f}s', va='center', fontsize=10)
    
    ax.set_xlim(0, fastest['time_seconds'].max() + 2)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fastest_pit_stops_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fastest_pit_stops_{year}.png")


def plot_pit_stop_distribution(pit_data, year=2024):
    """Box plot of pit stop time distribution per team."""
    df = pit_data[pit_data['year'] == year].copy()
    df['time_seconds'] = df['time'].apply(parse_pit_time)
    
    # Filter reasonable times
    df = df[(df['time_seconds'] >= 1.5) & (df['time_seconds'] <= 40)]
    
    teams = df['constructorId'].unique()
    team_data = [df[df['constructorId'] == t]['time_seconds'].dropna() for t in teams]
    team_data = [(t, d) for t, d in zip(teams, team_data) if len(d) > 0]
    
    # Sort by median
    team_data.sort(key=lambda x: x[1].median())
    teams = [t[0] for t in team_data]
    data = [t[1] for t in team_data]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    bp = ax.boxplot(data, labels=teams, patch_artist=True, vert=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Pit Stop Time (seconds)', fontsize=12)
    ax.set_title(f'{year} Season - Pit Stop Time Distribution by Team', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'pit_stop_distribution_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pit_stop_distribution_{year}.png")


def plot_pit_stops_per_race(pit_data, year=2024):
    """Stacked bar showing pit stops per driver per race."""
    df = pit_data[pit_data['year'] == year].copy()
    
    # Count stops per driver per race
    stops = df.groupby(['race', 'driverId']).size().reset_index(name='stops')
    
    # Pivot for stacked bar
    pivot = stops.pivot(index='race', columns='driverId', values='stops').fillna(0)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Get unique races
    races = pivot.index.tolist()[:15]  # Limit to first 15 races
    pivot = pivot.loc[races]
    
    # Average stops per race
    avg_stops = pivot.sum(axis=1)
    
    ax.bar(races, avg_stops, color='#00aaff')
    
    ax.set_xlabel('Race', fontsize=12)
    ax.set_ylabel('Total Pit Stops', fontsize=12)
    ax.set_title(f'{year} Season - Total Pit Stops per Race', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'pit_stops_per_race_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pit_stops_per_race_{year}.png")


# =============================================================================
# TIRE ANALYSIS (from FastF1)
# =============================================================================

def plot_tire_strategy_race(session, save_name="tire_strategy"):
    """
    Plot tire compound usage throughout the race for all drivers.
    """
    laps = session.laps
    
    if laps is None or laps.empty or 'Compound' not in laps.columns:
        print(f"  No tire data available")
        return
    
    # Get drivers
    drivers = laps['Driver'].unique()
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    for i, driver in enumerate(drivers):
        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        
        for _, lap in driver_laps.iterrows():
            compound = lap.get('Compound', 'UNKNOWN')
            color = compound_colors.get(compound, '#888888')
            ax.barh(i, 1, left=lap['LapNumber'] - 1, color=color, edgecolor='black', linewidth=0.3)
    
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Tire Strategy', fontsize=16, fontweight='bold')
    
    # Legend
    patches = [mpatches.Patch(color=color, label=compound) 
              for compound, color in compound_colors.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_tire_degradation(session, driver=None, save_name="tire_degradation"):
    """
    Plot lap time vs tire age to show degradation.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    if driver:
        laps = laps[laps['Driver'] == driver]
    
    # Convert lap times to seconds
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    # Remove outliers (pit laps, safety car, etc.)
    median_time = laps['LapTimeSeconds'].median()
    laps = laps[(laps['LapTimeSeconds'] > median_time * 0.95) & 
                (laps['LapTimeSeconds'] < median_time * 1.15)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    for compound in laps['Compound'].unique():
        compound_laps = laps[laps['Compound'] == compound]
        color = compound_colors.get(compound, '#888888')
        ax.scatter(compound_laps['TyreLife'], compound_laps['LapTimeSeconds'], 
                  color=color, label=compound, alpha=0.6, s=50)
    
    ax.set_xlabel('Tire Age (laps)', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    title = f'{session.event["EventName"]} - Tire Degradation'
    if driver:
        title += f' ({driver})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_compound_usage_pie(session, save_name="compound_usage"):
    """
    Pie chart of compound usage (total laps per compound).
    """
    laps = session.laps
    
    if laps is None or laps.empty or 'Compound' not in laps.columns:
        print(f"  No tire data available")
        return
    
    compound_counts = laps['Compound'].value_counts()
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    colors = [compound_colors.get(c, '#888888') for c in compound_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.pie(compound_counts.values, labels=compound_counts.index, autopct='%1.1f%%',
           colors=colors, shadow=True, startangle=90,
           textprops={'fontsize': 12, 'color': 'black'})
    
    ax.set_title(f'{session.event["EventName"]} - Compound Usage (Total Laps)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_stint_analysis(session, save_name="stint_analysis"):
    """
    Show stint lengths per driver with compound color coding.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    # Calculate stint lengths
    stints = []
    for driver in laps['Driver'].unique():
        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        
        for stint in driver_laps['Stint'].unique():
            stint_laps = driver_laps[driver_laps['Stint'] == stint]
            if len(stint_laps) > 0:
                compound = stint_laps['Compound'].iloc[0]
                stints.append({
                    'Driver': driver,
                    'Stint': stint,
                    'Compound': compound,
                    'Length': len(stint_laps),
                    'StartLap': stint_laps['LapNumber'].min(),
                    'EndLap': stint_laps['LapNumber'].max()
                })
    
    stint_df = pd.DataFrame(stints)
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    drivers = stint_df['Driver'].unique()
    
    for i, driver in enumerate(drivers):
        driver_stints = stint_df[stint_df['Driver'] == driver]
        
        for _, stint in driver_stints.iterrows():
            color = compound_colors.get(stint['Compound'], '#888888')
            ax.barh(i, stint['Length'], left=stint['StartLap'] - 1, 
                   color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Stint Analysis', fontsize=16, fontweight='bold')
    
    # Legend
    patches = [mpatches.Patch(color=color, label=compound) 
              for compound, color in compound_colors.items()]
    ax.legend(handles=patches, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_average_stint_length(session, save_name="avg_stint_length"):
    """
    Bar chart of average stint length per compound.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    # Calculate stint lengths
    stints = []
    for driver in laps['Driver'].unique():
        driver_laps = laps[laps['Driver'] == driver]
        for stint in driver_laps['Stint'].unique():
            stint_laps = driver_laps[driver_laps['Stint'] == stint]
            if len(stint_laps) > 0:
                compound = stint_laps['Compound'].iloc[0]
                stints.append({
                    'Compound': compound,
                    'Length': len(stint_laps)
                })
    
    stint_df = pd.DataFrame(stints)
    
    # Average stint length per compound
    avg_lengths = stint_df.groupby('Compound')['Length'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = [compound_colors.get(c, '#888888') for c in avg_lengths.index]
    
    bars = ax.bar(avg_lengths.index, avg_lengths.values, color=colors, edgecolor='black')
    
    for bar, length in zip(bars, avg_lengths.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{length:.1f} laps', ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Compound', fontsize=12)
    ax.set_ylabel('Average Stint Length (laps)', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Average Stint Length per Compound', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# ADVANCED TIRE DEGRADATION ANALYSIS
# =============================================================================

def calculate_degradation_rate(laps_data, compound):
    """
    Calculate degradation rate (seconds per lap) for a compound.
    Uses linear regression on clean laps.
    """
    compound_laps = laps_data[laps_data['Compound'] == compound].copy()
    
    if len(compound_laps) < 5:
        return None, None, None
    
    compound_laps['LapTimeSeconds'] = compound_laps['LapTime'].dt.total_seconds()
    
    # Remove outliers (pit laps, safety car)
    median_time = compound_laps['LapTimeSeconds'].median()
    clean_laps = compound_laps[
        (compound_laps['LapTimeSeconds'] > median_time * 0.97) & 
        (compound_laps['LapTimeSeconds'] < median_time * 1.08)
    ]
    
    if len(clean_laps) < 5:
        return None, None, None
    
    # Linear regression: lap_time = base_time + degradation_rate * tire_age
    from scipy import stats
    x = clean_laps['TyreLife'].values
    y = clean_laps['LapTimeSeconds'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return slope, intercept, r_value**2


def plot_degradation_rate_comparison(session, save_name="degradation_rates"):
    """
    Bar chart comparing degradation rates between compounds.
    Shows seconds lost per lap for each compound.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    compounds = laps['Compound'].unique()
    rates = []
    
    for compound in compounds:
        slope, intercept, r_squared = calculate_degradation_rate(laps, compound)
        if slope is not None:
            rates.append({
                'Compound': compound,
                'DegradationRate': slope,  # seconds per lap
                'BaseTime': intercept,
                'R_Squared': r_squared
            })
    
    if not rates:
        print(f"  Not enough data for degradation analysis")
        return
    
    rate_df = pd.DataFrame(rates).sort_values('DegradationRate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = [compound_colors.get(c, '#888888') for c in rate_df['Compound']]
    
    bars = ax.barh(rate_df['Compound'], rate_df['DegradationRate'], color=colors, edgecolor='black')
    
    for bar, (_, row) in zip(bars, rate_df.iterrows()):
        rate = row['DegradationRate']
        r2 = row['R_Squared']
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                f'+{rate:.3f}s/lap (R²={r2:.2f})', va='center', fontsize=10)
    
    ax.set_xlabel('Degradation Rate (seconds per lap)', fontsize=12)
    ax.set_ylabel('Compound', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Tire Degradation Rate by Compound\n'
                f'(Higher = Faster Drop-Off)', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_compound_dropoff_curves(session, save_name="dropoff_curves"):
    """
    Line chart showing lap time progression for each compound.
    Shows the 'drop-off curve' as tires age.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    # Remove outliers
    median_time = laps['LapTimeSeconds'].median()
    clean_laps = laps[
        (laps['LapTimeSeconds'] > median_time * 0.96) & 
        (laps['LapTimeSeconds'] < median_time * 1.10)
    ]
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for compound in clean_laps['Compound'].unique():
        compound_data = clean_laps[clean_laps['Compound'] == compound]
        
        # Group by tire age and get mean lap time
        avg_by_age = compound_data.groupby('TyreLife')['LapTimeSeconds'].agg(['mean', 'std', 'count'])
        avg_by_age = avg_by_age[avg_by_age['count'] >= 2]  # At least 2 data points
        
        if len(avg_by_age) < 3:
            continue
        
        color = compound_colors.get(compound, '#888888')
        
        # Plot mean with confidence band
        ax.plot(avg_by_age.index, avg_by_age['mean'], 
               color=color, linewidth=3, label=compound, marker='o', markersize=6)
        
        # Add confidence band
        ax.fill_between(avg_by_age.index, 
                       avg_by_age['mean'] - avg_by_age['std'],
                       avg_by_age['mean'] + avg_by_age['std'],
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Tire Age (laps)', fontsize=12)
    ax.set_ylabel('Average Lap Time (seconds)', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Tire Drop-Off Curves\n'
                f'(Shows how lap time increases with tire age)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_track_temp_vs_laptime(session, save_name="track_temp_laptime"):
    """
    Scatter plot showing relationship between track temperature and lap times.
    Uses weather data from session.
    """
    laps = session.laps
    weather = session.weather_data
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    if weather is None or weather.empty or 'TrackTemp' not in weather.columns:
        print(f"  No track temperature data available")
        return
    
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    # Remove outliers
    median_time = laps['LapTimeSeconds'].median()
    clean_laps = laps[
        (laps['LapTimeSeconds'] > median_time * 0.97) & 
        (laps['LapTimeSeconds'] < median_time * 1.08)
    ].copy()
    
    # Get track temperature summary
    avg_track_temp = weather['TrackTemp'].mean()
    min_track_temp = weather['TrackTemp'].min()
    max_track_temp = weather['TrackTemp'].max()
    
    # Use lap number as proxy for track temp evolution
    # Track usually heats up during the race
    clean_laps['EstTrackTemp'] = min_track_temp + (clean_laps['LapNumber'] / clean_laps['LapNumber'].max()) * (max_track_temp - min_track_temp)
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Lap time vs estimated track temp
    ax1 = axes[0]
    for compound in clean_laps['Compound'].unique():
        compound_data = clean_laps[clean_laps['Compound'] == compound]
        color = compound_colors.get(compound, '#888888')
        ax1.scatter(compound_data['EstTrackTemp'], compound_data['LapTimeSeconds'],
                  color=color, label=compound, alpha=0.5, s=30)
    
    ax1.set_xlabel('Estimated Track Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax1.set_title(f'Track Temperature vs Lap Time\nRange: {min_track_temp:.1f}°C - {max_track_temp:.1f}°C', 
                fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: Track temp evolution during race
    ax2 = axes[1]
    if 'Time' in weather.columns:
        weather_sorted = weather.reset_index()
        ax2.plot(range(len(weather_sorted)), weather_sorted['TrackTemp'], 
                color='#ff6600', linewidth=2, label='Track Temp')
        if 'AirTemp' in weather_sorted.columns:
            ax2.plot(range(len(weather_sorted)), weather_sorted['AirTemp'], 
                    color='#00aaff', linewidth=2, label='Air Temp')
        ax2.set_xlabel('Weather Sample', fontsize=12)
    else:
        ax2.plot(weather['TrackTemp'].values, color='#ff6600', linewidth=2, label='Track Temp')
        ax2.set_xlabel('Time', fontsize=12)
    
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title(f'Temperature Evolution During Race\nAvg: {avg_track_temp:.1f}°C', 
                fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{session.event["EventName"]} - Temperature Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_degradation_heatmap(session, save_name="degradation_heatmap"):
    """
    Heatmap showing lap time progression per driver, colored by tire age.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    # Remove extreme outliers
    median_time = laps['LapTimeSeconds'].median()
    clean_laps = laps[
        (laps['LapTimeSeconds'] > median_time * 0.90) & 
        (laps['LapTimeSeconds'] < median_time * 1.20)
    ]
    
    # Normalize lap times (percentage of median)
    clean_laps = clean_laps.copy()
    clean_laps['LapTimePct'] = (clean_laps['LapTimeSeconds'] / median_time - 1) * 100
    
    # Create pivot table
    pivot = clean_laps.pivot_table(
        values='LapTimePct', 
        index='Driver',
        columns='LapNumber',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(24, 12))
    
    sns.heatmap(pivot, cmap='RdYlGn_r', center=0, ax=ax,
               cbar_kws={'label': 'Lap Time vs Median (%)'})
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Lap Time Degradation Heatmap\n'
                f'(Green = Faster, Red = Slower)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_tire_cliff_analysis(session, save_name="tire_cliff"):
    """
    Identify 'tire cliff' - the point where performance drops dramatically.
    Shows max stint length before significant drop-off.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    # Remove outliers
    median_time = laps['LapTimeSeconds'].median()
    clean_laps = laps[
        (laps['LapTimeSeconds'] > median_time * 0.97) & 
        (laps['LapTimeSeconds'] < median_time * 1.10)
    ]
    
    compound_colors = {
        'SOFT': '#ff0000',
        'MEDIUM': '#ffff00',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#00ff00',
        'WET': '#0066ff'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    available_compounds = [c for c in compounds if c in clean_laps['Compound'].unique()]
    
    for idx, compound in enumerate(available_compounds[:3]):
        ax = axes[idx] if len(available_compounds) >= 3 else axes[idx % len(available_compounds)]
        compound_data = clean_laps[clean_laps['Compound'] == compound]
        
        # Group by tire age
        avg_by_age = compound_data.groupby('TyreLife')['LapTimeSeconds'].mean()
        
        if len(avg_by_age) < 3:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=14)
            ax.set_title(compound, fontsize=14, fontweight='bold', 
                        color=compound_colors.get(compound, 'white'))
            continue
        
        color = compound_colors.get(compound, '#888888')
        
        # Plot lap times
        ax.plot(avg_by_age.index, avg_by_age.values, color=color, linewidth=2, marker='o')
        
        # Calculate and show degradation per 5-lap window
        if len(avg_by_age) >= 10:
            # Find where degradation accelerates (cliff)
            deltas = avg_by_age.diff()
            rolling_delta = deltas.rolling(3).mean()
            
            # Cliff = where rolling delta exceeds 1.5x the mean delta
            mean_delta = deltas.mean()
            cliff_candidates = rolling_delta[rolling_delta > mean_delta * 1.5]
            
            if not cliff_candidates.empty:
                cliff_lap = cliff_candidates.index[0]
                ax.axvline(x=cliff_lap, color='red', linestyle='--', linewidth=2, 
                          label=f'Cliff at lap {cliff_lap}')
                ax.legend()
        
        ax.set_xlabel('Tire Age (laps)', fontsize=11)
        ax.set_ylabel('Lap Time (seconds)', fontsize=11)
        ax.set_title(compound, fontsize=14, fontweight='bold', 
                    color=compound_colors.get(compound, 'white'))
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{session.event["EventName"]} - Tire Cliff Analysis\n'
                f'(Red line indicates performance drop-off point)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def analyze_race_tires(year, event_name):
    """
    Full tire analysis for a race.
    """
    print(f"\n{'='*60}")
    print(f"LOADING: {year} {event_name} (Race)")
    print(f"{'='*60}")
    
    try:
        session = fastf1.get_session(year, event_name, 'R')
        session.load()
    except Exception as e:
        print(f"Error loading session: {e}")
        return None
    
    event_name_clean = session.event['EventName'].replace(' ', '_')
    prefix = f"{year}_{event_name_clean}"
    
    print("\nGenerating basic tire visualizations...")
    plot_tire_strategy_race(session, f"{prefix}_tire_strategy")
    plot_tire_degradation(session, save_name=f"{prefix}_tire_degradation")
    plot_compound_usage_pie(session, f"{prefix}_compound_usage")
    plot_stint_analysis(session, f"{prefix}_stint_analysis")
    plot_average_stint_length(session, f"{prefix}_avg_stint_length")
    
    print("\nGenerating advanced degradation analysis...")
    plot_degradation_rate_comparison(session, f"{prefix}_degradation_rates")
    plot_compound_dropoff_curves(session, f"{prefix}_dropoff_curves")
    plot_track_temp_vs_laptime(session, f"{prefix}_track_temp_laptime")
    plot_degradation_heatmap(session, f"{prefix}_degradation_heatmap")
    plot_tire_cliff_analysis(session, f"{prefix}_tire_cliff")
    
    return session


def main():
    print("=" * 60)
    print("F1 TIRE & PIT STOP ANALYSIS")
    print("=" * 60)
    
    # Load pit stop data from f1db
    print("\nLoading pit stop data...")
    pit_data = load_pit_stop_data()
    
    if not pit_data.empty:
        print(f"Loaded {len(pit_data)} pit stop records")
        
        print("\nGenerating pit stop visualizations...")
        plot_fastest_pit_stops(pit_data, 2024)
        plot_pit_stop_distribution(pit_data, 2024)
        plot_pit_stops_per_race(pit_data, 2024)
    else:
        print("No pit stop data available")
    
    # Analyze tire data from FastF1
    print("\n" + "=" * 60)
    print("TIRE ANALYSIS FROM TELEMETRY")
    print("=" * 60)
    
    # Analyze multiple races
    races = [
        (2024, 'Monaco'),
        (2024, 'Belgium'),
        (2024, 'Italian Grand Prix'),
    ]
    
    for year, event in races:
        analyze_race_tires(year, event)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
