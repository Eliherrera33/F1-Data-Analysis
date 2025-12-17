# -*- coding: utf-8 -*-
"""
F1 Advanced Telemetry Analysis - Enhanced Version
Track maps, driver lines, brake/throttle usage, G-force, braking points, and more.
Uses FastF1 for detailed telemetry data (2018+).

Tracks: Monaco, Spa-Francorchamps (Belgium), Monza (Italy)
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup FastF1
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# Directories
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = BASE_DIR / "telemetry_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dark theme
plt.style.use('dark_background')


# =============================================================================
# ENHANCED DRIVER COMPARISON WITH G-FORCE & BRAKING POINTS
# =============================================================================

def calculate_g_force(telemetry, smoothing=5):
    """
    Calculate longitudinal and lateral G-force from telemetry data.
    """
    tel = telemetry.copy()
    
    # Calculate time delta
    if 'Time' in tel.columns:
        tel['TimeDelta'] = tel['Time'].diff().dt.total_seconds()
    else:
        tel['TimeDelta'] = 0.01  # Assume 100Hz sampling
    
    # Longitudinal G-force (acceleration/braking) from speed change
    # Speed is in km/h, convert to m/s
    speed_ms = tel['Speed'] / 3.6
    
    # Calculate acceleration (m/s²)
    tel['Acceleration'] = speed_ms.diff() / tel['TimeDelta']
    
    # Convert to G-force (1G = 9.81 m/s²)
    tel['G_Longitudinal'] = tel['Acceleration'] / 9.81
    
    # Smooth the data
    tel['G_Longitudinal'] = tel['G_Longitudinal'].rolling(window=smoothing, center=True).mean()
    
    # Cap extreme values (sensor noise)
    tel['G_Longitudinal'] = tel['G_Longitudinal'].clip(-6, 6)
    
    return tel


def identify_braking_zones(telemetry, brake_threshold=0.5):
    """
    Identify braking zones from telemetry.
    Returns start and end distances of braking zones.
    """
    tel = telemetry.copy()
    brake_data = tel['Brake'] if 'Brake' in tel.columns else None
    
    if brake_data is None:
        return []
    
    braking_zones = []
    in_braking = False
    start_dist = 0
    
    for idx, row in tel.iterrows():
        if row['Brake'] > brake_threshold and not in_braking:
            in_braking = True
            start_dist = row['Distance']
        elif row['Brake'] <= brake_threshold and in_braking:
            in_braking = False
            braking_zones.append({
                'start': start_dist,
                'end': row['Distance'],
                'distance': row['Distance'] - start_dist
            })
    
    return braking_zones


def plot_detailed_driver_comparison(session, driver1, driver2, save_name="detailed_comparison"):
    """
    Ultra-detailed comparison with G-force, braking points, lap times, and throttle.
    """
    lap1 = session.laps.pick_driver(driver1).pick_fastest()
    lap2 = session.laps.pick_driver(driver2).pick_fastest()
    
    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()
    
    if tel1 is None or tel2 is None or tel1.empty or tel2.empty:
        print(f"  No telemetry data available for detailed comparison")
        return
    
    # Calculate G-forces
    tel1 = calculate_g_force(tel1)
    tel2 = calculate_g_force(tel2)
    
    # Get lap times
    lap1_time = lap1['LapTime']
    lap2_time = lap2['LapTime']
    
    fig, axes = plt.subplots(6, 1, figsize=(20, 18), sharex=True)
    
    # Colors
    color1 = '#ff3333'  # Red for driver 1
    color2 = '#33aaff'  # Blue for driver 2
    
    # 1. Speed comparison
    axes[0].plot(tel1['Distance'], tel1['Speed'], color=color1, label=f'{driver1}', linewidth=1.5)
    axes[0].plot(tel2['Distance'], tel2['Speed'], color=color2, label=f'{driver2}', linewidth=1.5)
    axes[0].set_ylabel('Speed (km/h)', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('SPEED COMPARISON', fontsize=10, loc='left', color='white')
    
    # 2. Throttle comparison
    axes[1].plot(tel1['Distance'], tel1['Throttle'], color=color1, linewidth=1.5, alpha=0.8)
    axes[1].plot(tel2['Distance'], tel2['Throttle'], color=color2, linewidth=1.5, alpha=0.8)
    axes[1].fill_between(tel1['Distance'], tel1['Throttle'], alpha=0.2, color=color1)
    axes[1].fill_between(tel2['Distance'], tel2['Throttle'], alpha=0.2, color=color2)
    axes[1].set_ylabel('Throttle %', fontsize=11, fontweight='bold')
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('THROTTLE APPLICATION', fontsize=10, loc='left', color='white')
    
    # 3. Brake comparison with braking zones highlighted
    # Brake is boolean (True/False), convert to 0-100 scale
    if 'Brake' in tel1.columns:
        brake1 = tel1['Brake'].astype(float) * 100
        brake2 = tel2['Brake'].astype(float) * 100
        axes[2].plot(tel1['Distance'], brake1, color=color1, linewidth=1.5)
        axes[2].plot(tel2['Distance'], brake2, color=color2, linewidth=1.5)
        axes[2].fill_between(tel1['Distance'], brake1, alpha=0.3, color=color1)
        axes[2].fill_between(tel2['Distance'], brake2, alpha=0.3, color=color2)
        
        # Mark braking zones
        braking_zones1 = identify_braking_zones(tel1)
        for zone in braking_zones1:
            axes[2].axvspan(zone['start'], zone['end'], alpha=0.15, color=color1)
    
    axes[2].set_ylabel('Brake On/Off', fontsize=11, fontweight='bold')
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('BRAKE STATUS (100 = Braking, 0 = Not Braking)', fontsize=10, loc='left', color='white')
    
    # 4. G-Force (Longitudinal - Braking/Acceleration)
    axes[3].plot(tel1['Distance'], tel1['G_Longitudinal'], color=color1, linewidth=1.2, alpha=0.8)
    axes[3].plot(tel2['Distance'], tel2['G_Longitudinal'], color=color2, linewidth=1.2, alpha=0.8)
    axes[3].axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[3].fill_between(tel1['Distance'], tel1['G_Longitudinal'], 0, 
                        where=tel1['G_Longitudinal'] < 0, alpha=0.3, color='#ff6600', label='Braking G')
    axes[3].fill_between(tel1['Distance'], tel1['G_Longitudinal'], 0, 
                        where=tel1['G_Longitudinal'] > 0, alpha=0.3, color='#00ff00', label='Accel G')
    axes[3].set_ylabel('G-Force (Long)', fontsize=11, fontweight='bold')
    axes[3].set_ylim(-5, 3)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].set_title('LONGITUDINAL G-FORCE (Braking = Negative, Acceleration = Positive)', fontsize=10, loc='left', color='white')
    
    # 5. Gear comparison
    if 'nGear' in tel1.columns:
        axes[4].plot(tel1['Distance'], tel1['nGear'], color=color1, linewidth=2, drawstyle='steps-post')
        axes[4].plot(tel2['Distance'], tel2['nGear'], color=color2, linewidth=2, drawstyle='steps-post', alpha=0.8)
        axes[4].set_ylabel('Gear', fontsize=11, fontweight='bold')
        axes[4].set_ylim(0, 9)
        axes[4].set_yticks(range(1, 9))
        axes[4].grid(True, alpha=0.3)
        axes[4].set_title('GEAR SELECTION', fontsize=10, loc='left', color='white')
    
    # 6. Speed delta (time gained/lost)
    min_len = min(len(tel1), len(tel2))
    speed_diff = tel1['Speed'].values[:min_len] - tel2['Speed'].values[:min_len]
    dist = tel1['Distance'].values[:min_len]
    
    axes[5].plot(dist, speed_diff, color='#ffff00', linewidth=1.5)
    axes[5].fill_between(dist, speed_diff, 0, where=speed_diff > 0, alpha=0.4, color=color1, label=f'{driver1} faster')
    axes[5].fill_between(dist, speed_diff, 0, where=speed_diff < 0, alpha=0.4, color=color2, label=f'{driver2} faster')
    axes[5].axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
    axes[5].set_ylabel('Speed Delta (km/h)', fontsize=11, fontweight='bold')
    axes[5].set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
    axes[5].legend(loc='upper right', fontsize=10)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_title('SPEED ADVANTAGE', fontsize=10, loc='left', color='white')
    
    # Format lap times
    def format_laptime(td):
        if pd.isna(td):
            return "N/A"
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"
    
    lap1_str = format_laptime(lap1_time)
    lap2_str = format_laptime(lap2_time)
    
    # Calculate gap
    if not pd.isna(lap1_time) and not pd.isna(lap2_time):
        gap = (lap1_time - lap2_time).total_seconds()
        if gap > 0:
            gap_str = f"+{gap:.3f}s"
            faster = driver2
        else:
            gap_str = f"+{abs(gap):.3f}s"
            faster = driver1
    else:
        gap_str = "N/A"
        faster = "N/A"
    
    # Get year safely
    try:
        year = session.event.year if hasattr(session.event, 'year') else session.event.get('Year', 2024)
    except:
        year = 2024
    
    fig.suptitle(f'{session.event["EventName"]} {year} - DETAILED TELEMETRY COMPARISON\n'
                f'{driver1}: {lap1_str}  |  {driver2}: {lap2_str}  |  Gap: {gap_str} ({faster} faster)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_track_with_braking_zones(session, driver=None, save_name="track_braking"):
    """
    Plot track map highlighted with braking zones.
    Brake data in FastF1 is boolean (True/False) - True means braking.
    """
    if driver:
        lap = session.laps.pick_driver(driver).pick_fastest()
        driver_name = driver
    else:
        lap = session.laps.pick_fastest()
        driver_name = lap['Driver']
    
    tel = lap.get_telemetry()
    
    if tel is None or tel.empty or 'Brake' not in tel.columns:
        print(f"  No brake data available")
        return
    
    x = tel['X'].values
    y = tel['Y'].values
    
    # Convert boolean brake to numeric (True=1, False=0)
    brake = tel['Brake'].astype(float).values
    
    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create custom colormap: gray for no brake, red for braking
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#333333', '#ff0000']  # Dark gray to bright red
    cmap = LinearSegmentedColormap.from_list('brake_cmap', colors)
    
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(brake)
    lc.set_linewidth(6)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    
    # Colorbar with proper labels
    cbar = fig.colorbar(lc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Coasting', 'BRAKING'])
    cbar.set_label('Brake Status', fontsize=12)
    
    ax.set_title(f'{session.event["EventName"]} - Braking Zones ({driver_name})\nRed = Braking | Gray = Coasting/Accelerating', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_track_with_throttle(session, driver=None, save_name="track_throttle"):
    """
    Plot track map colored by throttle application.
    """
    if driver:
        lap = session.laps.pick_driver(driver).pick_fastest()
        driver_name = driver
    else:
        lap = session.laps.pick_fastest()
        driver_name = lap['Driver']
    
    tel = lap.get_telemetry()
    
    if tel is None or tel.empty or 'Throttle' not in tel.columns:
        print(f"  No throttle data available")
        return
    
    x = tel['X'].values
    y = tel['Y'].values
    throttle = tel['Throttle'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, 100)
    lc = LineCollection(segments, cmap='Greens', norm=norm)
    lc.set_array(throttle)
    lc.set_linewidth(5)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(lc, ax=ax, label='Throttle %')
    
    ax.set_title(f'{session.event["EventName"]} - Throttle Application ({driver_name})', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_combined_track_analysis(session, driver=None, save_name="track_combined"):
    """
    4-panel track map: Speed, Throttle, Brake, Gear in one view.
    """
    if driver:
        lap = session.laps.pick_driver(driver).pick_fastest()
        driver_name = driver
    else:
        lap = session.laps.pick_fastest()
        driver_name = lap['Driver']
    
    tel = lap.get_telemetry()
    
    if tel is None or tel.empty:
        print(f"  No telemetry data available")
        return
    
    x = tel['X'].values
    y = tel['Y'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Create custom colormap for brake (gray to red)
    from matplotlib.colors import LinearSegmentedColormap
    brake_cmap = LinearSegmentedColormap.from_list('brake_cmap', ['#333333', '#ff0000'])
    
    # Convert brake boolean to float
    brake_data = tel['Brake'].astype(float).values if 'Brake' in tel.columns else None
    
    data_configs = [
        ('Speed', tel['Speed'].values, 'plasma', 'Speed (km/h)', None),
        ('Throttle', tel['Throttle'].values if 'Throttle' in tel.columns else None, 'Greens', 'Throttle %', None),
        ('Brake', brake_data, brake_cmap, 'Brake (Red=Braking)', (0, 1)),
        ('nGear', tel['nGear'].values if 'nGear' in tel.columns else None, 'rainbow', 'Gear', (1, 8)),
    ]
    
    titles = ['SPEED MAP', 'THROTTLE MAP', 'BRAKING ZONES (Red=Brake)', 'GEAR MAP']
    
    for ax, (col, data, cmap, label, norm_range), title in zip(axes, data_configs, titles):
        if data is None:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='white')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        if norm_range:
            norm = plt.Normalize(norm_range[0], norm_range[1])
        else:
            norm = plt.Normalize(data.min(), data.max())
        
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(data)
        lc.set_linewidth(4)
        
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal')
        
        if col == 'Brake':
            cbar = fig.colorbar(lc, ax=ax, ticks=[0, 1], shrink=0.8)
            cbar.ax.set_yticklabels(['Coast', 'BRAKE'])
        else:
            cbar = fig.colorbar(lc, ax=ax, label=label, shrink=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'{session.event["EventName"]} - Combined Track Analysis ({driver_name})', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_mini_sectors_comparison(session, driver1, driver2, save_name="mini_sectors"):
    """
    Compare mini-sector times between two drivers.
    Shows where each driver gains/loses time.
    """
    lap1 = session.laps.pick_driver(driver1).pick_fastest()
    lap2 = session.laps.pick_driver(driver2).pick_fastest()
    
    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()
    
    if tel1 is None or tel2 is None:
        print(f"  No telemetry data available for mini-sector comparison")
        return
    
    # Create mini-sectors (every 100m)
    sector_size = 100
    max_dist = min(tel1['Distance'].max(), tel2['Distance'].max())
    sectors = np.arange(0, max_dist, sector_size)
    
    sector_times1 = []
    sector_times2 = []
    
    for i, sector_start in enumerate(sectors[:-1]):
        sector_end = sectors[i + 1]
        
        # Find time in this sector for driver 1
        mask1 = (tel1['Distance'] >= sector_start) & (tel1['Distance'] < sector_end)
        mask2 = (tel2['Distance'] >= sector_start) & (tel2['Distance'] < sector_end)
        
        if mask1.any():
            time1 = (tel1.loc[mask1, 'Time'].max() - tel1.loc[mask1, 'Time'].min()).total_seconds()
        else:
            time1 = 0
        
        if mask2.any():
            time2 = (tel2.loc[mask2, 'Time'].max() - tel2.loc[mask2, 'Time'].min()).total_seconds()
        else:
            time2 = 0
        
        sector_times1.append(time1)
        sector_times2.append(time2)
    
    # Calculate delta per sector
    delta = np.array(sector_times1) - np.array(sector_times2)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = sectors[:-1] + sector_size/2  # Center of each sector
    colors = ['#ff0000' if d > 0 else '#00aaff' for d in delta]
    
    bars = ax.bar(x, delta * 1000, width=sector_size * 0.8, color=colors, alpha=0.8)
    
    ax.axhline(y=0, color='white', linestyle='-', linewidth=1)
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Time Delta (ms per sector)', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Mini-Sector Analysis\n'
                f'Red = {driver2} faster | Blue = {driver1} faster', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# ORIGINAL FUNCTIONS (Keep for compatibility)
# =============================================================================

def plot_track_map(session, save_name="track_map"):
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    
    if pos is None or pos.empty:
        print(f"  No position data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(pos['X'], pos['Y'], color='white', linewidth=3, alpha=0.5)
    ax.scatter(pos['X'].iloc[0], pos['Y'].iloc[0], color='green', s=200, zorder=5, label='Start/Finish')
    ax.set_aspect('equal')
    ax.set_title(f'{session.event["EventName"]} - Track Layout', fontsize=16, fontweight='bold')
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_track_with_speed(session, driver=None, save_name="track_speed"):
    if driver:
        lap = session.laps.pick_driver(driver).pick_fastest()
        driver_name = driver
    else:
        lap = session.laps.pick_fastest()
        driver_name = lap['Driver']
    
    tel = lap.get_telemetry()
    
    if tel is None or tel.empty:
        print(f"  No telemetry data available")
        return
    
    x = tel['X'].values
    y = tel['Y'].values
    speed = tel['Speed'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(speed.min(), speed.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(speed)
    lc.set_linewidth(4)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    cbar = fig.colorbar(lc, ax=ax, label='Speed (km/h)')
    ax.set_title(f'{session.event["EventName"]} - Speed Map ({driver_name})', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_weather_data(session, save_name="weather"):
    weather = session.weather_data
    
    if weather is None or weather.empty:
        print(f"  No weather data available")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    if 'AirTemp' in weather.columns:
        axes[0].plot(weather.index, weather['AirTemp'], color='#ff6600', linewidth=2)
        axes[0].set_ylabel('Air Temp (°C)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(weather.index, weather['AirTemp'], alpha=0.3, color='#ff6600')
    
    if 'TrackTemp' in weather.columns:
        axes[1].plot(weather.index, weather['TrackTemp'], color='#ff0000', linewidth=2)
        axes[1].set_ylabel('Track Temp (°C)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(weather.index, weather['TrackTemp'], alpha=0.3, color='#ff0000')
    
    if 'Humidity' in weather.columns:
        axes[2].plot(weather.index, weather['Humidity'], color='#00aaff', linewidth=2)
        axes[2].set_ylabel('Humidity (%)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(weather.index, weather['Humidity'], alpha=0.3, color='#00aaff')
    
    if 'WindSpeed' in weather.columns:
        axes[3].plot(weather.index, weather['WindSpeed'], color='#00ff00', linewidth=2)
        axes[3].set_ylabel('Wind Speed (m/s)', fontsize=11)
        axes[3].set_xlabel('Time', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        axes[3].fill_between(weather.index, weather['WindSpeed'], alpha=0.3, color='#00ff00')
    
    fig.suptitle(f'{session.event["EventName"]} - Weather Conditions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_lap_times_distribution(session, save_name="lap_times"):
    laps = session.laps.pick_quicklaps()
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    drivers = laps['Driver'].unique()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    driver_data = [laps[laps['Driver'] == d]['LapTimeSeconds'].dropna() for d in drivers]
    driver_data = [d for d in driver_data if len(d) > 0]
    drivers = [d for d, data in zip(drivers, [laps[laps['Driver'] == d]['LapTimeSeconds'].dropna() for d in drivers]) if len(data) > 0]
    
    bp = ax.boxplot(driver_data, labels=drivers, patch_artist=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Lap Time Distribution', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# ENHANCED SESSION ANALYSIS
# =============================================================================

def analyze_session_enhanced(year, event, session_type='Q', driver_pairs=None):
    """
    Enhanced telemetry analysis with detailed driver comparisons.
    """
    print(f"\n{'='*60}")
    print(f"LOADING: {year} {event} ({session_type})")
    print(f"{'='*60}")
    
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
    except Exception as e:
        print(f"Error loading session: {e}")
        return None
    
    event_name = session.event['EventName'].replace(' ', '_')
    prefix = f"{year}_{event_name}"
    
    # Track visualizations
    print("\nGenerating track visualizations...")
    plot_track_map(session, f"{prefix}_track_map")
    plot_track_with_speed(session, save_name=f"{prefix}_speed_map")
    plot_track_with_braking_zones(session, save_name=f"{prefix}_braking_map")
    plot_track_with_throttle(session, save_name=f"{prefix}_throttle_map")
    plot_combined_track_analysis(session, save_name=f"{prefix}_combined_map")
    
    # Weather
    print("\nGenerating weather data...")
    plot_weather_data(session, f"{prefix}_weather")
    
    # Lap distribution
    print("\nGenerating lap analysis...")
    plot_lap_times_distribution(session, f"{prefix}_lap_distribution")
    
    # Driver comparisons
    print("\nGenerating detailed driver comparisons...")
    
    if driver_pairs is None:
        # Get top drivers from qualifying/race
        try:
            laps = session.laps
            fastest = laps.groupby('Driver')['LapTime'].min().nsmallest(4)
            top_drivers = fastest.index.tolist()
            
            # Compare top 2 and 3-4
            driver_pairs = [
                (top_drivers[0], top_drivers[1]),
                (top_drivers[0], top_drivers[2]) if len(top_drivers) > 2 else None,
            ]
            driver_pairs = [p for p in driver_pairs if p is not None]
        except:
            driver_pairs = []
    
    for d1, d2 in driver_pairs:
        try:
            print(f"  Comparing {d1} vs {d2}...")
            plot_detailed_driver_comparison(session, d1, d2, 
                                           f"{prefix}_detailed_{d1}_vs_{d2}")
            plot_mini_sectors_comparison(session, d1, d2, 
                                        f"{prefix}_minisectors_{d1}_vs_{d2}")
        except Exception as e:
            print(f"  Could not compare {d1} vs {d2}: {e}")
    
    return session


def main():
    print("=" * 60)
    print("F1 ADVANCED TELEMETRY ANALYSIS - ENHANCED VERSION")
    print("=" * 60)
    print("\nTracks: Monaco, Spa-Francorchamps (Belgium), Monza (Italy)")
    print("Features: G-force, braking zones, throttle maps, lap times")
    
    # MONACO - Tight street circuit
    print("\n" + "=" * 60)
    print("1/3 - MONACO GRAND PRIX 2024 (Street Circuit)")
    print("=" * 60)
    analyze_session_enhanced(2024, 'Monaco', 'Q')
    
    # SPA-FRANCORCHAMPS - High speed with elevation
    print("\n" + "=" * 60)
    print("2/3 - BELGIAN GRAND PRIX 2024 (Spa-Francorchamps)")
    print("=" * 60)
    analyze_session_enhanced(2024, 'Belgium', 'Q')
    
    # MONZA - Temple of speed (Italian GP Round 16 in 2024)
    print("\n" + "=" * 60)
    print("3/3 - ITALIAN GRAND PRIX 2024 (Monza)")
    print("=" * 60)
    analyze_session_enhanced(2024, 'Italian Grand Prix', 'Q')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
