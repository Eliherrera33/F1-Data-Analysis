# -*- coding: utf-8 -*-
"""
F1 Car & Aerodynamic Analysis
DRS usage, speed profiles, chassis history, and downforce comparison.
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
OUTPUT_DIR = BASE_DIR / "car_aero_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DRS USAGE ANALYSIS
# =============================================================================

def plot_drs_zones(session, save_name="drs_zones"):
    """
    Plot track map colored by DRS activation zones.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    # Get fastest lap for track map
    fastest_lap = laps.pick_fastest()
    if fastest_lap is None:
        print(f"  No fastest lap available")
        return
    
    tel = fastest_lap.get_telemetry()
    
    if tel is None or tel.empty or 'DRS' not in tel.columns:
        print(f"  No DRS data available")
        return
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Color by DRS status
    drs_colors = {
        0: '#333333',   # Off
        1: '#00ff00',   # Open
        8: '#00ff00',   # Open (alternative code)
        10: '#ffff00',  # Eligible
        12: '#00ff00',  # Open
        14: '#00ff00',  # Open
    }
    
    # Plot track segments colored by DRS
    for i in range(len(tel) - 1):
        x = [tel.iloc[i]['X'], tel.iloc[i+1]['X']]
        y = [tel.iloc[i]['Y'], tel.iloc[i+1]['Y']]
        drs = tel.iloc[i]['DRS']
        color = '#00ff00' if drs > 0 else '#555555'
        ax.plot(x, y, color=color, linewidth=3)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{session.event["EventName"]} - DRS Zones\n(Green = DRS Active)', 
                fontsize=16, fontweight='bold')
    
    # Legend
    patches = [
        mpatches.Patch(color='#00ff00', label='DRS Active'),
        mpatches.Patch(color='#555555', label='DRS Inactive')
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_drs_usage_by_driver(session, save_name="drs_usage_drivers"):
    """
    Bar chart showing DRS usage percentage per driver.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    drs_usage = []
    
    for driver in laps['Driver'].unique():
        driver_laps = laps[laps['Driver'] == driver]
        
        total_drs_time = 0
        total_lap_time = 0
        
        for _, lap in driver_laps.iterrows():
            try:
                tel = lap.get_telemetry()
                if tel is not None and 'DRS' in tel.columns and len(tel) > 0:
                    drs_active = (tel['DRS'] > 0).sum()
                    total_drs_time += drs_active
                    total_lap_time += len(tel)
            except:
                continue
        
        if total_lap_time > 0:
            drs_pct = (total_drs_time / total_lap_time) * 100
            drs_usage.append({
                'Driver': driver,
                'DRS_Percentage': drs_pct
            })
    
    if not drs_usage:
        print(f"  No DRS data available for drivers")
        return
    
    df = pd.DataFrame(drs_usage).sort_values('DRS_Percentage', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.Greens(np.linspace(0.3, 1, len(df)))
    
    bars = ax.barh(df['Driver'], df['DRS_Percentage'], color=colors)
    
    for bar, pct in zip(bars, df['DRS_Percentage']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('DRS Active (%)', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - DRS Usage by Driver', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_drs_vs_speed(session, driver=None, save_name="drs_speed"):
    """
    Show speed increase when DRS is open vs closed.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    if driver:
        laps = laps[laps['Driver'] == driver]
    
    drs_on_speeds = []
    drs_off_speeds = []
    
    for _, lap in laps.iterrows():
        try:
            tel = lap.get_telemetry()
            if tel is not None and 'DRS' in tel.columns:
                drs_on = tel[tel['DRS'] > 0]['Speed']
                drs_off = tel[tel['DRS'] == 0]['Speed']
                drs_on_speeds.extend(drs_on.tolist())
                drs_off_speeds.extend(drs_off.tolist())
        except:
            continue
    
    if not drs_on_speeds or not drs_off_speeds:
        print(f"  No DRS comparison data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Histogram comparison
    ax1 = axes[0]
    ax1.hist(drs_off_speeds, bins=50, alpha=0.7, label='DRS Closed', color='#ff6666')
    ax1.hist(drs_on_speeds, bins=50, alpha=0.7, label='DRS Open', color='#66ff66')
    ax1.set_xlabel('Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Speed Distribution: DRS Open vs Closed', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2 = axes[1]
    bp = ax2.boxplot([drs_off_speeds, drs_on_speeds], 
                     labels=['DRS Closed', 'DRS Open'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6666')
    bp['boxes'][1].set_facecolor('#66ff66')
    
    avg_off = np.mean(drs_off_speeds)
    avg_on = np.mean(drs_on_speeds)
    speed_gain = avg_on - avg_off
    
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.set_title(f'DRS Speed Advantage: +{speed_gain:.1f} km/h average', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    title = f'{session.event["EventName"]} - DRS Speed Analysis'
    if driver:
        title += f' ({driver})'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# CAR SPEED PROFILES (DOWNFORCE INFERENCE)
# =============================================================================

def plot_cornering_speed_analysis(session, save_name="cornering_speed"):
    """
    Analyze cornering speeds to infer downforce levels.
    High downforce = faster through corners.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    corner_speeds = {}
    
    for driver in laps['Driver'].unique():
        fastest = laps.pick_driver(driver).pick_fastest()
        if fastest is None:
            continue
        
        try:
            tel = fastest.get_telemetry()
            if tel is None or tel.empty:
                continue
            
            # Identify corners: where speed drops below 80% of max
            max_speed = tel['Speed'].max()
            corner_threshold = max_speed * 0.80
            
            # Get minimum speeds (corners)
            corner_speeds_driver = tel[tel['Speed'] < corner_threshold]['Speed']
            if len(corner_speeds_driver) > 0:
                corner_speeds[driver] = {
                    'min_speed': corner_speeds_driver.min(),
                    'avg_corner_speed': corner_speeds_driver.mean(),
                    'max_speed': max_speed
                }
        except:
            continue
    
    if not corner_speeds:
        print(f"  No corner speed data available")
        return
    
    df = pd.DataFrame(corner_speeds).T.reset_index()
    df.columns = ['Driver', 'MinSpeed', 'AvgCornerSpeed', 'MaxSpeed']
    df = df.sort_values('AvgCornerSpeed', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['AvgCornerSpeed'], width, 
                   label='Avg Corner Speed', color='#ff9900')
    bars2 = ax.bar([i + width/2 for i in x], df['MinSpeed'], width, 
                   label='Min Corner Speed', color='#ff3300')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Driver'], rotation=45, ha='right')
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Cornering Speed Analysis\n'
                f'(Higher = Better Downforce/Grip)', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_speed_trace_comparison(session, drivers=None, save_name="speed_trace"):
    """
    Compare speed traces between drivers to show aero differences.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    if drivers is None:
        # Get top 3 finishers
        drivers = laps['Driver'].unique()[:3]
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    colors = plt.cm.tab10(range(len(drivers)))
    
    for driver, color in zip(drivers, colors):
        fastest = laps.pick_driver(driver).pick_fastest()
        if fastest is None:
            continue
        
        try:
            tel = fastest.get_telemetry()
            if tel is None or tel.empty:
                continue
            
            ax.plot(tel['Distance'], tel['Speed'], color=color, 
                   label=driver, linewidth=1.5, alpha=0.8)
        except:
            continue
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Speed Trace Comparison\n'
                f'(Shows aero efficiency differences)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_sector_speed_breakdown(session, save_name="sector_speeds"):
    """
    Bar chart showing average speed per sector for each driver.
    """
    laps = session.laps
    
    if laps is None or laps.empty:
        print(f"  No lap data available")
        return
    
    sector_data = []
    
    for driver in laps['Driver'].unique():
        fastest = laps.pick_driver(driver).pick_fastest()
        if fastest is None:
            continue
        
        try:
            tel = fastest.get_telemetry()
            if tel is None or tel.empty:
                continue
            
            total_dist = tel['Distance'].max()
            s1_end = total_dist / 3
            s2_end = 2 * total_dist / 3
            
            s1_speed = tel[tel['Distance'] <= s1_end]['Speed'].mean()
            s2_speed = tel[(tel['Distance'] > s1_end) & (tel['Distance'] <= s2_end)]['Speed'].mean()
            s3_speed = tel[tel['Distance'] > s2_end]['Speed'].mean()
            
            sector_data.append({
                'Driver': driver,
                'S1': s1_speed,
                'S2': s2_speed,
                'S3': s3_speed
            })
        except:
            continue
    
    if not sector_data:
        print(f"  No sector speed data available")
        return
    
    df = pd.DataFrame(sector_data)
    df['AvgSpeed'] = (df['S1'] + df['S2'] + df['S3']) / 3
    df = df.sort_values('AvgSpeed', ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = range(len(df))
    width = 0.25
    
    bars1 = ax.bar([i - width for i in x], df['S1'], width, label='Sector 1', color='#ff6666')
    bars2 = ax.bar(x, df['S2'], width, label='Sector 2', color='#66ff66')
    bars3 = ax.bar([i + width for i in x], df['S3'], width, label='Sector 3', color='#6666ff')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Driver'], rotation=45, ha='right')
    ax.set_ylabel('Average Speed (km/h)', fontsize=12)
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - Sector Speed Breakdown', 
                fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# CHASSIS USAGE VISUALIZATION
# =============================================================================

def load_chassis_data():
    """Load chassis data from reference CSV."""
    csv_path = DATA_DIR / "reference" / "chassis.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def plot_chassis_by_team(save_name="chassis_by_team"):
    """
    Show chassis models used by each team (modern era 2020-2024).
    """
    chassis_df = load_chassis_data()
    
    if chassis_df.empty:
        print(f"  No chassis data available")
        return
    
    # Filter for modern teams
    modern_teams = ['red-bull', 'mercedes', 'ferrari', 'mclaren', 'aston-martin',
                   'alpine', 'williams', 'alphatauri', 'alfa-romeo', 'haas']
    
    modern_chassis = chassis_df[chassis_df['constructorId'].isin(modern_teams)]
    
    # Get recent chassis (2020+)
    recent_patterns = ['20', '21', '22', '23', '24', '25']
    recent_chassis = modern_chassis[
        modern_chassis['name'].str.contains('|'.join(recent_patterns), na=False)
    ]
    
    if recent_chassis.empty:
        # Fall back to all modern team chassis
        recent_chassis = modern_chassis.tail(40)
    
    # Count chassis per team
    chassis_counts = recent_chassis.groupby('constructorId').size().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(chassis_counts)))
    
    bars = ax.barh(chassis_counts.index, chassis_counts.values, color=colors)
    
    for bar, count in zip(bars, chassis_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=11)
    
    ax.set_xlabel('Number of Chassis Models', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title('F1 Chassis Models by Team (Historical Database)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_chassis_timeline(save_name="chassis_timeline"):
    """
    Timeline showing chassis evolution for top teams.
    """
    chassis_df = load_chassis_data()
    
    if chassis_df.empty:
        print(f"  No chassis data available")
        return
    
    # Focus on recent years
    teams_of_interest = {
        'red-bull': 'Red Bull',
        'mercedes': 'Mercedes',
        'ferrari': 'Ferrari',
        'mclaren': 'McLaren',
        'aston-martin': 'Aston Martin'
    }
    
    fig, axes = plt.subplots(len(teams_of_interest), 1, figsize=(16, 12))
    
    for idx, (team_id, team_name) in enumerate(teams_of_interest.items()):
        ax = axes[idx]
        team_chassis = chassis_df[chassis_df['constructorId'] == team_id].tail(8)
        
        y_pos = range(len(team_chassis))
        ax.barh(y_pos, [1] * len(team_chassis), color=plt.cm.tab10(idx))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(team_chassis['name'].values)
        ax.set_xlim(0, 1.2)
        ax.set_title(team_name, fontsize=12, fontweight='bold', loc='left')
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    fig.suptitle('F1 Chassis Evolution - Top Teams (Recent Models)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# HIGH/LOW DOWNFORCE TRACK COMPARISON
# =============================================================================

def analyze_track_downforce(year, track_name, session_type='R'):
    """
    Analyze a track's characteristics for downforce requirements.
    Returns metrics for comparison.
    """
    try:
        session = fastf1.get_session(year, track_name, session_type)
        session.load()
    except Exception as e:
        print(f"  Error loading {track_name}: {e}")
        return None
    
    laps = session.laps
    fastest = laps.pick_fastest()
    
    if fastest is None:
        return None
    
    tel = fastest.get_telemetry()
    
    if tel is None or tel.empty:
        return None
    
    # Calculate metrics
    max_speed = tel['Speed'].max()
    avg_speed = tel['Speed'].mean()
    min_speed = tel['Speed'].min()
    
    # Corner vs straight ratio
    corner_threshold = max_speed * 0.75
    corner_points = (tel['Speed'] < corner_threshold).sum()
    straight_points = (tel['Speed'] >= corner_threshold).sum()
    corner_ratio = corner_points / (corner_points + straight_points) if (corner_points + straight_points) > 0 else 0
    
    # Speed variance (high variance = more corners)
    speed_std = tel['Speed'].std()
    
    return {
        'track': track_name,
        'max_speed': max_speed,
        'avg_speed': avg_speed,
        'min_speed': min_speed,
        'corner_ratio': corner_ratio,
        'speed_std': speed_std,
        'session': session
    }


def plot_high_low_downforce_comparison(save_name="downforce_comparison"):
    """
    Compare Monaco (high downforce) vs Monza (low downforce).
    """
    print("\n  Analyzing Monaco (High Downforce)...")
    monaco = analyze_track_downforce(2024, 'Monaco')
    
    print("  Analyzing Monza (Low Downforce)...")
    monza = analyze_track_downforce(2024, 'Italian Grand Prix')
    
    if monaco is None or monza is None:
        print(f"  Could not load track data for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Speed comparison bars
    ax1 = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    
    monaco_speeds = [monaco['max_speed'], monaco['avg_speed'], monaco['min_speed']]
    monza_speeds = [monza['max_speed'], monza['avg_speed'], monza['min_speed']]
    
    bars1 = ax1.bar(x - width/2, monaco_speeds, width, label='Monaco (High DF)', color='#ff6666')
    bars2 = ax1.bar(x + width/2, monza_speeds, width, label='Monza (Low DF)', color='#6666ff')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Max Speed', 'Avg Speed', 'Min Speed'])
    ax1.set_ylabel('Speed (km/h)', fontsize=12)
    ax1.set_title('Speed Comparison: Monaco vs Monza', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Corner ratio comparison
    ax2 = axes[0, 1]
    tracks = ['Monaco\n(High Downforce)', 'Monza\n(Low Downforce)']
    ratios = [monaco['corner_ratio'] * 100, monza['corner_ratio'] * 100]
    colors = ['#ff6666', '#6666ff']
    
    bars = ax2.bar(tracks, ratios, color=colors)
    ax2.set_ylabel('Time in Corners (%)', fontsize=12)
    ax2.set_title('Corner vs Straight Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, ratio in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # 3. Monaco speed trace
    ax3 = axes[1, 0]
    tel_monaco = monaco['session'].laps.pick_fastest().get_telemetry()
    ax3.plot(tel_monaco['Distance'], tel_monaco['Speed'], color='#ff6666', linewidth=1.5)
    ax3.fill_between(tel_monaco['Distance'], 0, tel_monaco['Speed'], alpha=0.3, color='#ff6666')
    ax3.set_xlabel('Distance (m)', fontsize=12)
    ax3.set_ylabel('Speed (km/h)', fontsize=12)
    ax3.set_title('Monaco Speed Profile (High Downforce Required)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Monza speed trace
    ax4 = axes[1, 1]
    tel_monza = monza['session'].laps.pick_fastest().get_telemetry()
    ax4.plot(tel_monza['Distance'], tel_monza['Speed'], color='#6666ff', linewidth=1.5)
    ax4.fill_between(tel_monza['Distance'], 0, tel_monza['Speed'], alpha=0.3, color='#6666ff')
    ax4.set_xlabel('Distance (m)', fontsize=12)
    ax4.set_ylabel('Speed (km/h)', fontsize=12)
    ax4.set_title('Monza Speed Profile (Low Downforce Required)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('High vs Low Downforce Track Comparison\nMonaco (Street Circuit) vs Monza (Temple of Speed)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_downforce_track_ranking(save_name="downforce_ranking"):
    """
    Rank multiple tracks by downforce requirements.
    """
    tracks = [
        (2024, 'Monaco', 'High'),
        (2024, 'Singapore', 'High'),
        (2024, 'Azerbaijan', 'Medium-High'),
        (2024, 'Belgium', 'Medium'),
        (2024, 'British Grand Prix', 'Medium'),
        (2024, 'Italian Grand Prix', 'Low'),
    ]
    
    track_data = []
    
    for year, track, category in tracks:
        print(f"  Analyzing {track}...")
        data = analyze_track_downforce(year, track)
        if data:
            data['category'] = category
            track_data.append(data)
    
    if not track_data:
        print(f"  No track data available for ranking")
        return
    
    df = pd.DataFrame(track_data)
    df = df.sort_values('avg_speed', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by category
    category_colors = {
        'High': '#ff6666',
        'Medium-High': '#ffaa66',
        'Medium': '#ffff66',
        'Medium-Low': '#aaffaa',
        'Low': '#66ff66'
    }
    
    colors = [category_colors.get(cat, '#888888') for cat in df['category']]
    
    bars = ax.barh(df['track'], df['avg_speed'], color=colors)
    
    for bar, speed, cat in zip(bars, df['avg_speed'], df['category']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{speed:.0f} km/h ({cat})', va='center', fontsize=10)
    
    ax.set_xlabel('Average Lap Speed (km/h)', fontsize=12)
    ax.set_ylabel('Track', fontsize=12)
    ax.set_title('Track Downforce Requirements\n(Lower avg speed = Higher downforce needed)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    patches = [mpatches.Patch(color=color, label=cat) 
               for cat, color in category_colors.items()]
    ax.legend(handles=patches, loc='lower right', title='Downforce Level')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def analyze_session_complete(year, event_name):
    """Run all car/aero analysis for a session."""
    print(f"\n{'='*60}")
    print(f"LOADING: {year} {event_name} (Race)")
    print(f"{'='*60}")
    
    try:
        session = fastf1.get_session(year, event_name, 'R')
        session.load()
    except Exception as e:
        print(f"Error loading session: {e}")
        return None
    
    event_clean = session.event['EventName'].replace(' ', '_')
    prefix = f"{year}_{event_clean}"
    
    print("\nGenerating DRS analysis...")
    plot_drs_zones(session, f"{prefix}_drs_zones")
    plot_drs_usage_by_driver(session, f"{prefix}_drs_usage_drivers")
    plot_drs_vs_speed(session, save_name=f"{prefix}_drs_speed")
    
    print("\nGenerating speed profile analysis...")
    plot_cornering_speed_analysis(session, f"{prefix}_cornering_speed")
    plot_speed_trace_comparison(session, save_name=f"{prefix}_speed_trace")
    plot_sector_speed_breakdown(session, f"{prefix}_sector_speeds")
    
    return session


def main():
    print("=" * 60)
    print("F1 CAR & AERODYNAMIC ANALYSIS")
    print("=" * 60)
    
    # Chassis analysis
    print("\n" + "=" * 60)
    print("CHASSIS USAGE ANALYSIS")
    print("=" * 60)
    plot_chassis_by_team()
    plot_chassis_timeline()
    
    # Session analysis for multiple tracks
    print("\n" + "=" * 60)
    print("DRS & SPEED PROFILE ANALYSIS")
    print("=" * 60)
    
    sessions_to_analyze = [
        (2024, 'Monaco'),
        (2024, 'Italian Grand Prix'),
        (2024, 'Belgium'),
    ]
    
    for year, event in sessions_to_analyze:
        analyze_session_complete(year, event)
    
    # Downforce comparison
    print("\n" + "=" * 60)
    print("DOWNFORCE COMPARISON ANALYSIS")
    print("=" * 60)
    plot_high_low_downforce_comparison()
    plot_downforce_track_ranking()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List all created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
