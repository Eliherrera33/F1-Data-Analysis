# -*- coding: utf-8 -*-
"""
F1 G-Force Analysis
Calculate and visualize longitudinal and lateral G-forces from telemetry.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
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
OUTPUT_DIR = BASE_DIR / "gforce_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_gforce(telemetry, smoothing_sigma=3):
    """
    Calculate longitudinal and lateral G-forces from telemetry data.
    
    Longitudinal G: From acceleration/braking (speed changes)
    Lateral G: From cornering (centripetal acceleration = v²/r)
    
    Args:
        telemetry: FastF1 telemetry DataFrame
        smoothing_sigma: Gaussian smoothing factor to reduce noise
    
    Returns:
        DataFrame with G-force columns added
    """
    tel = telemetry.copy()
    
    # Convert speed to m/s
    speed_ms = tel['Speed'].values / 3.6
    
    # Get time in seconds
    time_s = tel['Time'].dt.total_seconds().values
    dt = np.diff(time_s)
    dt = np.append(dt, dt[-1])  # Pad to same length
    dt = np.where(dt == 0, 0.01, dt)  # Avoid division by zero
    
    # =====  LONGITUDINAL G-FORCE (acceleration/braking) =====
    # a = dv/dt
    dv = np.diff(speed_ms)
    dv = np.append(dv, 0)
    
    accel = dv / dt
    accel_smoothed = gaussian_filter1d(accel, sigma=smoothing_sigma)
    long_g = accel_smoothed / 9.81
    
    # =====  LATERAL G-FORCE (cornering) =====
    # lat_g = v² / (r * g) where r is turn radius
    # Estimate radius from position changes
    x = tel['X'].values
    y = tel['Y'].values
    
    if np.isnan(x).all() or np.isnan(y).all():
        lat_g = np.zeros(len(tel))
    else:
        # Calculate heading direction
        dx = np.diff(x)
        dy = np.diff(y)
        dx = np.append(dx, dx[-1])
        dy = np.append(dy, dy[-1])
        
        # Distance traveled
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.where(ds == 0, 0.001, ds)
        
        # Heading angle
        heading = np.arctan2(dy, dx)
        
        # Rate of change of heading (curvature * ds = dθ)
        dheading = np.diff(heading)
        dheading = np.append(dheading, 0)
        
        # Wrap angles to [-π, π]
        dheading = np.where(dheading > np.pi, dheading - 2*np.pi, dheading)
        dheading = np.where(dheading < -np.pi, dheading + 2*np.pi, dheading)
        
        # Curvature κ = dθ/ds
        curvature = dheading / ds
        curvature_smoothed = gaussian_filter1d(curvature, sigma=smoothing_sigma)
        
        # Lateral acceleration = v² * κ
        lat_accel = speed_ms**2 * np.abs(curvature_smoothed)
        lat_g = lat_accel / 9.81
    
    # =====  COMBINED G-FORCE =====
    total_g = np.sqrt(long_g**2 + lat_g**2)
    
    tel['LongG'] = long_g
    tel['LatG'] = lat_g
    tel['TotalG'] = total_g
    
    return tel


def plot_gforce_trace(session, driver=None, save_name="gforce_trace"):
    """
    Plot G-force traces over lap distance.
    """
    laps = session.laps
    
    if driver:
        lap = laps.pick_driver(driver).pick_fastest()
    else:
        lap = laps.pick_fastest()
        driver = lap['Driver']
    
    if lap is None:
        print(f"  No lap data for {driver}")
        return
    
    tel = lap.get_telemetry()
    tel = calculate_gforce(tel)
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    
    # Speed
    ax1 = axes[0]
    ax1.plot(tel['Distance'], tel['Speed'], color='#00ffff', linewidth=1.5)
    ax1.set_ylabel('Speed (km/h)', fontsize=11)
    ax1.set_title(f'{session.event["EventName"]} - {driver} G-Force Analysis', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Longitudinal G
    ax2 = axes[1]
    colors = np.where(tel['LongG'] > 0, '#00ff00', '#ff0000')
    ax2.fill_between(tel['Distance'], 0, tel['LongG'], 
                    where=tel['LongG'] > 0, color='#00ff00', alpha=0.5, label='Acceleration')
    ax2.fill_between(tel['Distance'], 0, tel['LongG'], 
                    where=tel['LongG'] < 0, color='#ff0000', alpha=0.5, label='Braking')
    ax2.axhline(y=0, color='white', linewidth=0.5)
    ax2.set_ylabel('Longitudinal G', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-6, 3)
    
    # Lateral G
    ax3 = axes[2]
    ax3.fill_between(tel['Distance'], 0, tel['LatG'], color='#ffff00', alpha=0.6)
    ax3.set_ylabel('Lateral G', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 6)
    
    # Total G
    ax4 = axes[3]
    ax4.plot(tel['Distance'], tel['TotalG'], color='#ff00ff', linewidth=1.5)
    ax4.fill_between(tel['Distance'], 0, tel['TotalG'], color='#ff00ff', alpha=0.3)
    ax4.set_xlabel('Distance (m)', fontsize=11)
    ax4.set_ylabel('Total G', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_gforce_histogram(session, save_name="gforce_histogram"):
    """
    Histogram of G-force distribution.
    """
    lap = session.laps.pick_fastest()
    if lap is None:
        print(f"  No fastest lap available")
        return
    
    tel = lap.get_telemetry()
    tel = calculate_gforce(tel)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Longitudinal G
    ax1 = axes[0]
    ax1.hist(tel['LongG'], bins=50, color='#ff6666', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='white', linestyle='--')
    ax1.set_xlabel('Longitudinal G', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Braking (-G) / Acceleration (+G)\nMax Brake: {tel["LongG"].min():.1f}G', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Lateral G
    ax2 = axes[1]
    ax2.hist(tel['LatG'], bins=50, color='#ffff00', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Lateral G', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Cornering G-Force\nMax: {tel["LatG"].max():.1f}G', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Total G
    ax3 = axes[2]
    ax3.hist(tel['TotalG'], bins=50, color='#ff00ff', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Total G', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Combined G-Force\nMax: {tel["TotalG"].max():.1f}G', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f'{session.event["EventName"]} - G-Force Distribution', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_gforce_comparison(session, save_name="gforce_comparison"):
    """
    Compare G-forces across drivers.
    """
    laps = session.laps
    drivers = laps['Driver'].unique()
    
    gforce_data = []
    
    for driver in drivers:
        fastest = laps.pick_driver(driver).pick_fastest()
        if fastest is None:
            continue
        
        try:
            tel = fastest.get_telemetry()
            tel = calculate_gforce(tel)
            
            gforce_data.append({
                'Driver': driver,
                'MaxBrakeG': abs(tel['LongG'].min()),
                'MaxAccelG': tel['LongG'].max(),
                'MaxLatG': tel['LatG'].max(),
                'MaxTotalG': tel['TotalG'].max(),
                'AvgG': tel['TotalG'].mean()
            })
        except:
            continue
    
    if not gforce_data:
        print(f"  No G-force data available")
        return
    
    df = pd.DataFrame(gforce_data)
    df = df.sort_values('MaxTotalG', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = range(len(df))
    width = 0.2
    
    ax.barh([i - width*1.5 for i in x], df['MaxBrakeG'], width, 
            label='Max Braking G', color='#ff0000')
    ax.barh([i - width*0.5 for i in x], df['MaxLatG'], width, 
            label='Max Lateral G', color='#ffff00')
    ax.barh([i + width*0.5 for i in x], df['MaxTotalG'], width, 
            label='Max Total G', color='#ff00ff')
    ax.barh([i + width*1.5 for i in x], df['AvgG'], width, 
            label='Avg Total G', color='#00ffff')
    
    ax.set_yticks(x)
    ax.set_yticklabels(df['Driver'])
    ax.set_xlabel('G-Force', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title(f'{session.event["EventName"]} - G-Force Comparison by Driver', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_gforce_map(session, driver=None, save_name="gforce_map"):
    """
    Track map colored by G-force.
    """
    laps = session.laps
    
    if driver:
        lap = laps.pick_driver(driver).pick_fastest()
    else:
        lap = laps.pick_fastest()
        driver = lap['Driver']
    
    if lap is None:
        print(f"  No lap data for {driver}")
        return
    
    tel = lap.get_telemetry()
    tel = calculate_gforce(tel)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Lateral G map
    ax1 = axes[0]
    scatter1 = ax1.scatter(tel['X'], tel['Y'], c=tel['LatG'], 
                          cmap='YlOrRd', s=5, vmin=0, vmax=5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Lateral G (Cornering)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='G-Force')
    
    # Braking G map
    ax2 = axes[1]
    brake_g = -tel['LongG']  # Invert for braking (positive = braking)
    brake_g = np.clip(brake_g, 0, 6)  # Only show braking
    scatter2 = ax2.scatter(tel['X'], tel['Y'], c=brake_g, 
                          cmap='Reds', s=5, vmin=0, vmax=5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Braking G', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='G-Force')
    
    # Total G map
    ax3 = axes[2]
    scatter3 = ax3.scatter(tel['X'], tel['Y'], c=tel['TotalG'], 
                          cmap='plasma', s=5, vmin=0, vmax=6)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Total G', fontsize=14, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='G-Force')
    
    fig.suptitle(f'{session.event["EventName"]} - {driver} G-Force Track Map', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_gforce_track_comparison(save_name="gforce_track_comparison"):
    """
    Compare G-forces across different tracks.
    """
    tracks = [
        (2024, 'Monaco', 'High Downforce'),
        (2024, 'Belgium', 'Medium Downforce'),
        (2024, 'Italian Grand Prix', 'Low Downforce'),
    ]
    
    track_data = []
    
    for year, track, category in tracks:
        print(f"  Loading {track}...")
        try:
            session = fastf1.get_session(year, track, 'R')
            session.load()
            
            lap = session.laps.pick_fastest()
            tel = lap.get_telemetry()
            tel = calculate_gforce(tel)
            
            track_data.append({
                'Track': session.event['EventName'],
                'Category': category,
                'MaxBrakeG': abs(tel['LongG'].min()),
                'MaxLatG': tel['LatG'].max(),
                'MaxTotalG': tel['TotalG'].max(),
                'AvgG': tel['TotalG'].mean(),
                'Time95pctAbove3G': (tel['TotalG'] > 3).sum() / len(tel) * 100
            })
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if not track_data:
        print(f"  No track data available")
        return
    
    df = pd.DataFrame(track_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Max G comparison
    ax1 = axes[0]
    x = range(len(df))
    width = 0.25
    
    ax1.bar([i - width for i in x], df['MaxBrakeG'], width, label='Max Brake G', color='#ff0000')
    ax1.bar(x, df['MaxLatG'], width, label='Max Lateral G', color='#ffff00')
    ax1.bar([i + width for i in x], df['MaxTotalG'], width, label='Max Total G', color='#ff00ff')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}\n({c})" for t, c in zip(df['Track'], df['Category'])], fontsize=10)
    ax1.set_ylabel('G-Force', fontsize=12)
    ax1.set_title('Maximum G-Forces by Track', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Time above 3G
    ax2 = axes[1]
    colors = ['#ff6666', '#ffff66', '#66ff66']
    bars = ax2.bar(df['Track'], df['Time95pctAbove3G'], color=colors)
    ax2.set_ylabel('% of Lap Above 3G', fontsize=12)
    ax2.set_title('Physical Demand: Time Above 3G', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, df['Time95pctAbove3G']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
    
    fig.suptitle('G-Force Comparison Across Tracks', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def analyze_session_gforce(year, event_name):
    """
    Full G-force analysis for a session.
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
    
    event_clean = session.event['EventName'].replace(' ', '_')
    prefix = f"{year}_{event_clean}"
    
    print("\nGenerating G-force visualizations...")
    
    plot_gforce_trace(session, save_name=f"{prefix}_gforce_trace")
    plot_gforce_histogram(session, f"{prefix}_gforce_histogram")
    plot_gforce_comparison(session, f"{prefix}_gforce_comparison")
    plot_gforce_map(session, save_name=f"{prefix}_gforce_map")
    
    return session


def main():
    print("=" * 60)
    print("F1 G-FORCE ANALYSIS")
    print("=" * 60)
    
    # Analyze multiple tracks
    races = [
        (2024, 'Monaco'),
        (2024, 'Belgium'),
        (2024, 'Italian Grand Prix'),
    ]
    
    for year, event in races:
        analyze_session_gforce(year, event)
    
    # Cross-track comparison
    print("\n" + "=" * 60)
    print("G-FORCE TRACK COMPARISON")
    print("=" * 60)
    plot_gforce_track_comparison()
    
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
