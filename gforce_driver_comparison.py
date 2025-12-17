# -*- coding: utf-8 -*-
"""
G-Force Driver Comparison Across Tracks
Creates circular and diamond G-force gauges comparing drivers.
"""

import fastf1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

fastf1.Cache.enable_cache('cache')
plt.style.use('dark_background')

OUTPUT_DIR = Path('gforce_visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_gforce(tel, smoothing=5):
    """Calculate longitudinal and lateral G-forces."""
    speed_ms = tel['Speed'].values / 3.6
    time_s = tel['Time'].dt.total_seconds().values
    dt = np.diff(time_s)
    dt = np.append(dt, dt[-1])
    dt = np.where(dt == 0, 0.01, dt)
    
    dv = np.diff(speed_ms)
    dv = np.append(dv, 0)
    long_g = gaussian_filter1d(dv / dt, sigma=smoothing) / 9.81
    
    x, y = tel['X'].values, tel['Y'].values
    dx, dy = np.diff(x), np.diff(y)
    dx, dy = np.append(dx, dx[-1]), np.append(dy, dy[-1])
    ds = np.sqrt(dx**2 + dy**2)
    ds = np.where(ds == 0, 0.001, ds)
    heading = np.arctan2(dy, dx)
    dheading = np.diff(heading)
    dheading = np.append(dheading, 0)
    dheading = np.where(dheading > np.pi, dheading - 2*np.pi, dheading)
    dheading = np.where(dheading < -np.pi, dheading + 2*np.pi, dheading)
    curvature = gaussian_filter1d(dheading / ds, sigma=smoothing)
    lat_g = (speed_ms**2 * curvature) / 9.81
    
    return long_g, lat_g


def draw_circular_gauge(ax, long_g, lat_g, driver, color):
    """Draw circular G-meter with driver trace."""
    for r in [5, 3.5, 2]:
        circle = plt.Circle((0, 0), r, fill=False, color='#444444', 
                            linewidth=1.5 if r==5 else 1, 
                            linestyle='-' if r==5 else '--')
        ax.add_patch(circle)
    
    ax.axhline(y=0, color='#333333', linewidth=1)
    ax.axvline(x=0, color='#333333', linewidth=1)
    
    gx = np.clip(lat_g, -5, 5)
    gy = np.clip(-long_g, -5, 5)
    ax.scatter(gx, gy, c=color, alpha=0.3, s=2)
    
    max_brake_idx = np.argmin(long_g)
    max_lat_idx = np.argmax(np.abs(lat_g))
    
    ax.plot(gx[max_brake_idx], gy[max_brake_idx], 'o', color='#ff0000', markersize=10, 
            markeredgecolor='white', markeredgewidth=2, label=f'Max Brake: {-long_g[max_brake_idx]:.1f}G')
    ax.plot(gx[max_lat_idx], gy[max_lat_idx], 's', color='#ffff00', markersize=10,
            markeredgecolor='white', markeredgewidth=2, label=f'Max Lateral: {abs(lat_g[max_lat_idx]):.1f}G')
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(driver, fontsize=14, fontweight='bold', color=color)


def compare_drivers_on_track(year, track, df_level):
    """Create driver comparison for a single track."""
    print(f'Loading {track}...')
    session = fastf1.get_session(year, track, 'R')
    session.load()
    
    drivers = session.laps['Driver'].unique()[:4]
    driver_colors = ['#00d2be', '#dc0000', '#0600ef', '#ff8700']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    driver_stats = []
    
    for i, driver in enumerate(drivers):
        lap = session.laps.pick_driver(driver).pick_fastest()
        if lap is None:
            continue
        
        try:
            tel = lap.get_telemetry()
            long_g, lat_g = calculate_gforce(tel)
            
            ax = axes[0, i]
            draw_circular_gauge(ax, long_g, lat_g, driver, driver_colors[i])
            ax.legend(loc='lower right', fontsize=8)
            
            stats = {
                'Driver': driver,
                'MaxBrake': -np.min(long_g),
                'MaxAccel': np.max(long_g),
                'MaxLat': np.max(np.abs(lat_g)),
                'AvgG': np.mean(np.sqrt(long_g**2 + lat_g**2))
            }
            driver_stats.append(stats)
            
        except Exception as e:
            print(f'  Error with {driver}: {e}')
    
    if driver_stats:
        ax_bars = axes[1, :]
        
        ax_bars[0].bar([s['Driver'] for s in driver_stats], [s['MaxBrake'] for s in driver_stats], 
                      color=['#ff0000']*len(driver_stats))
        ax_bars[0].set_title('Max Braking G', fontsize=12, fontweight='bold')
        ax_bars[0].set_ylabel('G-Force')
        ax_bars[0].grid(True, alpha=0.3, axis='y')
        for j, s in enumerate(driver_stats):
            ax_bars[0].text(j, s['MaxBrake']+0.1, f"{s['MaxBrake']:.1f}", ha='center', fontsize=10)
        
        ax_bars[1].bar([s['Driver'] for s in driver_stats], [s['MaxAccel'] for s in driver_stats],
                      color=['#00ff00']*len(driver_stats))
        ax_bars[1].set_title('Max Acceleration G', fontsize=12, fontweight='bold')
        ax_bars[1].grid(True, alpha=0.3, axis='y')
        for j, s in enumerate(driver_stats):
            ax_bars[1].text(j, s['MaxAccel']+0.05, f"{s['MaxAccel']:.1f}", ha='center', fontsize=10)
        
        ax_bars[2].bar([s['Driver'] for s in driver_stats], [s['MaxLat'] for s in driver_stats],
                      color=['#ffff00']*len(driver_stats))
        ax_bars[2].set_title('Max Lateral G', fontsize=12, fontweight='bold')
        ax_bars[2].grid(True, alpha=0.3, axis='y')
        for j, s in enumerate(driver_stats):
            ax_bars[2].text(j, s['MaxLat']+0.1, f"{s['MaxLat']:.1f}", ha='center', fontsize=10)
        
        ax_bars[3].bar([s['Driver'] for s in driver_stats], [s['AvgG'] for s in driver_stats],
                      color=['#ff00ff']*len(driver_stats))
        ax_bars[3].set_title('Average Total G', fontsize=12, fontweight='bold')
        ax_bars[3].grid(True, alpha=0.3, axis='y')
        for j, s in enumerate(driver_stats):
            ax_bars[3].text(j, s['AvgG']+0.05, f"{s['AvgG']:.1f}", ha='center', fontsize=10)
    
    event_name = session.event['EventName'].replace(' ', '_')
    fig.suptitle(f"{session.event['EventName']} - Driver G-Force Comparison\n({df_level})", 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{event_name}_driver_gforce_comparison.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'  Saved: {event_name}_driver_gforce_comparison.png')
    
    return session


def create_cross_track_comparison(tracks):
    """Create cross-track G-force comparison."""
    print('\nCreating cross-track comparison...')
    
    fig, axes = plt.subplots(3, 5, figsize=(24, 16))
    
    for row, (year, track, df_level) in enumerate(tracks):
        session = fastf1.get_session(year, track, 'R')
        session.load()
        
        lap = session.laps.pick_fastest()
        driver = lap['Driver']
        tel = lap.get_telemetry()
        long_g, lat_g = calculate_gforce(tel)
        
        # Circular gauge
        ax = axes[row, 0]
        for r in [5, 3.5, 2]:
            circle = plt.Circle((0, 0), r, fill=False, color='#444444', linewidth=1.5 if r==5 else 1)
            ax.add_patch(circle)
        ax.axhline(y=0, color='#333333', linewidth=1)
        ax.axvline(x=0, color='#333333', linewidth=1)
        gx = np.clip(lat_g, -5, 5)
        gy = np.clip(-long_g, -5, 5)
        ax.scatter(gx, gy, c=np.sqrt(gx**2 + gy**2), cmap='plasma', alpha=0.5, s=3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{session.event['EventName']}\n{df_level}", fontsize=12, fontweight='bold')
        
        # Diamond gauge
        ax2 = axes[row, 1]
        diamond = patches.Polygon([(0, 5), (5, 0), (0, -5), (-5, 0)], fill=False, edgecolor='#444444', linewidth=2)
        ax2.add_patch(diamond)
        ax2.plot([0, 0], [-5, 5], color='#333333', linewidth=1)
        ax2.plot([-5, 5], [0, 0], color='#333333', linewidth=1)
        ax2.scatter(gx, gy, c=np.sqrt(gx**2 + gy**2), cmap='plasma', alpha=0.5, s=3)
        ax2.set_xlim(-6.5, 6.5)
        ax2.set_ylim(-6.5, 6.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Diamond View - {driver}', fontsize=10)
        
        # Stats bars
        stats = [-np.min(long_g), np.max(long_g), np.max(np.abs(lat_g))]
        labels = ['Max Brake', 'Max Accel', 'Max Lateral']
        colors = ['#ff0000', '#00ff00', '#ffff00']
        
        ax3 = axes[row, 2]
        bars = ax3.bar(labels, stats, color=colors)
        for bar, val in zip(bars, stats):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.1f}G', ha='center', va='bottom')
        ax3.set_ylabel('G-Force')
        ax3.set_title('Peak G-Forces', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # G distribution
        ax4 = axes[row, 3]
        total_g = np.sqrt(long_g**2 + lat_g**2)
        ax4.hist(total_g, bins=30, color='#ff00ff', alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.mean(total_g), color='#00ffff', linewidth=2, linestyle='--', label=f'Avg: {np.mean(total_g):.1f}G')
        ax4.set_xlabel('Total G')
        ax4.set_ylabel('Frequency')
        ax4.set_title('G-Force Distribution', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Time above thresholds
        ax5 = axes[row, 4]
        thresholds = [3, 4, 5]
        pcts = [(total_g > t).sum() / len(total_g) * 100 for t in thresholds]
        bars = ax5.bar([f'>{t}G' for t in thresholds], pcts, color=['#00ff00', '#ffff00', '#ff0000'])
        for bar, pct in zip(bars, pcts):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{pct:.0f}%', ha='center', va='bottom')
        ax5.set_ylabel('% of Lap')
        ax5.set_title('Time Above G Threshold', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('G-Force Comparison: Monaco vs Spa vs Monza\n(High vs Medium vs Low Downforce Tracks)', 
                fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gforce_track_gauge_comparison.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print('Saved: gforce_track_gauge_comparison.png')


def main():
    print("=" * 60)
    print("G-FORCE DRIVER COMPARISON")
    print("=" * 60)
    
    tracks = [
        (2024, 'Monaco', 'High Downforce'),
        (2024, 'Belgium', 'Medium Downforce'),
        (2024, 'Italian Grand Prix', 'Low Downforce'),
    ]
    
    # Per-track driver comparisons
    for year, track, df_level in tracks:
        compare_drivers_on_track(year, track, df_level)
    
    # Cross-track comparison
    create_cross_track_comparison(tracks)
    
    print("\n" + "=" * 60)
    print("ALL COMPARISONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
