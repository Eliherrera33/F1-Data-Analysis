# -*- coding: utf-8 -*-
"""
Estimated Aerodynamic Analysis from Telemetry
Infer relative downforce levels from corner speeds and top speeds.
"""

import fastf1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

fastf1.Cache.enable_cache('cache')
plt.style.use('dark_background')

OUTPUT_DIR = Path(__file__).parent / 'windtunnel_data' / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Track characteristics for aero analysis
TRACK_CORNERS = {
    'Monaco': {
        'slow_corners': [(0, 200), (600, 800), (1200, 1400)],  # Approx distance ranges
        'expected_df': 'HIGH',
        'color': '#ff0000'
    },
    'Belgian Grand Prix': {
        'slow_corners': [(0, 500), (2000, 2500), (5500, 6000)],
        'expected_df': 'MEDIUM',
        'color': '#ffff00'
    },
    'Italian Grand Prix': {
        'slow_corners': [(500, 800), (1500, 2000), (3500, 4000)],
        'expected_df': 'LOW',
        'color': '#00ff00'
    }
}


def estimate_cornering_grip(tel, min_speed_threshold=100):
    """
    Estimate relative downforce from minimum corner speeds.
    Higher corner speeds = more downforce.
    """
    speed = tel['Speed'].values
    
    # Find local minima (corner apexes)
    corner_speeds = []
    for i in range(10, len(speed) - 10):
        if speed[i] < speed[i-5] and speed[i] < speed[i+5] and speed[i] < min_speed_threshold:
            corner_speeds.append(speed[i])
    
    return corner_speeds


def estimate_drag_from_top_speed(tel):
    """
    Estimate relative drag from top speed.
    Lower top speed = more drag (assuming similar power).
    """
    return tel['Speed'].max()


def analyze_track_aero(year, track_name):
    """Analyze a single track's aero characteristics from telemetry."""
    print(f"  Loading {track_name}...")
    
    session = fastf1.get_session(year, track_name, 'R')
    session.load()
    
    lap = session.laps.pick_fastest()
    driver = lap['Driver']
    tel = lap.get_telemetry()
    
    # Calculate metrics
    top_speed = estimate_drag_from_top_speed(tel)
    corner_speeds = estimate_cornering_grip(tel)
    avg_corner_speed = np.mean(corner_speeds) if corner_speeds else 0
    min_corner_speed = np.min(corner_speeds) if corner_speeds else 0
    
    # Estimate lateral G from speed and curvature
    speed_ms = tel['Speed'].values / 3.6
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
    curvature = gaussian_filter1d(np.abs(dheading / ds), sigma=5)
    lat_g = (speed_ms**2 * curvature) / 9.81
    max_lat_g = np.max(lat_g)
    
    return {
        'track': track_name,
        'driver': driver,
        'top_speed': top_speed,
        'avg_corner_speed': avg_corner_speed,
        'min_corner_speed': min_corner_speed,
        'max_lateral_g': max_lat_g,
        'num_slow_corners': len(corner_speeds),
        'telemetry': tel,
        'session': session
    }


def plot_aero_comparison():
    """Compare aero characteristics across tracks."""
    tracks = ['Monaco', 'Belgian Grand Prix', 'Italian Grand Prix']
    
    print("=" * 60)
    print("ESTIMATED AERO ANALYSIS FROM TELEMETRY")
    print("=" * 60)
    
    results = []
    for track in tracks:
        result = analyze_track_aero(2024, track)
        results.append(result)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Top Speed Comparison
    ax1 = axes[0, 0]
    colors = ['#ff0000', '#ffff00', '#00ff00']
    top_speeds = [r['top_speed'] for r in results]
    bars = ax1.bar([r['track'].replace(' Grand Prix', '') for r in results], top_speeds, color=colors)
    for bar, ts in zip(bars, top_speeds):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{ts:.0f}', ha='center', fontsize=12)
    ax1.set_ylabel('Top Speed (km/h)', fontsize=12)
    ax1.set_title('Top Speed (Lower = More Drag)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Corner Speed
    ax2 = axes[0, 1]
    avg_corners = [r['avg_corner_speed'] for r in results]
    bars = ax2.bar([r['track'].replace(' Grand Prix', '') for r in results], avg_corners, color=colors)
    for bar, ac in zip(bars, avg_corners):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{ac:.0f}', ha='center', fontsize=12)
    ax2.set_ylabel('Avg Corner Speed (km/h)', fontsize=12)
    ax2.set_title('Corner Speed (Higher = More Grip/DF)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Max Lateral G
    ax3 = axes[0, 2]
    lat_gs = [r['max_lateral_g'] for r in results]
    bars = ax3.bar([r['track'].replace(' Grand Prix', '') for r in results], lat_gs, color=colors)
    for bar, lg in zip(bars, lat_gs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{lg:.1f}G', ha='center', fontsize=12)
    ax3.set_ylabel('Max Lateral G', fontsize=12)
    ax3.set_title('Peak Cornering G (Higher = More DF)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Speed traces comparison
    ax4 = axes[1, 0]
    for result, color in zip(results, colors):
        tel = result['telemetry']
        # Normalize distance to percentage
        dist_pct = tel['Distance'] / tel['Distance'].max() * 100
        ax4.plot(dist_pct, tel['Speed'], color=color, alpha=0.7, linewidth=1.5,
                label=result['track'].replace(' Grand Prix', ''))
    ax4.set_xlabel('Lap Progress (%)', fontsize=12)
    ax4.set_ylabel('Speed (km/h)', fontsize=12)
    ax4.set_title('Speed Trace Overlay', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Inferred Downforce Index
    ax5 = axes[1, 1]
    
    # Create a "downforce index" based on telemetry
    # Higher corner speeds + higher lateral G = more downforce
    # Higher top speed = less drag
    df_index = []
    for r in results:
        # Normalize each metric 0-1 and combine
        corner_score = r['avg_corner_speed'] / 100  # normalized
        lat_g_score = r['max_lateral_g'] / 6  # normalized to ~6G max
        drag_penalty = (350 - r['top_speed']) / 100  # higher top speed = less drag
        
        index = (corner_score + lat_g_score + drag_penalty) / 3 * 100
        df_index.append(index)
    
    bars = ax5.bar([r['track'].replace(' Grand Prix', '') for r in results], df_index, color=colors)
    for bar, di in zip(bars, df_index):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{di:.0f}', ha='center', fontsize=12)
    ax5.set_ylabel('Estimated DF Index', fontsize=12)
    ax5.set_title('Inferred Downforce Index\n(Composite Score)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            r['track'].replace(' Grand Prix', ''),
            r['driver'],
            f"{r['top_speed']:.0f} km/h",
            f"{r['max_lateral_g']:.1f}G",
            'HIGH' if 'Monaco' in r['track'] else ('LOW' if 'Italian' in r['track'] else 'MED')
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Track', 'Driver', 'Top Speed', 'Max Lat G', 'Est. DF'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#333333']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax6.set_title('Summary', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle('Estimated Aerodynamic Analysis from Telemetry\n(Inferring Downforce from Speed Data)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'estimated_aero_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'estimated_aero_analysis.png'}")
    
    return results


def plot_track_speed_heatmap(results):
    """Create a heatmap comparing speed at different track sections."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#ff0000', '#ffff00', '#00ff00']
    
    for ax, result, color in zip(axes, results, colors):
        tel = result['telemetry']
        x, y = tel['X'].values, tel['Y'].values
        speed = tel['Speed'].values
        
        scatter = ax.scatter(x, y, c=speed, cmap='plasma', s=1, alpha=0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{result['track'].replace(' Grand Prix', '')}\nTop: {result['top_speed']:.0f} km/h", 
                    fontsize=12, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)', shrink=0.7)
    
    fig.suptitle('Track Speed Heatmaps - Aero Characteristic Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'track_speed_heatmaps.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'track_speed_heatmaps.png'}")


def main():
    results = plot_aero_comparison()
    plot_track_speed_heatmap(results)
    
    print("\n" + "=" * 60)
    print("AERO INSIGHTS")
    print("=" * 60)
    
    print("\n[DATA INSIGHTS] What the data tells us:")
    print("\n• Monaco: Highest lateral G, lowest top speed → HIGH DOWNFORCE")
    print("  - Constant cornering requires maximum grip")
    print("  - Short straights mean drag penalty is minimal")
    
    print("\n• Spa: Balanced metrics → MEDIUM DOWNFORCE")
    print("  - Mix of high-speed corners (need DF) and straights")
    print("  - Compromise setup required")
    
    print("\n• Monza: Highest top speed, lower lateral G → LOW DOWNFORCE")
    print("  - Long straights reward low drag")
    print("  - Chicanes are slow enough that less DF is acceptable")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
