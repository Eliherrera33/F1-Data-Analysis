# -*- coding: utf-8 -*-
"""
F1 Race Simulation Visualizations
Animated race replays for Monaco, Spa, Monza with top drivers
Using FastF1 telemetry data for realistic car positions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import fastf1
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# Setup FastF1 cache
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path(__file__).parent / 'race_simulations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Team colors
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'AlphaTauri': '#5E8FAA',
    'RB': '#6692FF',
    'Alfa Romeo': '#C92D4B',
    'Haas F1 Team': '#B6BABD',
    'Kick Sauber': '#52E252',
}

# Driver abbreviations to full names
DRIVER_NAMES = {
    'VER': 'Verstappen', 'PER': 'Perez', 'LEC': 'Leclerc', 'SAI': 'Sainz',
    'HAM': 'Hamilton', 'RUS': 'Russell', 'NOR': 'Norris', 'PIA': 'Piastri',
    'ALO': 'Alonso', 'STR': 'Stroll', 'OCO': 'Ocon', 'GAS': 'Gasly',
    'BOT': 'Bottas', 'ZHO': 'Zhou', 'MAG': 'Magnussen', 'HUL': 'Hulkenberg',
    'RIC': 'Ricciardo', 'TSU': 'Tsunoda', 'ALB': 'Albon', 'SAR': 'Sargeant',
}


def get_driver_color(driver_info):
    """Get team color for a driver."""
    team = driver_info.get('TeamName', 'Unknown')
    return TEAM_COLORS.get(team, '#888888')


def load_race_data(year, race_name, session_type='R'):
    """Load race session data from FastF1."""
    print(f"  Loading {year} {race_name} {session_type} data...")
    
    try:
        session = fastf1.get_session(year, race_name, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None


def get_top_drivers(session, n_drivers=6):
    """Get top N drivers based on fastest lap."""
    laps = session.laps.pick_quicklaps()
    
    # Get fastest lap per driver
    fastest = laps.groupby('Driver')['LapTime'].min().sort_values()
    top_drivers = fastest.head(n_drivers).index.tolist()
    
    return top_drivers


def create_track_outline(session):
    """Create track outline from telemetry data."""
    # Get fastest lap for track shape
    fastest_lap = session.laps.pick_fastest()
    tel = fastest_lap.get_telemetry()
    
    x = tel['X'].values
    y = tel['Y'].values
    
    return x, y


def animate_race_simulation(session, track_name, n_drivers=6, n_laps=5):
    """Create animated race simulation."""
    print(f"\n  Creating {track_name} race simulation...")
    
    # Get track outline
    track_x, track_y = create_track_outline(session)
    
    # Normalize track coordinates
    x_center = (track_x.max() + track_x.min()) / 2
    y_center = (track_y.max() + track_y.min()) / 2
    scale = max(track_x.max() - track_x.min(), track_y.max() - track_y.min())
    
    track_x_norm = (track_x - x_center) / scale * 10
    track_y_norm = (track_y - y_center) / scale * 10
    
    # Get top drivers
    top_drivers = get_top_drivers(session, n_drivers)
    print(f"  Top drivers: {', '.join(top_drivers)}")
    
    # Get driver telemetry for each driver
    driver_data = {}
    
    for driver in top_drivers:
        try:
            driver_laps = session.laps.pick_driver(driver)
            if len(driver_laps) == 0:
                continue
                
            # Get completed laps
            completed_laps = driver_laps[driver_laps['LapTime'].notna()]
            if len(completed_laps) == 0:
                continue
            
            # Get telemetry for first few laps
            all_tel = []
            lap_count = 0
            
            for _, lap in completed_laps.iterrows():
                if lap_count >= n_laps:
                    break
                try:
                    tel = lap.get_telemetry()
                    if len(tel) > 0:
                        tel_x = (tel['X'].values - x_center) / scale * 10
                        tel_y = (tel['Y'].values - y_center) / scale * 10
                        all_tel.append((tel_x, tel_y, tel['Speed'].values))
                        lap_count += 1
                except:
                    continue
            
            if all_tel:
                # Concatenate all laps
                x_all = np.concatenate([t[0] for t in all_tel])
                y_all = np.concatenate([t[1] for t in all_tel])
                speed_all = np.concatenate([t[2] for t in all_tel])
                
                driver_info = session.get_driver(driver)
                color = get_driver_color(driver_info)
                
                driver_data[driver] = {
                    'x': x_all,
                    'y': y_all,
                    'speed': speed_all,
                    'color': color,
                    'name': DRIVER_NAMES.get(driver, driver),
                    'team': driver_info.get('TeamName', 'Unknown')
                }
        except Exception as e:
            print(f"    Error processing {driver}: {e}")
            continue
    
    if len(driver_data) < 2:
        print("  Not enough driver data, skipping...")
        return
    
    # Create animation
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Determine frame count based on shortest telemetry
    min_frames = min(len(d['x']) for d in driver_data.values())
    n_frames = min(min_frames // 10, 300)  # Subsample for performance
    frame_step = min_frames // n_frames
    
    print(f"  Generating {n_frames} frames...")
    
    def animate(frame):
        ax.clear()
        
        idx = frame * frame_step
        
        # Draw track
        ax.plot(track_x_norm, track_y_norm, color='#333333', linewidth=20, 
                solid_capstyle='round', zorder=1)
        ax.plot(track_x_norm, track_y_norm, color='#1a1a1a', linewidth=16,
                solid_capstyle='round', zorder=2)
        
        # Track markings (white lines)
        ax.plot(track_x_norm, track_y_norm, color='#444444', linewidth=18,
                linestyle='--', dashes=(1, 3), zorder=1)
        
        # Draw start/finish line
        sf_idx = 0
        ax.plot([track_x_norm[sf_idx]-0.3, track_x_norm[sf_idx]+0.3],
               [track_y_norm[sf_idx]-0.3, track_y_norm[sf_idx]+0.3],
               color='white', linewidth=3, zorder=3)
        
        # Draw each driver
        positions = []
        for driver, data in driver_data.items():
            if idx < len(data['x']):
                x = data['x'][idx]
                y = data['y'][idx]
                speed = data['speed'][idx]
                
                # Store position for ordering
                positions.append((driver, data, x, y, speed))
        
        # Draw drivers (with trail effect)
        for driver, data, x, y, speed in positions:
            # Trail (last 50 points)
            trail_start = max(0, idx - 50)
            trail_x = data['x'][trail_start:idx+1]
            trail_y = data['y'][trail_start:idx+1]
            
            if len(trail_x) > 1:
                # Create gradient trail
                points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                alphas = np.linspace(0.1, 0.6, len(segments))
                colors = [(*plt.matplotlib.colors.to_rgb(data['color']), a) for a in alphas]
                
                lc = LineCollection(segments, colors=colors, linewidth=4, zorder=4)
                ax.add_collection(lc)
            
            # Draw car (circle with team color)
            car = Circle((x, y), 0.25, color=data['color'], zorder=10,
                         edgecolor='white', linewidth=2)
            ax.add_patch(car)
            
            # Driver abbreviation
            ax.text(x, y, driver, ha='center', va='center', fontsize=7,
                   fontweight='bold', color='white', zorder=11)
        
        # Leaderboard
        lb_x = 5.5
        lb_y = 5
        
        ax.add_patch(FancyBboxPatch((lb_x - 0.3, lb_y - len(positions) * 0.55 - 0.3), 
                                    2.5, len(positions) * 0.55 + 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#111111', edgecolor='#333333',
                                    alpha=0.9, zorder=20))
        
        ax.text(lb_x + 0.9, lb_y + 0.2, 'RACE ORDER', ha='center', fontsize=10,
               fontweight='bold', color='#e10600', zorder=21)
        
        # Sort by track position (approximate)
        for i, (driver, data, x, y, speed) in enumerate(positions):
            y_pos = lb_y - 0.5 - i * 0.5
            
            # Position number
            ax.text(lb_x, y_pos, f'P{i+1}', ha='left', fontsize=9,
                   fontweight='bold', color='white', zorder=21)
            
            # Team color bar
            ax.add_patch(plt.Rectangle((lb_x + 0.4, y_pos - 0.15), 0.1, 0.3,
                                       color=data['color'], zorder=21))
            
            # Driver name
            ax.text(lb_x + 0.6, y_pos, data['name'], ha='left', fontsize=9,
                   color='white', zorder=21)
            
            # Speed
            ax.text(lb_x + 2.1, y_pos, f'{speed:.0f}', ha='right', fontsize=8,
                   color='#888888', zorder=21)
        
        # Title and info
        current_lap = (idx // (min_frames // n_laps)) + 1
        progress = (idx % (min_frames // n_laps)) / (min_frames // n_laps) * 100
        
        ax.set_title(f'{track_name.upper()} GRAND PRIX - RACE SIMULATION\n'
                    f'Lap {current_lap}/{n_laps}  |  Progress: {progress:.0f}%',
                    fontsize=16, fontweight='bold', color='#e10600', pad=20)
        
        # Track info box
        ax.text(-5.5, 5, f'🏁 {track_name}\n'
                        f'📍 Lap {current_lap}\n'
                        f'🏎️ {len(positions)} drivers',
               fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#333333'),
               verticalalignment='top')
        
        ax.set_xlim(-7, 8)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                   interval=50, blit=False)
    
    # Save
    output_file = OUTPUT_DIR / f'{track_name.lower().replace(" ", "_")}_race_simulation.gif'
    print(f"  Saving GIF to {output_file}...")
    anim.save(output_file, writer='pillow', fps=20, dpi=100)
    plt.close()
    print(f"  [OK] Saved: {output_file}")


def create_multi_track_comparison():
    """Create side-by-side comparison of all 3 tracks."""
    print("\nCreating multi-track comparison...")
    
    tracks = [
        (2024, 'Monaco', 'Monaco'),
        (2024, 'Belgium', 'Spa'),
        (2024, 'Italy', 'Monza')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for ax, (year, gp_name, display_name) in zip(axes, tracks):
        try:
            session = load_race_data(year, gp_name, 'R')
            if session is None:
                ax.text(0.5, 0.5, f'{display_name}\nData unavailable', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='#888888')
                ax.set_title(display_name, fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
            
            # Get track outline
            track_x, track_y = create_track_outline(session)
            
            # Normalize
            x_center = (track_x.max() + track_x.min()) / 2
            y_center = (track_y.max() + track_y.min()) / 2
            scale = max(track_x.max() - track_x.min(), track_y.max() - track_y.min())
            
            track_x_norm = (track_x - x_center) / scale * 10
            track_y_norm = (track_y - y_center) / scale * 10
            
            # Get fastest lap for color
            fastest = session.laps.pick_fastest()
            tel = fastest.get_telemetry()
            speed = tel['Speed'].values
            
            # Normalize speed for coloring
            speed_norm = (speed - speed.min()) / (speed.max() - speed.min())
            
            # Create colored track
            points = np.array([track_x_norm, track_y_norm]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, cmap='plasma', linewidth=6)
            lc.set_array(speed_norm)
            ax.add_collection(lc)
            
            # Track stats
            driver = fastest['Driver']
            lap_time = fastest['LapTime']
            lap_time_str = f"{lap_time.total_seconds()//60:.0f}:{lap_time.total_seconds()%60:05.2f}"
            
            ax.text(0.5, -0.1, f'Fastest: {driver} - {lap_time_str}',
                   ha='center', transform=ax.transAxes, fontsize=10, color='#00ff00')
            ax.text(0.5, -0.15, f'Top Speed: {speed.max():.0f} km/h',
                   ha='center', transform=ax.transAxes, fontsize=9, color='#888888')
            
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal')
            ax.set_title(f'{display_name.upper()}\n{year} Grand Prix',
                        fontsize=14, fontweight='bold', color='#e10600')
            ax.axis('off')
            
        except Exception as e:
            print(f"  Error with {display_name}: {e}")
            ax.text(0.5, 0.5, f'{display_name}\nError loading data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='#ff4444')
            ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative Speed', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Slow', 'Medium', 'Fast'])
    
    fig.suptitle('F1 TRACK COMPARISON - Monaco • Spa • Monza',
                fontsize=18, fontweight='bold', color='#e10600', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'track_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'track_comparison.png'}")


def main():
    print("=" * 60)
    print("F1 RACE SIMULATION GENERATOR")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Track configurations
    tracks = [
        (2024, 'Monaco', 'Monaco GP'),
        (2024, 'Belgium', 'Spa GP'),
        (2024, 'Italy', 'Monza GP')
    ]
    
    # Generate race simulations for each track
    for year, gp_name, display_name in tracks:
        print(f"\n{'='*40}")
        print(f"Processing: {display_name}")
        print('='*40)
        
        session = load_race_data(year, gp_name, 'R')
        if session:
            animate_race_simulation(session, display_name, n_drivers=6, n_laps=3)
    
    # Create comparison view
    create_multi_track_comparison()
    
    print("\n" + "=" * 60)
    print("RACE SIMULATIONS COMPLETE!")
    print("=" * 60)
    print(f"""
Generated files in {OUTPUT_DIR}:
  • monaco_gp_race_simulation.gif
  • spa_gp_race_simulation.gif
  • monza_gp_race_simulation.gif
  • track_comparison.png
    """)


if __name__ == "__main__":
    main()
