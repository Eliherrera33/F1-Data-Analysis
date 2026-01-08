# -*- coding: utf-8 -*-
"""
F1 Advanced Race Simulation with Full Telemetry
Enhanced visualization showing G-forces, tire usage, throttle/brake, DRS, gear
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch, Wedge, Rectangle
from matplotlib.collections import LineCollection
import fastf1
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# Setup
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path(__file__).parent / 'race_simulations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Constants
G = 9.81  # m/s^2

# Team colors
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D', 'Mercedes': '#27F4D2',
    'McLaren': '#FF8000', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
    'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
}

# Tire colors
TIRE_COLORS = {'SOFT': '#ff0000', 'MEDIUM': '#ffff00', 'HARD': '#ffffff', 
               'INTERMEDIATE': '#00ff00', 'WET': '#0080ff'}


def extract_telemetry_features(telemetry):
    """Extract advanced features from telemetry data."""
    features = {}
    
    # Basic telemetry
    features['Speed'] = telemetry['Speed'].values
    features['X'] = telemetry['X'].values
    features['Y'] = telemetry['Y'].values
    
    # Throttle and Brake
    features['Throttle'] = telemetry['Throttle'].values if 'Throttle' in telemetry else np.zeros(len(telemetry))
    features['Brake'] = telemetry['Brake'].values if 'Brake' in telemetry else np.zeros(len(telemetry))
    
    # Gear
    features['nGear'] = telemetry['nGear'].values if 'nGear' in telemetry else np.ones(len(telemetry)) * 4
    
    # DRS
    features['DRS'] = telemetry['DRS'].values if 'DRS' in telemetry else np.zeros(len(telemetry))
    
    # RPM
    features['RPM'] = telemetry['RPM'].values if 'RPM' in telemetry else np.ones(len(telemetry)) * 10000
    
    # Calculate G-Forces
    speed_ms = features['Speed'] / 3.6
    dt = 0.1  # Approximate time step
    
    # Longitudinal G (acceleration/braking)
    speed_diff = np.diff(speed_ms, prepend=speed_ms[0])
    long_g = speed_diff / (dt * G)
    long_g = np.clip(long_g, -6, 6)  # Realistic limits
    features['LongG'] = long_g
    
    # Lateral G (cornering)
    dx = np.diff(features['X'], prepend=features['X'][0])
    dy = np.diff(features['Y'], prepend=features['Y'][0])
    
    # Heading angle
    heading = np.arctan2(dy, dx)
    heading_rate = np.diff(heading, prepend=heading[0])
    
    # Unwrap heading changes
    heading_rate = np.where(heading_rate > np.pi, heading_rate - 2*np.pi, heading_rate)
    heading_rate = np.where(heading_rate < -np.pi, heading_rate + 2*np.pi, heading_rate)
    
    # Lateral acceleration = v^2 * curvature = v * heading_rate / dt
    lat_g = (speed_ms * heading_rate / dt) / G
    lat_g = np.clip(lat_g, -6, 6)
    features['LatG'] = lat_g
    
    # Total G
    features['TotalG'] = np.sqrt(long_g**2 + lat_g**2)
    
    return features


def create_telemetry_dashboard(ax_main, ax_speed, ax_gforce, ax_inputs, ax_info,
                               driver_data, current_idx, driver_abbr, 
                               track_x, track_y, tire_compound='MEDIUM'):
    """Create comprehensive telemetry dashboard for a driver."""
    
    data = driver_data[driver_abbr]
    idx = min(current_idx, len(data['Speed']) - 1)
    
    # Get current values
    speed = data['Speed'][idx]
    long_g = data['LongG'][idx]
    lat_g = data['LatG'][idx]
    total_g = data['TotalG'][idx]
    throttle = data['Throttle'][idx]
    brake = data['Brake'][idx]
    gear = int(data['nGear'][idx])
    drs = data['DRS'][idx] > 0
    rpm = data['RPM'][idx]
    
    # ===== MAIN TRACK VIEW =====
    ax_main.clear()
    
    # Draw track
    ax_main.plot(track_x, track_y, color='#333333', linewidth=12, solid_capstyle='round')
    ax_main.plot(track_x, track_y, color='#1a1a1a', linewidth=9, solid_capstyle='round')
    
    # Draw all cars with trails
    for drv, d in driver_data.items():
        drv_idx = min(current_idx, len(d['X']) - 1)
        
        # Trail
        trail_start = max(0, drv_idx - 30)
        trail_x = d['X'][trail_start:drv_idx+1]
        trail_y = d['Y'][trail_start:drv_idx+1]
        
        if len(trail_x) > 1:
            points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.1, 0.7, len(segments))
            colors = [(*plt.matplotlib.colors.to_rgb(d['color']), a) for a in alphas]
            lc = LineCollection(segments, colors=colors, linewidth=3)
            ax_main.add_collection(lc)
        
        # Car dot
        x, y = d['X'][drv_idx], d['Y'][drv_idx]
        size = 0.4 if drv == driver_abbr else 0.25
        car = Circle((x, y), size, color=d['color'], zorder=10, 
                     edgecolor='white' if drv == driver_abbr else d['color'], 
                     linewidth=2 if drv == driver_abbr else 0)
        ax_main.add_patch(car)
        
        # Label for main driver
        if drv == driver_abbr:
            ax_main.text(x, y, drv, ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white', zorder=11)
    
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_title(f"{data['name']} - {data['team']}", fontsize=14, 
                     fontweight='bold', color=data['color'])
    
    # ===== SPEED GAUGE =====
    ax_speed.clear()
    ax_speed.set_xlim(-1.5, 1.5)
    ax_speed.set_ylim(-1.5, 1.5)
    
    # Gauge background
    theta = np.linspace(np.pi*0.8, np.pi*0.2, 100)
    for r in [0.9, 1.0, 1.1]:
        ax_speed.plot(r*np.cos(theta), r*np.sin(theta), color='#333333', linewidth=2)
    
    # Speed arc (colored by speed)
    speed_angle = np.pi*0.8 - (speed/400) * np.pi*0.6
    theta_speed = np.linspace(np.pi*0.8, max(speed_angle, np.pi*0.2), 50)
    
    # Color gradient
    for i, t in enumerate(theta_speed[:-1]):
        color_val = i / len(theta_speed)
        color = plt.cm.plasma(color_val)
        ax_speed.plot([1.0*np.cos(t), 1.0*np.cos(theta_speed[i+1])],
                     [1.0*np.sin(t), 1.0*np.sin(theta_speed[i+1])],
                     color=color, linewidth=8)
    
    # Needle
    ax_speed.plot([0, 0.8*np.cos(speed_angle)], [0, 0.8*np.sin(speed_angle)],
                 color='#e10600', linewidth=3)
    ax_speed.scatter([0], [0], color='#e10600', s=100, zorder=10)
    
    # Speed text
    ax_speed.text(0, -0.3, f'{speed:.0f}', ha='center', fontsize=24, 
                 fontweight='bold', color='white')
    ax_speed.text(0, -0.6, 'KM/H', ha='center', fontsize=10, color='#888888')
    
    # Gear indicator
    ax_speed.text(0, 0.4, f'{gear}', ha='center', fontsize=28, 
                 fontweight='bold', color='#00ff00')
    ax_speed.text(0, 0.7, 'GEAR', ha='center', fontsize=8, color='#888888')
    
    # DRS indicator
    drs_color = '#00ff00' if drs else '#333333'
    ax_speed.add_patch(Rectangle((-0.3, -1.1), 0.6, 0.25, color=drs_color, alpha=0.8))
    ax_speed.text(0, -0.98, 'DRS', ha='center', fontsize=10, fontweight='bold',
                 color='black' if drs else '#666666')
    
    ax_speed.axis('off')
    ax_speed.set_title('SPEED', fontsize=10, color='#888888')
    
    # ===== G-FORCE DISPLAY =====
    ax_gforce.clear()
    ax_gforce.set_xlim(-7, 7)
    ax_gforce.set_ylim(-7, 7)
    
    # G-force circles
    for r in [2, 4, 6]:
        circle = Circle((0, 0), r, fill=False, color='#333333', linewidth=1)
        ax_gforce.add_patch(circle)
        ax_gforce.text(r+0.2, 0.2, f'{r}G', fontsize=8, color='#555555')
    
    # Crosshairs
    ax_gforce.axhline(0, color='#444444', linewidth=1)
    ax_gforce.axvline(0, color='#444444', linewidth=1)
    
    # G-force dot with trail
    trail_len = 20
    start_idx = max(0, idx - trail_len)
    lat_trail = data['LatG'][start_idx:idx+1]
    long_trail = data['LongG'][start_idx:idx+1]
    
    if len(lat_trail) > 1:
        points = np.array([lat_trail, long_trail]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.1, 0.8, len(segments))
        colors = [(1, 0.5, 0, a) for a in alphas]
        lc = LineCollection(segments, colors=colors, linewidth=2)
        ax_gforce.add_collection(lc)
    
    # Current G-force point
    g_color = plt.cm.hot(min(total_g/6, 1))
    ax_gforce.scatter([lat_g], [long_g], s=200, c=[g_color], 
                     edgecolors='white', linewidth=2, zorder=10)
    
    # Labels
    ax_gforce.text(0, 6.5, 'ACCEL', ha='center', fontsize=8, color='#00ff00')
    ax_gforce.text(0, -6.5, 'BRAKE', ha='center', fontsize=8, color='#ff0000')
    ax_gforce.text(6.5, 0, 'RIGHT', ha='center', fontsize=8, color='#ffff00', rotation=90)
    ax_gforce.text(-6.5, 0, 'LEFT', ha='center', fontsize=8, color='#ffff00', rotation=90)
    
    # G values text
    ax_gforce.text(5, 5, f'LAT: {abs(lat_g):.1f}G', fontsize=9, color='#ffff00')
    ax_gforce.text(5, 4, f'LONG: {abs(long_g):.1f}G', fontsize=9, 
                  color='#00ff00' if long_g > 0 else '#ff0000')
    ax_gforce.text(5, 3, f'TOTAL: {total_g:.1f}G', fontsize=9, color='#ff8800', fontweight='bold')
    
    ax_gforce.set_aspect('equal')
    ax_gforce.axis('off')
    ax_gforce.set_title('G-FORCE', fontsize=10, color='#888888')
    
    # ===== THROTTLE/BRAKE BARS =====
    ax_inputs.clear()
    ax_inputs.set_xlim(0, 100)
    ax_inputs.set_ylim(0, 3)
    
    # Throttle bar
    ax_inputs.add_patch(Rectangle((0, 1.8), 100, 0.8, color='#222222'))
    ax_inputs.add_patch(Rectangle((0, 1.8), throttle, 0.8, color='#00ff00'))
    ax_inputs.text(50, 2.2, f'THROTTLE {throttle:.0f}%', ha='center', fontsize=9, 
                  color='white', fontweight='bold')
    
    # Brake bar
    ax_inputs.add_patch(Rectangle((0, 0.6), 100, 0.8, color='#222222'))
    brake_val = brake if isinstance(brake, (int, float)) else (100 if brake else 0)
    ax_inputs.add_patch(Rectangle((0, 0.6), brake_val, 0.8, color='#ff0000'))
    ax_inputs.text(50, 1.0, f'BRAKE {brake_val:.0f}%', ha='center', fontsize=9,
                  color='white', fontweight='bold')
    
    ax_inputs.axis('off')
    
    # ===== INFO PANEL =====
    ax_info.clear()
    ax_info.set_xlim(0, 10)
    ax_info.set_ylim(0, 10)
    
    # Tire indicator
    tire_color = TIRE_COLORS.get(tire_compound, '#888888')
    tire_circle = Circle((1, 8), 0.8, color=tire_color, edgecolor='white', linewidth=2)
    ax_info.add_patch(tire_circle)
    ax_info.text(1, 8, tire_compound[0], ha='center', va='center', fontsize=12,
                fontweight='bold', color='black')
    ax_info.text(2.5, 8, tire_compound, fontsize=10, color=tire_color, va='center')
    
    # RPM bar
    ax_info.add_patch(Rectangle((0.5, 5.5), 9, 0.5, color='#222222'))
    rpm_pct = min(rpm / 15000, 1) * 9
    rpm_color = '#00ff00' if rpm < 12000 else '#ffff00' if rpm < 14000 else '#ff0000'
    ax_info.add_patch(Rectangle((0.5, 5.5), rpm_pct, 0.5, color=rpm_color))
    ax_info.text(5, 6.5, f'RPM: {rpm:.0f}', ha='center', fontsize=9, color='white')
    
    # Delta time (simulated)
    delta = np.random.uniform(-0.5, 0.5)
    delta_color = '#00ff00' if delta < 0 else '#ff0000'
    delta_text = f'{delta:+.3f}' if delta != 0 else '+0.000'
    ax_info.text(5, 3.5, delta_text, ha='center', fontsize=16, 
                fontweight='bold', color=delta_color)
    ax_info.text(5, 2.5, 'INTERVAL', ha='center', fontsize=8, color='#888888')
    
    ax_info.axis('off')


def create_advanced_simulation(year, gp_name, display_name, n_drivers=4, n_laps=2):
    """Create advanced race simulation with full telemetry."""
    print(f"\n[INFO] Creating advanced simulation: {display_name}")
    
    # Load session
    print(f"  Loading {year} {gp_name} data...")
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
    except Exception as e:
        print(f"  Error loading: {e}")
        return
    
    # Get track
    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()
    
    x_raw, y_raw = tel['X'].values, tel['Y'].values
    x_center = (x_raw.max() + x_raw.min()) / 2
    y_center = (y_raw.max() + y_raw.min()) / 2
    scale = max(x_raw.max() - x_raw.min(), y_raw.max() - y_raw.min())
    
    track_x = (x_raw - x_center) / scale * 10
    track_y = (y_raw - y_center) / scale * 10
    
    # Get top drivers
    laps = session.laps.pick_quicklaps()
    fastest_per_driver = laps.groupby('Driver')['LapTime'].min().sort_values()
    top_drivers = fastest_per_driver.head(n_drivers).index.tolist()
    print(f"  Drivers: {', '.join(top_drivers)}")
    
    # Extract telemetry for each driver
    driver_data = {}
    
    for driver in top_drivers:
        try:
            driver_laps = session.laps.pick_driver(driver)
            completed = driver_laps[driver_laps['LapTime'].notna()]
            
            if len(completed) == 0:
                continue
            
            all_features = []
            lap_count = 0
            
            for _, lap in completed.iterrows():
                if lap_count >= n_laps:
                    break
                try:
                    tel = lap.get_telemetry()
                    if len(tel) > 0:
                        features = extract_telemetry_features(tel)
                        # Normalize positions
                        features['X'] = (features['X'] - x_center) / scale * 10
                        features['Y'] = (features['Y'] - y_center) / scale * 10
                        all_features.append(features)
                        lap_count += 1
                except:
                    continue
            
            if all_features:
                # Concatenate features
                combined = {}
                for key in all_features[0].keys():
                    combined[key] = np.concatenate([f[key] for f in all_features])
                
                driver_info = session.get_driver(driver)
                combined['color'] = TEAM_COLORS.get(driver_info.get('TeamName', ''), '#888888')
                combined['name'] = driver_info.get('FullName', driver)
                combined['team'] = driver_info.get('TeamName', 'Unknown')
                
                driver_data[driver] = combined
                
        except Exception as e:
            print(f"    Error with {driver}: {e}")
            continue
    
    if len(driver_data) < 2:
        print("  Not enough driver data")
        return
    
    # Get tire compound from first lap
    tire_compound = 'MEDIUM'
    try:
        first_lap = session.laps.pick_driver(top_drivers[0]).iloc[0]
        tire_compound = first_lap.get('Compound', 'MEDIUM')
    except:
        pass
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Layout
    ax_main = fig.add_axes([0.02, 0.3, 0.5, 0.65])      # Track view
    ax_speed = fig.add_axes([0.55, 0.5, 0.2, 0.45])     # Speed gauge
    ax_gforce = fig.add_axes([0.77, 0.5, 0.22, 0.45])   # G-force
    ax_inputs = fig.add_axes([0.55, 0.32, 0.44, 0.15])  # Throttle/brake
    ax_info = fig.add_axes([0.55, 0.05, 0.44, 0.25])    # Info panel
    ax_title = fig.add_axes([0.02, 0.02, 0.5, 0.25])    # Leaderboard
    
    # Get main driver (leader)
    main_driver = top_drivers[0]
    
    # Determine frame count
    min_len = min(len(d['X']) for d in driver_data.values())
    n_frames = min(min_len // 15, 200)  # Subsample
    frame_step = min_len // n_frames
    
    print(f"  Generating {n_frames} frames with full telemetry...")
    
    def animate(frame):
        idx = frame * frame_step
        
        # Main dashboard
        create_telemetry_dashboard(ax_main, ax_speed, ax_gforce, ax_inputs, ax_info,
                                  driver_data, idx, main_driver, track_x, track_y,
                                  tire_compound)
        
        # Leaderboard
        ax_title.clear()
        ax_title.set_xlim(0, 10)
        ax_title.set_ylim(0, 5)
        
        ax_title.text(5, 4.5, f'{display_name.upper()}', ha='center', fontsize=18,
                     fontweight='bold', color='#e10600')
        
        lap_num = (idx // (min_len // n_laps)) + 1
        ax_title.text(5, 3.8, f'LAP {lap_num}/{n_laps}', ha='center', fontsize=12,
                     color='#888888')
        
        # Position list
        for i, (drv, data) in enumerate(driver_data.items()):
            y_pos = 2.8 - i * 0.7
            ax_title.add_patch(Rectangle((1, y_pos-0.2), 0.3, 0.5, color=data['color']))
            ax_title.text(1.5, y_pos, f'P{i+1}', fontsize=10, color='white', va='center')
            ax_title.text(2, y_pos, drv, fontsize=11, fontweight='bold', 
                         color='white', va='center')
            drv_idx = min(idx, len(data['Speed'])-1)
            ax_title.text(3.5, y_pos, f"{data['Speed'][drv_idx]:.0f} km/h", 
                         fontsize=9, color='#888888', va='center')
        
        ax_title.axis('off')
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=60, blit=False)
    
    output_file = OUTPUT_DIR / f'{display_name.lower().replace(" ", "_")}_advanced.gif'
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=8, dpi=80)
    plt.close()
    print(f"  [OK] Saved: {output_file}")


def main():
    print("=" * 60)
    print("F1 ADVANCED RACE SIMULATION")
    print("Full Telemetry: Speed, G-Force, Throttle/Brake, Tire, DRS")
    print("=" * 60)
    
    tracks = [
        (2024, 'Monaco', 'Monaco GP'),
        (2024, 'Belgium', 'Spa GP'),
        (2024, 'Italy', 'Monza GP'),
    ]
    
    for year, gp, name in tracks:
        create_advanced_simulation(year, gp, name, n_drivers=4, n_laps=2)
    
    print("\n" + "=" * 60)
    print("ADVANCED SIMULATIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
