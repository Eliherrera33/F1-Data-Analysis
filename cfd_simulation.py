# -*- coding: utf-8 -*-
"""
F1 CFD-Style Aerodynamic Simulation
Simplified computational fluid dynamics visualization using potential flow theory
and PERRINN F1 CFD coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from pathlib import Path

plt.style.use('dark_background')

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'cfd_visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical constants
AIR_DENSITY = 1.225  # kg/m³
FREESTREAM_VELOCITY = 83.33  # m/s (300 km/h)

# PERRINN F1 CFD Reference Data
PERRINN = {
    'sCx': 1.16,      # Drag coefficient × area (m²)
    'sCz': 3.25,      # Downforce coefficient × area (m²)
    'frontal_area': 1.5,  # m²
    'Cd': 0.77,       # Drag coefficient
    'Cl': 2.17,       # Lift (downforce) coefficient
    'L_D': 2.80,      # Lift-to-drag ratio
}

# F1 Car profile (simplified 2D shape)
def create_f1_profile():
    """Create a simplified F1 car side profile."""
    # X coordinates (normalized 0-5m car length)
    x = np.array([0, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 3.8, 4.2, 4.5, 4.8, 5.0,
                  5.0, 4.8, 4.5, 4.2, 3.8, 3.2, 2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0])
    
    # Y coordinates (height profile)
    y_top = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 0.9, 0.85, 0.7, 0.5, 0.3, 0.15,
                      0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08, 0.1, 0.1, 0.1])
    
    return x, y_top


def create_detailed_f1_shape():
    """Create detailed F1 car components for visualization."""
    components = {}
    
    # Main body
    body_x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
    body_y = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.65, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.3])
    components['body'] = (body_x, body_y)
    
    # Front wing
    fw_x = np.array([0, 0.5, 0.5, 0])
    fw_y = np.array([0.05, 0.05, 0.15, 0.2])
    components['front_wing'] = (fw_x, fw_y)
    
    # Rear wing
    rw_x = np.array([4.3, 4.8, 4.8, 4.3])
    rw_y = np.array([0.8, 0.8, 0.95, 0.95])
    components['rear_wing'] = (rw_x, rw_y)
    
    # Wheels
    components['front_wheel'] = plt.Circle((0.8, 0.33), 0.33, fill=False)
    components['rear_wheel'] = plt.Circle((4.0, 0.33), 0.33, fill=False)
    
    # Floor/diffuser
    floor_x = np.array([0.5, 4.3, 4.5, 0.5])
    floor_y = np.array([0.05, 0.05, 0.15, 0.05])
    components['floor'] = (floor_x, floor_y)
    
    return components


def potential_flow_around_cylinder(X, Y, center, radius, U_inf):
    """
    Calculate potential flow around a cylinder (basic aerodynamic model).
    Used to simulate flow deflection around car body.
    """
    x_rel = X - center[0]
    y_rel = Y - center[1]
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)
    
    # Avoid division by zero
    r = np.where(r < 0.1, 0.1, r)
    
    # Velocity components (potential flow with circulation)
    u = U_inf * (1 - (radius**2 / r**2) * np.cos(2*theta))
    v = -U_inf * (radius**2 / r**2) * np.sin(2*theta)
    
    return u, v


def calculate_velocity_field(X, Y, car_center, car_length=5.0):
    """
    Calculate simplified velocity field around F1 car.
    Combines multiple source/sink pairs to simulate complex flow.
    """
    U_inf = FREESTREAM_VELOCITY
    u = np.ones_like(X) * U_inf
    v = np.zeros_like(Y)
    
    # Add perturbations for front wing
    u1, v1 = potential_flow_around_cylinder(X, Y, (car_center[0]-2, car_center[1]+0.3), 0.4, U_inf*0.3)
    u += u1 - U_inf*0.3
    v += v1
    
    # Add perturbation for body
    u2, v2 = potential_flow_around_cylinder(X, Y, (car_center[0], car_center[1]+0.5), 0.8, U_inf*0.5)
    u += u2 - U_inf*0.5
    v += v2
    
    # Add perturbation for rear wing
    u3, v3 = potential_flow_around_cylinder(X, Y, (car_center[0]+2, car_center[1]+0.8), 0.5, U_inf*0.4)
    u += u3 - U_inf*0.4
    v += v3
    
    # Ground effect - accelerate flow under car
    ground_mask = (Y < car_center[1]) & (X > car_center[0]-2.5) & (X < car_center[0]+2.5)
    u[ground_mask] *= 1.3
    
    return u, v


def calculate_pressure_field(u, v):
    """
    Calculate pressure coefficient using Bernoulli's equation.
    Cp = 1 - (V/V_inf)²
    """
    V_inf = FREESTREAM_VELOCITY
    V = np.sqrt(u**2 + v**2)
    Cp = 1 - (V / V_inf)**2
    return Cp


def plot_streamlines():
    """Create beautiful streamline visualization."""
    print("Generating streamline visualization...")
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create grid
    x = np.linspace(-3, 8, 200)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate velocity field
    car_center = (2.5, 0.5)
    u, v = calculate_velocity_field(X, Y, car_center)
    
    # Mask inside car body
    car_mask = ((X > 0) & (X < 5) & (Y > 0) & (Y < 0.8) & 
                (Y < 0.3 + 0.2*np.sin(np.pi*X/5)))
    u[car_mask] = np.nan
    v[car_mask] = np.nan
    
    # Calculate speed for coloring
    speed = np.sqrt(u**2 + v**2)
    
    # Plot streamlines
    strm = ax.streamplot(X, Y, u, v, density=2.5, color=speed, cmap='plasma',
                         linewidth=1.5, arrowsize=1.2, arrowstyle='->')
    
    # Add colorbar
    cbar = plt.colorbar(strm.lines, ax=ax, label='Velocity (m/s)', shrink=0.8)
    
    # Draw F1 car silhouette
    car_x, car_y = create_f1_profile()
    ax.fill(car_x, car_y, color='#1a1a2e', edgecolor='#e10600', linewidth=2)
    
    # Add ground
    ax.axhline(y=0, color='#333333', linewidth=3)
    ax.fill_between([-3, 8], -1, 0, color='#0a0a0a', alpha=0.8)
    
    # Annotations
    ax.annotate('High velocity\n(Low pressure)', xy=(2.5, -0.2), fontsize=10,
                color='#00ffff', ha='center', fontweight='bold')
    ax.annotate('Flow separation', xy=(5.2, 0.6), fontsize=10,
                color='#ff6b6b', ha='left')
    ax.annotate('Stagnation\npoint', xy=(-0.3, 0.3), fontsize=10,
                color='#ffd93d', ha='right')
    
    ax.set_xlim(-3, 8)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('F1 CAR AERODYNAMIC STREAMLINES\nSimplified CFD Visualization @ 300 km/h', 
                fontsize=16, fontweight='bold', color='#e10600')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'streamlines_visualization.png', dpi=200, 
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'streamlines_visualization.png'}")


def plot_pressure_distribution():
    """Create pressure coefficient heatmap."""
    print("Generating pressure distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create grid
    x = np.linspace(-2, 7, 300)
    y = np.linspace(-0.5, 2.5, 150)
    X, Y = np.meshgrid(x, y)
    
    # Calculate velocity and pressure
    car_center = (2.5, 0.5)
    u, v = calculate_velocity_field(X, Y, car_center)
    Cp = calculate_pressure_field(u, v)
    
    # Mask inside car
    car_mask = ((X > 0) & (X < 5) & (Y > 0) & (Y < 0.9))
    Cp[car_mask] = np.nan
    
    # Left plot: Pressure coefficient
    ax1 = axes[0]
    
    # Custom colormap (blue=low pressure, red=high pressure)
    colors = ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('pressure', colors)
    
    im1 = ax1.contourf(X, Y, Cp, levels=50, cmap=cmap, vmin=-2, vmax=1)
    ax1.contour(X, Y, Cp, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    
    # Draw car
    car_x, car_y = create_f1_profile()
    ax1.fill(car_x, car_y, color='#1a1a2e', edgecolor='white', linewidth=2)
    ax1.axhline(y=0, color='#444444', linewidth=2)
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='Pressure Coefficient (Cp)', shrink=0.8)
    ax1.set_title('PRESSURE COEFFICIENT DISTRIBUTION', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_xlim(-2, 7)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')
    
    # Annotations
    ax1.annotate('Low Pressure\n(Suction)', xy=(2.5, -0.15), fontsize=9,
                color='white', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax1.annotate('High Pressure\n(Stagnation)', xy=(0, 0.5), fontsize=9,
                color='white', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # Right plot: Velocity magnitude
    ax2 = axes[1]
    V = np.sqrt(u**2 + v**2)
    V[car_mask] = np.nan
    
    im2 = ax2.contourf(X, Y, V, levels=50, cmap='hot', vmin=0, vmax=120)
    ax2.contour(X, Y, V, levels=10, colors='white', linewidths=0.5, alpha=0.3)
    
    car_x, car_y = create_f1_profile()
    ax2.fill(car_x, car_y, color='#1a1a2e', edgecolor='white', linewidth=2)
    ax2.axhline(y=0, color='#444444', linewidth=2)
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='Velocity (m/s)', shrink=0.8)
    ax2.set_title('VELOCITY MAGNITUDE FIELD', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_xlim(-2, 7)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect('equal')
    
    ax2.annotate('Accelerated Flow\n(Ground Effect)', xy=(2.5, -0.15), fontsize=9,
                color='white', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ff6600', alpha=0.7))
    
    fig.suptitle('F1 CFD ANALYSIS - PRESSURE & VELOCITY FIELDS @ 300 km/h',
                fontsize=16, fontweight='bold', color='#e10600', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pressure_velocity_fields.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'pressure_velocity_fields.png'}")


def plot_force_vectors():
    """Visualize aerodynamic forces on F1 car."""
    print("Generating force vector diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Draw car
    car_x, car_y = create_f1_profile()
    ax.fill(car_x, car_y, color='#2a2a4a', edgecolor='#e10600', linewidth=2)
    
    # Ground
    ax.axhline(y=0, color='#333333', linewidth=3)
    ax.fill_between([-1, 6], -0.5, 0, color='#0a0a0a')
    
    # Calculate forces at 300 km/h
    speed_kmh = 300
    speed_ms = speed_kmh / 3.6
    q = 0.5 * AIR_DENSITY * speed_ms**2  # Dynamic pressure
    
    drag = q * PERRINN['sCx']
    downforce = q * PERRINN['sCz']
    
    # Scale for visualization
    force_scale = 0.0003
    
    # Draw force arrows
    # Drag (horizontal, opposing motion)
    ax.annotate('', xy=(3.5 - drag*force_scale, 0.5), xytext=(3.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#ff4444', lw=4))
    ax.text(3.5 - drag*force_scale/2, 0.65, f'DRAG\n{drag:.0f} N', 
            ha='center', fontsize=12, fontweight='bold', color='#ff4444')
    
    # Downforce (vertical, pushing down)
    # Front wing contribution (~35%)
    front_df = downforce * 0.35
    ax.annotate('', xy=(0.5, 0.2 - front_df*force_scale*0.5), xytext=(0.5, 0.2),
                arrowprops=dict(arrowstyle='->', color='#00ff00', lw=3))
    ax.text(0.5, -0.25, f'Front\n{front_df:.0f} N', ha='center', fontsize=10, color='#00ff00')
    
    # Rear wing contribution (~35%)
    rear_df = downforce * 0.35
    ax.annotate('', xy=(4.5, 0.9 - rear_df*force_scale*0.5), xytext=(4.5, 0.9),
                arrowprops=dict(arrowstyle='->', color='#00ff00', lw=3))
    ax.text(4.5, 0.5, f'Rear Wing\n{rear_df:.0f} N', ha='center', fontsize=10, color='#00ff00')
    
    # Floor/diffuser contribution (~30%)
    floor_df = downforce * 0.30
    ax.annotate('', xy=(2.5, 0.05 - floor_df*force_scale*0.5), xytext=(2.5, 0.15),
                arrowprops=dict(arrowstyle='->', color='#00ffff', lw=3))
    ax.text(2.5, -0.35, f'Floor/Diffuser\n{floor_df:.0f} N', ha='center', fontsize=10, color='#00ffff')
    
    # Total downforce arrow
    ax.annotate('', xy=(2.5, 0.5 - downforce*force_scale*0.3), xytext=(2.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#ffff00', lw=5))
    
    # Info box
    info_text = f"""
    AERODYNAMIC FORCES @ {speed_kmh} km/h
    ═══════════════════════════════
    Total Downforce: {downforce:,.0f} N ({downforce/9.81:,.0f} kg)
    Total Drag: {drag:,.0f} N
    L/D Ratio: {downforce/drag:.2f}
    
    Downforce Distribution:
    • Front Wing: ~35% ({front_df:,.0f} N)
    • Rear Wing: ~35% ({rear_df:,.0f} N)  
    • Floor/Diffuser: ~30% ({floor_df:,.0f} N)
    
    Note: Downforce > Car Weight!
    (Car could drive upside down)
    """
    
    ax.text(6.5, 0.8, info_text, fontsize=11, fontfamily='monospace',
            color='white', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#e10600', alpha=0.9))
    
    # Airflow arrows (incoming)
    for y_pos in [0.2, 0.5, 0.8, 1.1]:
        ax.annotate('', xy=(-0.3, y_pos), xytext=(-1, y_pos),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))
    ax.text(-1.5, 0.65, 'Airflow\n300 km/h', ha='center', fontsize=10, color='#888888')
    
    ax.set_xlim(-2, 10)
    ax.set_ylim(-0.8, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('F1 AERODYNAMIC FORCE ANALYSIS\nBased on PERRINN CFD Data', 
                fontsize=18, fontweight='bold', color='#e10600', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'force_vectors.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'force_vectors.png'}")


def plot_speed_comparison():
    """Compare aerodynamic forces at different speeds."""
    print("Generating speed comparison chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    speeds = np.linspace(50, 370, 100)
    
    # Calculate forces
    downforce = []
    drag = []
    power = []
    ground_clearance_effect = []
    
    for s in speeds:
        v = s / 3.6
        q = 0.5 * AIR_DENSITY * v**2
        df = q * PERRINN['sCz']
        dr = q * PERRINN['sCx']
        pw = dr * v / 1000  # kW
        
        downforce.append(df / 9.81)  # kg
        drag.append(dr)
        power.append(pw)
        
    downforce = np.array(downforce)
    drag = np.array(drag)
    power = np.array(power)
    
    # Plot 1: Downforce vs Speed
    ax1 = axes[0, 0]
    ax1.fill_between(speeds, 0, downforce, alpha=0.3, color='#00ff00')
    ax1.plot(speeds, downforce, color='#00ff00', linewidth=3, label='Downforce')
    ax1.axhline(y=800, color='#ff00ff', linestyle='--', linewidth=2, label='Car Weight (800 kg)')
    ax1.axvline(x=200, color='#ffff00', linestyle=':', alpha=0.5)
    ax1.text(205, 200, 'DF > Weight\nat ~200 km/h!', fontsize=10, color='#ffff00')
    ax1.set_xlabel('Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Downforce (kg)', fontsize=12)
    ax1.set_title('DOWNFORCE vs SPEED', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(50, 370)
    
    # Plot 2: Drag vs Speed
    ax2 = axes[0, 1]
    ax2.fill_between(speeds, 0, drag, alpha=0.3, color='#ff4444')
    ax2.plot(speeds, drag, color='#ff4444', linewidth=3)
    ax2.set_xlabel('Speed (km/h)', fontsize=12)
    ax2.set_ylabel('Drag Force (N)', fontsize=12)
    ax2.set_title('DRAG vs SPEED', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(50, 370)
    
    # Plot 3: Power Required
    ax3 = axes[1, 0]
    ax3.fill_between(speeds, 0, power, alpha=0.3, color='#ff8800')
    ax3.plot(speeds, power, color='#ff8800', linewidth=3)
    ax3.axhline(y=750, color='#ff0000', linestyle='--', linewidth=2, label='~1000 HP limit')
    ax3.set_xlabel('Speed (km/h)', fontsize=12)
    ax3.set_ylabel('Power to Overcome Drag (kW)', fontsize=12)
    ax3.set_title('POWER REQUIRED vs SPEED', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(50, 370)
    
    # Plot 4: L/D Ratio (constant for simplified model)
    ax4 = axes[1, 1]
    
    # Show different setup comparisons
    setups = {
        'Monaco (High DF)': {'sCx': 1.35, 'sCz': 4.0, 'color': '#ff0000'},
        'Silverstone (Med)': {'sCx': 1.16, 'sCz': 3.25, 'color': '#ffff00'},
        'Monza (Low DF)': {'sCx': 0.95, 'sCz': 2.5, 'color': '#00ff00'},
    }
    
    for name, setup in setups.items():
        df_setup = [0.5 * AIR_DENSITY * (s/3.6)**2 * setup['sCz'] / 9.81 for s in speeds]
        ax4.plot(speeds, df_setup, color=setup['color'], linewidth=2.5, label=name)
    
    ax4.axhline(y=800, color='#ff00ff', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Speed (km/h)', fontsize=12)
    ax4.set_ylabel('Downforce (kg)', fontsize=12)
    ax4.set_title('SETUP COMPARISON - Downforce by Configuration', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(50, 370)
    
    fig.suptitle('F1 AERODYNAMIC PERFORMANCE ANALYSIS\nForce Scaling with Speed (V²)',
                fontsize=18, fontweight='bold', color='#e10600', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'speed_comparison.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'speed_comparison.png'}")


def plot_component_breakdown():
    """Detailed breakdown of aerodynamic components."""
    print("Generating component breakdown...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Component contributions (approximate from F1 data)
    components = {
        'Front Wing': {'df': 35, 'drag': 25, 'color': '#ff6b6b'},
        'Rear Wing': {'df': 35, 'drag': 35, 'color': '#4ecdc4'},
        'Floor/Diffuser': {'df': 30, 'drag': 15, 'color': '#45b7d1'},
        'Body': {'df': 0, 'drag': 20, 'color': '#96ceb4'},
        'Wheels/Suspension': {'df': 0, 'drag': 5, 'color': '#ffeaa7'},
    }
    
    # Left: Downforce pie chart
    ax1 = fig.add_subplot(121)
    df_values = [c['df'] for c in components.values() if c['df'] > 0]
    df_labels = [k for k, v in components.items() if v['df'] > 0]
    df_colors = [components[l]['color'] for l in df_labels]
    
    wedges, texts, autotexts = ax1.pie(df_values, labels=df_labels, autopct='%1.0f%%',
                                        colors=df_colors, explode=[0.02]*len(df_values),
                                        textprops={'fontsize': 12, 'color': 'white'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax1.set_title('DOWNFORCE DISTRIBUTION\nby Component', fontsize=16, fontweight='bold', pad=20)
    
    # Right: Drag breakdown bar chart
    ax2 = fig.add_subplot(122)
    drag_values = [c['drag'] for c in components.values()]
    drag_labels = list(components.keys())
    drag_colors = [c['color'] for c in components.values()]
    
    bars = ax2.barh(drag_labels, drag_values, color=drag_colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, drag_values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Contribution (%)', fontsize=12)
    ax2.set_title('DRAG DISTRIBUTION\nby Component', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlim(0, 45)
    ax2.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('F1 AERODYNAMIC COMPONENT ANALYSIS',
                fontsize=20, fontweight='bold', color='#e10600', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'component_breakdown.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'component_breakdown.png'}")


def plot_ground_effect():
    """Visualize ground effect aerodynamics."""
    print("Generating ground effect visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Ride height effect on downforce
    ax1 = axes[0]
    
    ride_heights = np.linspace(10, 100, 50)  # mm
    
    # Simplified model: ground effect increases suction at lower ride heights
    # But too low causes flow separation (stall)
    base_df = 100  # normalized
    ground_effect_factor = 1 + 0.5 * np.exp(-(ride_heights - 30)**2 / 500)
    stall_penalty = np.where(ride_heights < 20, 0.7 + 0.015 * ride_heights, 1.0)
    
    total_df = base_df * ground_effect_factor * stall_penalty
    
    ax1.plot(ride_heights, total_df, color='#00ff00', linewidth=3)
    ax1.fill_between(ride_heights, base_df, total_df, 
                     where=total_df > base_df, alpha=0.3, color='#00ff00', label='Ground Effect Gain')
    ax1.fill_between(ride_heights, base_df, total_df,
                     where=total_df < base_df, alpha=0.3, color='#ff0000', label='Stall Loss')
    ax1.axhline(y=base_df, color='#888888', linestyle='--', linewidth=2, label='Baseline (no ground effect)')
    ax1.axvline(x=30, color='#ffff00', linestyle=':', alpha=0.7)
    ax1.text(32, 125, 'Optimal\nRide Height', fontsize=10, color='#ffff00')
    
    ax1.set_xlabel('Ride Height (mm)', fontsize=12)
    ax1.set_ylabel('Relative Downforce (%)', fontsize=12)
    ax1.set_title('GROUND EFFECT vs RIDE HEIGHT', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 100)
    
    # Right: Venturi tunnel visualization
    ax2 = axes[1]
    
    # Create Venturi tunnel shape
    x = np.linspace(0, 5, 100)
    floor = np.zeros_like(x)
    car_bottom = 0.3 - 0.15 * np.sin(np.pi * (x - 1) / 3)  # Venturi curve
    car_bottom = np.where((x > 0.5) & (x < 4.5), car_bottom, 0.5)
    
    # Velocity in venturi (continuity: A1*V1 = A2*V2)
    area_ratio = 0.5 / (car_bottom + 0.01)
    velocity = 83.33 * np.minimum(area_ratio, 2.0)  # Cap at 2x
    
    # Color by velocity
    points = np.array([x, (floor + car_bottom) / 2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(80, 160)
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=8)
    lc.set_array(velocity)
    ax2.add_collection(lc)
    
    # Draw car bottom and floor
    ax2.fill_between(x, car_bottom, 0.6, color='#2a2a4a', alpha=0.9)
    ax2.fill_between(x, floor, -0.1, color='#1a1a1a')
    
    # Velocity arrows
    for i in range(0, 100, 10):
        arrow_len = velocity[i] / 200
        ax2.annotate('', xy=(x[i] + arrow_len, (floor[i] + car_bottom[i])/2),
                    xytext=(x[i], (floor[i] + car_bottom[i])/2),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    cbar = plt.colorbar(lc, ax=ax2, label='Velocity (m/s)', shrink=0.8)
    
    ax2.text(2.5, -0.2, 'VENTURI TUNNEL\n(Ground Effect Zone)', ha='center',
            fontsize=12, fontweight='bold', color='#00ffff')
    ax2.text(0.2, 0.1, 'Entry', fontsize=10, color='#888888')
    ax2.text(4.5, 0.1, 'Diffuser\nExit', fontsize=10, color='#888888', ha='right')
    
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.3, 0.8)
    ax2.set_xlabel('Position along car (m)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('VENTURI TUNNEL FLOW ACCELERATION', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    
    fig.suptitle('F1 GROUND EFFECT AERODYNAMICS',
                fontsize=18, fontweight='bold', color='#e10600', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ground_effect.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'ground_effect.png'}")


def plot_cfd_infographic():
    """Create comprehensive CFD summary infographic."""
    print("Generating CFD infographic...")
    
    fig = plt.figure(figsize=(20, 14))
    
    # Title
    fig.suptitle('F1 COMPUTATIONAL FLUID DYNAMICS ANALYSIS\nAerodynamic Performance Summary @ 300 km/h',
                fontsize=22, fontweight='bold', color='#e10600', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. CFD Coefficients Panel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    coeff_text = """
╔═══════════════════════════════╗
║      CFD COEFFICIENTS          ║
╠═══════════════════════════════╣
║  Cd (Drag):      0.77          ║
║  Cl (Lift):      2.17          ║
║  L/D Ratio:      2.80          ║
║                                ║
║  sCx (Drag×A):   1.16 m²       ║
║  sCz (Lift×A):   3.25 m²       ║
║  Frontal Area:   1.50 m²       ║
╚═══════════════════════════════╝
    """
    ax1.text(0.5, 0.5, coeff_text, transform=ax1.transAxes, fontsize=10,
            fontfamily='monospace', ha='center', va='center', color='#00ffff',
            bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#00ffff'))
    ax1.set_title('PERRINN CFD DATA', fontsize=12, fontweight='bold')
    
    # 2. Forces at Speed Panel
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    v = 300 / 3.6
    q = 0.5 * AIR_DENSITY * v**2
    df = q * PERRINN['sCz']
    dr = q * PERRINN['sCx']
    
    forces_text = f"""
╔═══════════════════════════════╗
║     FORCES @ 300 km/h          ║
╠═══════════════════════════════╣
║  Downforce:    {df:>8,.0f} N      ║
║               ({df/9.81:>6,.0f} kg)     ║
║                                ║
║  Drag:         {dr:>8,.0f} N      ║
║                                ║
║  Power Loss:   {dr*v/1000:>6,.0f} kW     ║
║               ({dr*v/1000*1.341:>6,.0f} HP)    ║
╚═══════════════════════════════╝
    """
    ax2.text(0.5, 0.5, forces_text, transform=ax2.transAxes, fontsize=10,
            fontfamily='monospace', ha='center', va='center', color='#00ff00',
            bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#00ff00'))
    ax2.set_title('CALCULATED FORCES', fontsize=12, fontweight='bold')
    
    # 3. Key Metrics Panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    metrics_text = """
╔═══════════════════════════════╗
║       KEY METRICS              ║
╠═══════════════════════════════╣
║  DF exceeds weight at:         ║
║           ~200 km/h ✓          ║
║                                ║
║  Max theoretical top speed:    ║
║           ~380 km/h            ║
║                                ║
║  Ground effect contribution:   ║
║           ~30% of total DF     ║
╚═══════════════════════════════╝
    """
    ax3.text(0.5, 0.5, metrics_text, transform=ax3.transAxes, fontsize=10,
            fontfamily='monospace', ha='center', va='center', color='#ffff00',
            bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#ffff00'))
    ax3.set_title('PERFORMANCE METRICS', fontsize=12, fontweight='bold')
    
    # 4. Setup Comparison Panel
    ax4 = fig.add_subplot(gs[0, 3])
    
    setups = ['Monaco\n(High DF)', 'Silverstone\n(Medium)', 'Monza\n(Low DF)']
    ld_ratios = [4.0/1.35, 3.25/1.16, 2.5/0.95]
    colors = ['#ff0000', '#ffff00', '#00ff00']
    
    bars = ax4.bar(setups, ld_ratios, color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, ld_ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('L/D Ratio', fontsize=11)
    ax4.set_title('SETUP EFFICIENCY', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5-6. Streamline mini-view
    ax5 = fig.add_subplot(gs[1, :2])
    
    x = np.linspace(-1, 6, 120)
    y = np.linspace(-0.3, 2, 60)
    X, Y = np.meshgrid(x, y)
    
    car_center = (2.5, 0.5)
    u, v = calculate_velocity_field(X, Y, car_center)
    
    car_mask = ((X > 0) & (X < 5) & (Y > 0) & (Y < 0.8))
    u[car_mask] = np.nan
    v[car_mask] = np.nan
    
    speed = np.sqrt(u**2 + v**2)
    strm = ax5.streamplot(X, Y, u, v, density=2, color=speed, cmap='plasma',
                         linewidth=1.2, arrowsize=0.8)
    
    car_x, car_y = create_f1_profile()
    ax5.fill(car_x, car_y, color='#1a1a2e', edgecolor='#e10600', linewidth=2)
    ax5.axhline(y=0, color='#333333', linewidth=2)
    
    ax5.set_xlim(-1, 6)
    ax5.set_ylim(-0.3, 2)
    ax5.set_aspect('equal')
    ax5.set_title('STREAMLINE VISUALIZATION', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 7-8. Pressure distribution mini-view
    ax6 = fig.add_subplot(gs[1, 2:])
    
    Cp = calculate_pressure_field(u, v)
    Cp[car_mask] = np.nan
    
    colors = ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('pressure', colors)
    
    im = ax6.contourf(X, Y, Cp, levels=40, cmap=cmap, vmin=-2, vmax=1)
    ax6.fill(car_x, car_y, color='#1a1a2e', edgecolor='white', linewidth=2)
    ax6.axhline(y=0, color='#333333', linewidth=2)
    
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8, label='Cp')
    
    ax6.set_xlim(-1, 6)
    ax6.set_ylim(-0.3, 2)
    ax6.set_aspect('equal')
    ax6.set_title('PRESSURE COEFFICIENT FIELD', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 9. Downforce vs Speed Chart
    ax7 = fig.add_subplot(gs[2, :2])
    
    speeds = np.linspace(50, 370, 100)
    downforce = [0.5 * AIR_DENSITY * (s/3.6)**2 * PERRINN['sCz'] / 9.81 for s in speeds]
    
    ax7.fill_between(speeds, 0, downforce, alpha=0.3, color='#00ff00')
    ax7.plot(speeds, downforce, color='#00ff00', linewidth=3)
    ax7.axhline(y=800, color='#ff00ff', linestyle='--', linewidth=2, label='Car Weight')
    ax7.axvline(x=300, color='#e10600', linestyle=':', linewidth=2, alpha=0.7)
    ax7.text(305, 2500, '300 km/h\nReference', fontsize=9, color='#e10600')
    
    ax7.set_xlabel('Speed (km/h)', fontsize=11)
    ax7.set_ylabel('Downforce (kg)', fontsize=11)
    ax7.set_title('DOWNFORCE SCALING WITH SPEED', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(50, 370)
    
    # 10. Component Distribution
    ax8 = fig.add_subplot(gs[2, 2:])
    
    components = ['Front\nWing', 'Rear\nWing', 'Floor/\nDiffuser', 'Body', 'Other']
    df_dist = [35, 35, 30, 0, 0]
    drag_dist = [25, 35, 15, 20, 5]
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    bars1 = ax8.bar(x_pos - width/2, df_dist, width, label='Downforce %', color='#00ff00', edgecolor='white')
    bars2 = ax8.bar(x_pos + width/2, drag_dist, width, label='Drag %', color='#ff4444', edgecolor='white')
    
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(components)
    ax8.set_ylabel('Contribution (%)', fontsize=11)
    ax8.set_title('COMPONENT CONTRIBUTIONS', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cfd_infographic.png', dpi=200,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'cfd_infographic.png'}")


def main():
    print("=" * 60)
    print("F1 CFD SIMULATION & VISUALIZATION")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate all visualizations
    plot_streamlines()
    plot_pressure_distribution()
    plot_force_vectors()
    plot_speed_comparison()
    plot_component_breakdown()
    plot_ground_effect()
    plot_cfd_infographic()
    
    print("\n" + "=" * 60)
    print("CFD SIMULATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files in: {OUTPUT_DIR}")
    print("""
Files created:
  • streamlines_visualization.png
  • pressure_velocity_fields.png
  • force_vectors.png
  • speed_comparison.png
  • component_breakdown.png
  • ground_effect.png
  • cfd_infographic.png
    """)


if __name__ == "__main__":
    main()
