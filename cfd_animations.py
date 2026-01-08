# -*- coding: utf-8 -*-
"""
F1 CFD 3D Animated Visualizations
Creates rotating 3D views and animated flow simulations as GIFs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use('dark_background')

OUTPUT_DIR = Path(__file__).parent / 'cfd_visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical constants
AIR_DENSITY = 1.225
FREESTREAM_VELOCITY = 83.33  # 300 km/h

# PERRINN CFD Data
PERRINN = {'sCx': 1.16, 'sCz': 3.25, 'Cd': 0.77, 'Cl': 2.17}


def create_f1_car_3d():
    """Create 3D F1 car mesh for visualization."""
    # Car dimensions (meters)
    length = 5.0
    width = 2.0
    height = 0.95
    
    vertices = []
    faces = []
    
    # Main body (simplified box with aero shaping)
    # Bottom points
    body_bottom = np.array([
        [0.5, -0.8, 0.05],   # Front left
        [0.5, 0.8, 0.05],    # Front right
        [4.5, 0.8, 0.05],    # Rear right
        [4.5, -0.8, 0.05],   # Rear left
    ])
    
    # Top points (curved)
    body_top = np.array([
        [0.8, -0.6, 0.4],    # Front left
        [0.8, 0.6, 0.4],     # Front right
        [4.2, 0.6, 0.6],     # Rear right
        [4.2, -0.6, 0.6],    # Rear left
    ])
    
    # Nose cone
    nose = np.array([
        [0, 0, 0.2],         # Tip
        [0.5, -0.3, 0.1],    # Left
        [0.5, 0.3, 0.1],     # Right
        [0.5, 0, 0.3],       # Top
    ])
    
    return {
        'body_bottom': body_bottom,
        'body_top': body_top,
        'nose': nose,
        'length': length,
        'width': width,
        'height': height
    }


def draw_f1_car_3d(ax, car_data, color='#e10600', alpha=0.8):
    """Draw simplified 3D F1 car outline."""
    # Main body sides
    bb = car_data['body_bottom']
    bt = car_data['body_top']
    
    # Side panels
    left_side = [bb[0], bb[3], bt[3], bt[0]]
    right_side = [bb[1], bb[2], bt[2], bt[1]]
    front = [bb[0], bb[1], bt[1], bt[0]]
    back = [bb[2], bb[3], bt[3], bt[2]]
    top = [bt[0], bt[1], bt[2], bt[3]]
    bottom = [bb[0], bb[1], bb[2], bb[3]]
    
    # Create mesh
    verts = [left_side, right_side, front, back, top, bottom]
    
    poly = Poly3DCollection(verts, alpha=alpha, facecolor='#1a1a2e', 
                            edgecolor=color, linewidth=2)
    ax.add_collection3d(poly)
    
    # Front wing
    fw_verts = [
        [[0, -1, 0.05], [0.3, -1, 0.05], [0.3, -1, 0.15], [0, -1, 0.2]],
        [[0, 1, 0.05], [0.3, 1, 0.05], [0.3, 1, 0.15], [0, 1, 0.2]],
        [[0, -1, 0.05], [0, 1, 0.05], [0, 1, 0.2], [0, -1, 0.2]],
    ]
    fw_poly = Poly3DCollection(fw_verts, alpha=0.7, facecolor='#2a2a4a', 
                               edgecolor='#00ffff', linewidth=1.5)
    ax.add_collection3d(fw_poly)
    
    # Rear wing
    rw_verts = [
        [[4.3, -0.8, 0.7], [4.8, -0.8, 0.7], [4.8, -0.8, 0.95], [4.3, -0.8, 0.95]],
        [[4.3, 0.8, 0.7], [4.8, 0.8, 0.7], [4.8, 0.8, 0.95], [4.3, 0.8, 0.95]],
        [[4.3, -0.8, 0.95], [4.3, 0.8, 0.95], [4.8, 0.8, 0.95], [4.8, -0.8, 0.95]],
    ]
    rw_poly = Poly3DCollection(rw_verts, alpha=0.7, facecolor='#2a2a4a',
                               edgecolor='#00ffff', linewidth=1.5)
    ax.add_collection3d(rw_poly)
    
    # Wheels (simplified as cylinders represented by circles)
    theta = np.linspace(0, 2*np.pi, 20)
    r = 0.33
    
    # Front wheels
    for y_pos in [-0.9, 0.9]:
        x_wheel = 0.8 + r * np.cos(theta)
        z_wheel = r + r * np.sin(theta)
        y_wheel = np.full_like(theta, y_pos)
        ax.plot(x_wheel, y_wheel, z_wheel, color='#888888', linewidth=2)
    
    # Rear wheels
    for y_pos in [-0.9, 0.9]:
        x_wheel = 4.0 + r * np.cos(theta)
        z_wheel = r + r * np.sin(theta)
        y_wheel = np.full_like(theta, y_pos)
        ax.plot(x_wheel, y_wheel, z_wheel, color='#888888', linewidth=2)


def create_streamlines_3d(n_lines=30):
    """Generate 3D streamline data around car."""
    streamlines = []
    
    # Create streamlines at different y positions
    y_positions = np.linspace(-1.5, 1.5, n_lines)
    
    for y_start in y_positions:
        x = np.linspace(-2, 8, 100)
        y = np.full_like(x, y_start)
        
        # Vertical displacement based on car body
        z_base = 0.5 + 0.3 * np.abs(y_start)  # Higher for outer streamlines
        
        # Deflection around car (simplified potential flow)
        z_deflection = np.zeros_like(x)
        
        # Front rise
        mask_front = (x > -0.5) & (x < 1.5) & (np.abs(y_start) < 1.2)
        z_deflection[mask_front] = 0.4 * np.exp(-((x[mask_front] - 0.5)**2) / 1)
        
        # Body rise
        mask_body = (x > 1) & (x < 4) & (np.abs(y_start) < 1)
        z_deflection[mask_body] = np.maximum(z_deflection[mask_body], 
                                              0.3 + 0.2 * np.sin(np.pi * (x[mask_body] - 1) / 3))
        
        # Rear wing rise
        mask_rear = (x > 3.5) & (x < 5.5) & (np.abs(y_start) < 1)
        z_deflection[mask_rear] = np.maximum(z_deflection[mask_rear],
                                              0.5 * np.exp(-((x[mask_rear] - 4.5)**2) / 0.5))
        
        z = z_base + z_deflection
        
        # Ground effect - streamlines compressed under car
        if np.abs(y_start) < 0.8:
            mask_under = (x > 0.5) & (x < 4.5) & (z_base < 0.3)
            z_under = 0.1 + 0.05 * np.sin(np.pi * (x - 0.5) / 4)
            # Add under-car streamlines
            streamlines.append((x, y, z_under * np.ones_like(x)))
        
        streamlines.append((x, y, z))
    
    return streamlines


def animate_3d_rotation():
    """Create rotating 3D view of car with streamlines."""
    print("Creating 3D rotating animation...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    car_data = create_f1_car_3d()
    streamlines = create_streamlines_3d(n_lines=20)
    
    def init():
        ax.clear()
        return []
    
    def animate(frame):
        ax.clear()
        
        # Set viewing angle
        ax.view_init(elev=25, azim=frame * 2)
        
        # Draw car
        draw_f1_car_3d(ax, car_data)
        
        # Draw streamlines with color gradient
        for i, (x, y, z) in enumerate(streamlines):
            # Color by position
            color_val = i / len(streamlines)
            color = plt.cm.plasma(color_val)
            ax.plot(x, y, z, color=color, alpha=0.6, linewidth=1)
            
            # Add arrow at end
            if len(x) > 2:
                ax.quiver(x[-2], y[-2], z[-2], 
                         x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2],
                         color=color, arrow_length_ratio=0.3, alpha=0.8)
        
        # Draw ground plane
        xx, yy = np.meshgrid(np.linspace(-2, 8, 10), np.linspace(-2, 2, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='#333333')
        
        # Labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'F1 CAR 3D AERODYNAMIC FLOW\nView Angle: {frame * 2}°', 
                    fontsize=14, fontweight='bold', color='#e10600')
        
        # Set limits
        ax.set_xlim(-2, 8)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 2)
        
        # Equal aspect ratio
        ax.set_box_aspect([10, 4, 2])
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=50, blit=False)
    
    # Save as GIF
    print("  Saving GIF (this may take a minute)...")
    anim.save(OUTPUT_DIR / '3d_rotation.gif', writer='pillow', fps=20, dpi=100)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '3d_rotation.gif'}")


def animate_streamline_flow():
    """Animate particles flowing along streamlines."""
    print("Creating streamline flow animation...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create velocity field
    x = np.linspace(-2, 8, 150)
    y = np.linspace(-1, 2, 75)
    X, Y = np.meshgrid(x, y)
    
    # Simplified velocity field
    U = FREESTREAM_VELOCITY * np.ones_like(X)
    V = np.zeros_like(Y)
    
    # Add perturbations for car (centered at x=2.5)
    car_center_x = 2.5
    
    # Front wing effect
    mask_fw = (X > 0) & (X < 1) & (Y > 0) & (Y < 0.5)
    V[mask_fw] += 15 * np.exp(-((X[mask_fw] - 0.5)**2 + (Y[mask_fw] - 0.2)**2) / 0.3)
    
    # Body deflection
    mask_body = (X > 0.5) & (X < 4.5) & (Y > 0.05) & (Y < 1.2)
    V[mask_body] += 10 * np.exp(-((Y[mask_body] - 0.6)**2) / 0.3)
    
    # Ground effect (acceleration under car)
    mask_ground = (X > 0.5) & (X < 4.5) & (Y < 0.2) & (Y > -0.1)
    U[mask_ground] *= 1.4
    
    # Mask inside car
    car_mask = ((X > 0) & (X < 5) & (Y > 0.05) & (Y < 0.8))
    
    # Create particles
    n_particles = 100
    particles_x = np.random.uniform(-2, -1, n_particles)
    particles_y = np.random.uniform(-0.5, 1.8, n_particles)
    
    # Create car profile
    car_x = np.array([0, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 3.8, 4.2, 4.5, 4.8, 5.0,
                      5.0, 4.8, 4.5, 4.2, 3.8, 3.2, 2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0])
    car_y = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 0.9, 0.85, 0.7, 0.5, 0.3, 0.15,
                      0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08, 0.1, 0.1, 0.1])
    
    # Particle trails
    trail_length = 10
    trails_x = np.zeros((n_particles, trail_length))
    trails_y = np.zeros((n_particles, trail_length))
    
    def init():
        return []
    
    def animate(frame):
        nonlocal particles_x, particles_y, trails_x, trails_y
        
        ax.clear()
        
        # Background streamlines (static)
        speed = np.sqrt(U**2 + V**2)
        speed_masked = np.ma.array(speed, mask=car_mask)
        
        # Draw streamlines
        strm = ax.streamplot(X, Y, U, V, density=1.5, color=speed_masked, 
                            cmap='plasma', linewidth=0.8, arrowsize=0.8)
        
        # Draw car
        ax.fill(car_x, car_y, color='#1a1a2e', edgecolor='#e10600', linewidth=3)
        
        # Add car details
        # Front wing
        ax.fill([0, 0.4, 0.4, 0], [0.05, 0.05, 0.18, 0.18], 
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        # Rear wing
        ax.fill([4.3, 4.8, 4.8, 4.3], [0.75, 0.75, 0.95, 0.95],
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        # Wheels
        wheel1 = plt.Circle((0.8, 0.33), 0.28, fill=True, color='#333333', 
                            edgecolor='#666666', linewidth=2)
        wheel2 = plt.Circle((4.0, 0.33), 0.28, fill=True, color='#333333',
                            edgecolor='#666666', linewidth=2)
        ax.add_patch(wheel1)
        ax.add_patch(wheel2)
        
        # Update particle positions
        dt = 0.05
        
        # Shift trails
        trails_x[:, 1:] = trails_x[:, :-1]
        trails_y[:, 1:] = trails_y[:, :-1]
        trails_x[:, 0] = particles_x
        trails_y[:, 0] = particles_y
        
        for i in range(n_particles):
            # Get velocity at particle position
            xi = int((particles_x[i] + 2) / 10 * 149)
            yi = int((particles_y[i] + 1) / 3 * 74)
            
            xi = np.clip(xi, 0, 149)
            yi = np.clip(yi, 0, 74)
            
            u_p = U[yi, xi]
            v_p = V[yi, xi]
            
            # Check if inside car
            if (0 < particles_x[i] < 5) and (0.05 < particles_y[i] < 0.8):
                # Reset particle
                particles_x[i] = np.random.uniform(-2, -1)
                particles_y[i] = np.random.uniform(-0.5, 1.8)
                trails_x[i, :] = particles_x[i]
                trails_y[i, :] = particles_y[i]
            else:
                # Move particle
                particles_x[i] += u_p * dt
                particles_y[i] += v_p * dt
            
            # Reset if out of bounds
            if particles_x[i] > 8:
                particles_x[i] = np.random.uniform(-2, -1)
                particles_y[i] = np.random.uniform(-0.5, 1.8)
                trails_x[i, :] = particles_x[i]
                trails_y[i, :] = particles_y[i]
        
        # Draw particles with trails
        for i in range(n_particles):
            # Trail color based on speed
            speed_i = np.sqrt(U[int((trails_y[i, 0] + 1) / 3 * 74) % 75, 
                                int((trails_x[i, 0] + 2) / 10 * 149) % 150]**2)
            color = plt.cm.hot(np.clip(speed_i / 120, 0, 1))
            
            # Draw trail
            valid = (trails_x[i] > -2) & (trails_x[i] < 8)
            if np.sum(valid) > 1:
                ax.plot(trails_x[i][valid], trails_y[i][valid], 
                       color=color, alpha=0.6, linewidth=1.5)
            
            # Draw particle
            ax.scatter(particles_x[i], particles_y[i], 
                      c=[color], s=30, zorder=5, edgecolors='white', linewidth=0.5)
        
        # Ground
        ax.axhline(y=0, color='#444444', linewidth=3)
        ax.fill_between([-2, 8], -1, 0, color='#0a0a0a')
        
        # Labels
        ax.set_xlim(-2, 8)
        ax.set_ylim(-0.5, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title(f'F1 REAL-TIME AIRFLOW SIMULATION @ 300 km/h\nFrame: {frame}',
                    fontsize=14, fontweight='bold', color='#e10600')
        
        # Speed legend
        ax.text(6.5, 1.8, f'Freestream:\n{FREESTREAM_VELOCITY:.0f} m/s\n(300 km/h)', 
               fontsize=10, color='#888888', 
               bbox=dict(boxstyle='round', facecolor='#111111', alpha=0.8))
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=50, blit=False)
    
    print("  Saving GIF...")
    anim.save(OUTPUT_DIR / 'streamline_flow.gif', writer='pillow', fps=20, dpi=100)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'streamline_flow.gif'}")


def animate_pressure_field():
    """Animate pressure field with car outline."""
    print("Creating pressure field animation...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Grid
    x = np.linspace(-2, 8, 200)
    y = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Car profile
    car_x = np.array([0, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 3.8, 4.2, 4.5, 4.8, 5.0,
                      5.0, 4.8, 4.5, 4.2, 3.8, 3.2, 2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0])
    car_y = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 0.9, 0.85, 0.7, 0.5, 0.3, 0.15,
                      0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08, 0.1, 0.1, 0.1])
    
    def calculate_pressure(speed_factor):
        """Calculate pressure field for given speed."""
        V_inf = FREESTREAM_VELOCITY * speed_factor
        
        # Velocity field (simplified)
        U = V_inf * np.ones_like(X)
        V = np.zeros_like(Y)
        
        # Perturbations
        for cx, cy, r, strength in [(0.5, 0.3, 0.5, 0.3), (2.5, 0.5, 0.8, 0.5), (4.5, 0.8, 0.4, 0.3)]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            factor = np.exp(-(dist**2) / (r**2))
            U -= V_inf * strength * factor
            V += V_inf * strength * 0.5 * (Y - cy) / (dist + 0.1) * factor
        
        # Ground effect
        ground_mask = (Y < 0.15) & (X > 0.5) & (X < 4.5)
        U[ground_mask] *= 1.4
        
        # Calculate pressure
        V_mag = np.sqrt(U**2 + V**2)
        Cp = 1 - (V_mag / V_inf)**2
        
        # Mask car
        car_mask = ((X > 0) & (X < 5) & (Y > 0.05) & (Y < 0.85))
        Cp[car_mask] = np.nan
        
        return Cp, V_inf
    
    def animate(frame):
        ax.clear()
        
        # Oscillating speed (200-350 km/h)
        speed_factor = 0.7 + 0.5 * np.sin(frame * 0.1)
        speed_kmh = speed_factor * 300
        
        Cp, V_inf = calculate_pressure(speed_factor)
        
        # Plot pressure
        levels = np.linspace(-2, 1, 50)
        cf = ax.contourf(X, Y, Cp, levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(X, Y, Cp, levels=10, colors='white', linewidths=0.5, alpha=0.3)
        
        # Draw car with bold outline
        ax.fill(car_x, car_y, color='#1a1a2e', edgecolor='#e10600', linewidth=4)
        
        # Add car components
        ax.fill([0, 0.4, 0.4, 0], [0.05, 0.05, 0.18, 0.18], 
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        ax.fill([4.3, 4.8, 4.8, 4.3], [0.75, 0.75, 0.95, 0.95],
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        
        wheel1 = plt.Circle((0.8, 0.33), 0.28, fill=True, color='#333333',
                            edgecolor='#888888', linewidth=2)
        wheel2 = plt.Circle((4.0, 0.33), 0.28, fill=True, color='#333333',
                            edgecolor='#888888', linewidth=2)
        ax.add_patch(wheel1)
        ax.add_patch(wheel2)
        
        # Ground
        ax.axhline(y=0, color='#444444', linewidth=3)
        ax.fill_between([-2, 8], -0.5, 0, color='#0a0a0a')
        
        # Calculate forces
        q = 0.5 * AIR_DENSITY * V_inf**2
        downforce = q * PERRINN['sCz'] / 9.81
        drag = q * PERRINN['sCx']
        
        # Info panel
        info_text = f"""Speed: {speed_kmh:.0f} km/h
Downforce: {downforce:,.0f} kg
Drag: {drag:,.0f} N
L/D: {downforce*9.81/drag:.2f}"""
        
        ax.text(6.5, 2.2, info_text, fontsize=11, fontfamily='monospace',
               color='white', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#e10600', alpha=0.9))
        
        # Annotations
        ax.annotate('LOW PRESSURE\n(Suction Zone)', xy=(2.5, -0.15), fontsize=9,
                   color='white', ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        ax.annotate('HIGH PRESSURE\n(Stagnation)', xy=(-0.3, 0.5), fontsize=9,
                   color='white', ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        ax.set_xlim(-2, 8)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title(f'F1 PRESSURE COEFFICIENT FIELD - Dynamic Speed Simulation',
                    fontsize=14, fontweight='bold', color='#e10600')
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=80, interval=80, blit=False)
    
    print("  Saving GIF...")
    anim.save(OUTPUT_DIR / 'pressure_animation.gif', writer='pillow', fps=12, dpi=100)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'pressure_animation.gif'}")


def animate_force_buildup():
    """Animate force vectors building up with speed."""
    print("Creating force buildup animation...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Car profile
    car_x = np.array([0, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 3.8, 4.2, 4.5, 4.8, 5.0,
                      5.0, 4.8, 4.5, 4.2, 3.8, 3.2, 2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0])
    car_y = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 0.9, 0.85, 0.7, 0.5, 0.3, 0.15,
                      0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08, 0.1, 0.1, 0.1])
    
    def animate(frame):
        ax.clear()
        
        # Speed increases from 50 to 350 km/h
        speed_kmh = 50 + frame * 5
        speed_ms = speed_kmh / 3.6
        
        # Calculate forces
        q = 0.5 * AIR_DENSITY * speed_ms**2
        downforce = q * PERRINN['sCz']
        drag = q * PERRINN['sCx']
        
        # Scale for visualization
        force_scale = 0.00015
        
        # Draw car
        ax.fill(car_x, car_y, color='#1a1a2e', edgecolor='#e10600', linewidth=3)
        
        # Car components
        ax.fill([0, 0.4, 0.4, 0], [0.05, 0.05, 0.18, 0.18], 
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        ax.fill([4.3, 4.8, 4.8, 4.3], [0.75, 0.75, 0.95, 0.95],
                color='#2a2a4a', edgecolor='#00ffff', linewidth=2)
        
        wheel1 = plt.Circle((0.8, 0.33), 0.28, fill=True, color='#333333',
                            edgecolor='#666666', linewidth=2)
        wheel2 = plt.Circle((4.0, 0.33), 0.28, fill=True, color='#333333',
                            edgecolor='#666666', linewidth=2)
        ax.add_patch(wheel1)
        ax.add_patch(wheel2)
        
        # Ground
        ax.axhline(y=0, color='#444444', linewidth=3)
        ax.fill_between([-1, 7], -0.8, 0, color='#0a0a0a')
        
        # Draw force arrows
        # Total downforce (green)
        df_arrow = downforce * force_scale
        ax.annotate('', xy=(2.5, 0.5 - df_arrow), xytext=(2.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color='#00ff00', lw=4))
        
        # Component downforces
        front_df = downforce * 0.35 * force_scale
        ax.annotate('', xy=(0.5, 0.15 - front_df*0.7), xytext=(0.5, 0.15),
                   arrowprops=dict(arrowstyle='->', color='#00cc00', lw=2.5))
        
        rear_df = downforce * 0.35 * force_scale
        ax.annotate('', xy=(4.5, 0.85 - rear_df*0.7), xytext=(4.5, 0.85),
                   arrowprops=dict(arrowstyle='->', color='#00cc00', lw=2.5))
        
        floor_df = downforce * 0.30 * force_scale
        ax.annotate('', xy=(2.5, 0.05 - floor_df*0.5), xytext=(2.5, 0.1),
                   arrowprops=dict(arrowstyle='->', color='#00aaaa', lw=2.5))
        
        # Drag (red)
        drag_arrow = drag * force_scale
        ax.annotate('', xy=(3.5 - drag_arrow, 0.5), xytext=(3.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color='#ff4444', lw=4))
        
        # Airflow arrows
        for y_pos in [0.2, 0.5, 0.8, 1.1]:
            ax.annotate('', xy=(-0.3, y_pos), xytext=(-0.8, y_pos),
                       arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))
        
        # Speed indicator (speedometer style)
        theta = np.linspace(-np.pi*0.75, np.pi*0.75, 100)
        r = 0.6
        cx, cy = 6.3, 1.8
        
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 
               color='#444444', linewidth=3)
        
        # Speed needle
        speed_angle = -np.pi*0.75 + (speed_kmh / 400) * np.pi * 1.5
        ax.plot([cx, cx + r*0.9*np.cos(speed_angle)], 
               [cy, cy + r*0.9*np.sin(speed_angle)], 
               color='#e10600', linewidth=3)
        ax.text(cx, cy - 0.15, f'{speed_kmh:.0f}', ha='center', fontsize=14, 
               fontweight='bold', color='#e10600')
        ax.text(cx, cy - 0.35, 'km/h', ha='center', fontsize=10, color='#888888')
        
        # Force meters
        meter_x = 6.3
        
        # Downforce meter
        ax.text(meter_x, 0.9, 'DOWNFORCE', ha='center', fontsize=9, color='#00ff00')
        ax.add_patch(plt.Rectangle((meter_x-0.4, 0.2), 0.8, 0.6, 
                                   fill=False, edgecolor='#00ff00', linewidth=2))
        df_height = min(0.58, (downforce / 30000) * 0.58)
        ax.add_patch(plt.Rectangle((meter_x-0.38, 0.22), 0.76, df_height,
                                   fill=True, color='#00ff00', alpha=0.5))
        ax.text(meter_x, 0.5, f'{downforce/9.81:.0f} kg', ha='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Drag meter
        ax.text(meter_x, -0.15, 'DRAG', ha='center', fontsize=9, color='#ff4444')
        ax.add_patch(plt.Rectangle((meter_x-0.4, -0.75), 0.8, 0.5,
                                   fill=False, edgecolor='#ff4444', linewidth=2))
        drag_height = min(0.48, (drag / 15000) * 0.48)
        ax.add_patch(plt.Rectangle((meter_x-0.38, -0.73), 0.76, drag_height,
                                   fill=True, color='#ff4444', alpha=0.5))
        ax.text(meter_x, -0.5, f'{drag:.0f} N', ha='center',
               fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlim(-1, 7.5)
        ax.set_ylim(-0.8, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('F1 AERODYNAMIC FORCES - Speed Buildup Simulation',
                    fontsize=16, fontweight='bold', color='#e10600', pad=10)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=False)
    
    print("  Saving GIF...")
    anim.save(OUTPUT_DIR / 'force_buildup.gif', writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'force_buildup.gif'}")


def main():
    print("=" * 60)
    print("F1 CFD 3D ANIMATED VISUALIZATIONS")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate all animations
    animate_3d_rotation()
    animate_streamline_flow()
    animate_pressure_field()
    animate_force_buildup()
    
    print("\n" + "=" * 60)
    print("ALL ANIMATIONS COMPLETE!")
    print("=" * 60)
    print(f"""
Generated GIF files:
  • 3d_rotation.gif       - Rotating 3D view with streamlines
  • streamline_flow.gif   - Real-time particle flow simulation  
  • pressure_animation.gif - Dynamic pressure field
  • force_buildup.gif     - Force vectors growing with speed
    """)


if __name__ == "__main__":
    main()
