# -*- coding: utf-8 -*-
"""
Downforce vs Speed Analysis
Visualize how aerodynamic forces scale with velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.style.use('dark_background')

OUTPUT_DIR = Path(__file__).parent / 'windtunnel_data' / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

AIR_DENSITY = 1.225  # kg/m³

# Setup configurations
SETUPS = {
    'Monaco (High DF)': {'sCx': 1.35, 'sCz': 4.0, 'color': '#FF0000'},
    'Silverstone (Med)': {'sCx': 1.16, 'sCz': 3.25, 'color': '#FFD700'},
    'Monza (Low DF)': {'sCx': 0.95, 'sCz': 2.5, 'color': '#00FF00'},
}


def downforce(speed_kmh, sCz):
    """Calculate downforce in kg."""
    v = speed_kmh / 3.6
    return (0.5 * AIR_DENSITY * v**2 * sCz) / 9.81


def drag(speed_kmh, sCx):
    """Calculate drag in N."""
    v = speed_kmh / 3.6
    return 0.5 * AIR_DENSITY * v**2 * sCx


def plot_downforce_curves():
    """Main downforce analysis visualization."""
    speeds = np.linspace(0, 370, 200)
    
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Downforce vs Speed (main plot)
    ax1 = fig.add_subplot(2, 2, 1)
    
    for name, setup in SETUPS.items():
        df = [downforce(s, setup['sCz']) for s in speeds]
        ax1.plot(speeds, df, linewidth=3, color=setup['color'], label=name)
    
    # Mark key speeds
    key_speeds = [100, 200, 300]
    for s in key_speeds:
        ax1.axvline(x=s, color='white', linestyle=':', alpha=0.3)
    
    ax1.axhline(y=800, color='#ff00ff', linestyle='--', linewidth=2, alpha=0.7, 
                label='Car Weight (~800kg)')
    
    ax1.fill_between(speeds, 0, [downforce(s, SETUPS['Monaco (High DF)']['sCz']) for s in speeds],
                     alpha=0.1, color='#FF0000')
    
    ax1.set_xlabel('Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Downforce (kg)', fontsize=12)
    ax1.set_title('Downforce vs Speed\n(Force scales with V²)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 370)
    ax1.set_ylim(0, 5000)
    
    # Plot 2: Downforce / Weight ratio
    ax2 = fig.add_subplot(2, 2, 2)
    
    car_weight = 800  # kg
    
    for name, setup in SETUPS.items():
        ratio = [downforce(s, setup['sCz']) / car_weight for s in speeds]
        ax2.plot(speeds, ratio, linewidth=3, color=setup['color'], label=name)
    
    ax2.axhline(y=1.0, color='white', linestyle='--', linewidth=2, 
                label='Downforce = Weight')
    ax2.axhline(y=2.0, color='#00ffff', linestyle='--', linewidth=1, alpha=0.5,
                label='Downforce = 2× Weight')
    
    ax2.fill_between(speeds, 0, 1, alpha=0.1, color='#ff0000', label='Normal grip zone')
    ax2.fill_between(speeds, 1, 5, alpha=0.1, color='#00ff00', label='Could drive upside down!')
    
    ax2.set_xlabel('Speed (km/h)', fontsize=12)
    ax2.set_ylabel('Downforce / Car Weight', fontsize=12)
    ax2.set_title('Downforce to Weight Ratio\n(>1 means more DF than weight)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 370)
    ax2.set_ylim(0, 5)
    
    # Plot 3: Top speed potential
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Power available vs drag
    max_power_kw = 750  # ~1000 HP
    
    for name, setup in SETUPS.items():
        # Power required = Drag × Velocity
        power_req = [(drag(s, setup['sCx']) * s / 3.6) / 1000 for s in speeds]
        ax3.plot(speeds, power_req, linewidth=3, color=setup['color'], label=name)
    
    ax3.axhline(y=max_power_kw, color='#ff0000', linestyle='--', linewidth=2, 
                label=f'Max Power (~{max_power_kw} kW / 1000 HP)')
    
    ax3.set_xlabel('Speed (km/h)', fontsize=12)
    ax3.set_ylabel('Power Required (kW)', fontsize=12)
    ax3.set_title('Power Required vs Speed\n(Intersection = theoretical top speed)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 400)
    ax3.set_ylim(0, 1000)
    
    # Add top speed annotations
    for name, setup in SETUPS.items():
        # Find where power_req = max_power
        for s in speeds:
            power_req = (drag(s, setup['sCx']) * s / 3.6) / 1000
            if power_req >= max_power_kw:
                ax3.plot(s, max_power_kw, 'o', color=setup['color'], markersize=12)
                ax3.annotate(f'{s:.0f} km/h', (s, max_power_kw), 
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=10, color=setup['color'])
                break
    
    # Plot 4: L/D efficiency vs speed (flat, but informative)
    ax4 = fig.add_subplot(2, 2, 4)
    
    setup_names = list(SETUPS.keys())
    efficiencies = [SETUPS[n]['sCz'] / SETUPS[n]['sCx'] for n in setup_names]
    colors = [SETUPS[n]['color'] for n in setup_names]
    
    bars = ax4.bar(setup_names, efficiencies, color=colors, edgecolor='white', linewidth=2)
    
    for bar, eff in zip(bars, efficiencies):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'L/D: {eff:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('Aerodynamic Efficiency (L/D)', fontsize=12)
    ax4.set_title('Setup Efficiency Comparison\n(Higher = More DF per unit Drag)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=15)
    
    fig.suptitle('F1 Downforce Analysis: How Speed Changes Everything', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'downforce_vs_speed_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'downforce_vs_speed_analysis.png'}")


def plot_speed_comparison_at_points():
    """Bar chart comparing setups at specific speeds."""
    test_speeds = [100, 150, 200, 250, 300]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    x = np.arange(len(test_speeds))
    width = 0.25
    
    # Downforce comparison
    ax1 = axes[0]
    for i, (name, setup) in enumerate(SETUPS.items()):
        df_values = [downforce(s, setup['sCz']) for s in test_speeds]
        bars = ax1.bar(x + i*width, df_values, width, label=name, color=setup['color'])
    
    ax1.set_xlabel('Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Downforce (kg)', fontsize=12)
    ax1.set_title('Downforce at Different Speeds', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{s}' for s in test_speeds])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Drag comparison
    ax2 = axes[1]
    for i, (name, setup) in enumerate(SETUPS.items()):
        drag_values = [drag(s, setup['sCx']) for s in test_speeds]
        bars = ax2.bar(x + i*width, drag_values, width, label=name, color=setup['color'])
    
    ax2.set_xlabel('Speed (km/h)', fontsize=12)
    ax2.set_ylabel('Drag Force (N)', fontsize=12)
    ax2.set_title('Drag at Different Speeds', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{s}' for s in test_speeds])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'downforce_speed_bars.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'downforce_speed_bars.png'}")


def main():
    print("=" * 60)
    print("DOWNFORCE VS SPEED ANALYSIS")
    print("=" * 60)
    
    plot_downforce_curves()
    plot_speed_comparison_at_points()
    
    # Print key stats
    print("\nKey Statistics:")
    print("-" * 40)
    
    for name, setup in SETUPS.items():
        print(f"\n{name}:")
        print(f"  At 250 km/h: {downforce(250, setup['sCz']):.0f} kg downforce")
        print(f"  At 300 km/h: {downforce(300, setup['sCz']):.0f} kg downforce")
        
        # Find speed where downforce = car weight
        for s in range(50, 400):
            if downforce(s, setup['sCz']) >= 800:
                print(f"  Downforce > Car Weight at: {s} km/h")
                break
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
