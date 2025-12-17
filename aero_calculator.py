# -*- coding: utf-8 -*-
"""
F1 Aerodynamic Calculator
Calculate downforce and drag at any speed using CFD coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('dark_background')

# Air properties (standard conditions)
AIR_DENSITY = 1.225  # kg/m³

# PERRINN 2017 F1 Car CFD Data (from research)
PERRINN_DATA = {
    'frontal_area': 1.5,  # m²
    'sCx': 1.16,          # m² (drag coefficient × area)
    'sCz': 3.25,          # m² (downforce coefficient × area)
    'Cd': 1.16 / 1.5,     # ~0.77
    'Cl': 3.25 / 1.5,     # ~2.17
}

# Typical F1 setup ranges
SETUPS = {
    'high_downforce': {'sCx': 1.35, 'sCz': 4.0, 'name': 'Monaco/Hungary'},
    'medium_downforce': {'sCx': 1.16, 'sCz': 3.25, 'name': 'Silverstone/Spa'},
    'low_downforce': {'sCx': 0.95, 'sCz': 2.5, 'name': 'Monza/Baku'},
}

OUTPUT_DIR = Path(__file__).parent / 'windtunnel_data' / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def calculate_downforce(speed_kmh, sCz=PERRINN_DATA['sCz'], rho=AIR_DENSITY):
    """
    Calculate downforce at a given speed.
    
    Downforce = 0.5 × ρ × V² × sCz
    
    Args:
        speed_kmh: Speed in km/h
        sCz: Downforce coefficient × frontal area (m²)
        rho: Air density (kg/m³)
    
    Returns:
        Downforce in Newtons
    """
    speed_ms = speed_kmh / 3.6
    return 0.5 * rho * speed_ms**2 * sCz


def calculate_drag(speed_kmh, sCx=PERRINN_DATA['sCx'], rho=AIR_DENSITY):
    """
    Calculate drag force at a given speed.
    
    Drag = 0.5 × ρ × V² × sCx
    
    Args:
        speed_kmh: Speed in km/h
        sCx: Drag coefficient × frontal area (m²)
        rho: Air density (kg/m³)
    
    Returns:
        Drag force in Newtons
    """
    speed_ms = speed_kmh / 3.6
    return 0.5 * rho * speed_ms**2 * sCx


def calculate_power_to_overcome_drag(speed_kmh, sCx=PERRINN_DATA['sCx'], rho=AIR_DENSITY):
    """
    Calculate power required to overcome drag.
    
    Power = Drag × Velocity
    
    Returns:
        Power in kW
    """
    speed_ms = speed_kmh / 3.6
    drag = calculate_drag(speed_kmh, sCx, rho)
    power_watts = drag * speed_ms
    return power_watts / 1000  # kW


def print_aero_table(speeds=[100, 150, 200, 250, 300, 350]):
    """Print aerodynamic forces at various speeds."""
    print("\n" + "=" * 80)
    print("F1 AERODYNAMIC FORCE CALCULATOR")
    print("Based on PERRINN 2017 F1 Car CFD Data")
    print("=" * 80)
    
    print(f"\nReference values: sCx = {PERRINN_DATA['sCx']} m², sCz = {PERRINN_DATA['sCz']} m²")
    print(f"Air density: {AIR_DENSITY} kg/m³")
    
    print("\n" + "-" * 80)
    print(f"{'Speed':>10} {'Downforce':>15} {'Drag':>15} {'L/D':>10} {'Power':>15}")
    print(f"{'(km/h)':>10} {'(N / kg)':>15} {'(N)':>15} {'-':>10} {'(kW / HP)':>15}")
    print("-" * 80)
    
    for speed in speeds:
        downforce = calculate_downforce(speed)
        drag = calculate_drag(speed)
        ld_ratio = downforce / drag if drag > 0 else 0
        power = calculate_power_to_overcome_drag(speed)
        
        downforce_kg = downforce / 9.81
        power_hp = power * 1.341
        
        print(f"{speed:>10} {downforce:>8.0f} / {downforce_kg:>5.0f} {drag:>15.0f} {ld_ratio:>10.2f} {power:>7.0f} / {power_hp:>5.0f}")
    
    print("-" * 80)
    print("\nNote: At 250 km/h, downforce (~22,000N / 2,200kg) exceeds car weight (~800kg)!")
    print("      This means an F1 car could theoretically drive upside down at high speed.")


def plot_downforce_vs_speed():
    """Plot downforce and drag curves for different setups."""
    speeds = np.linspace(0, 350, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Downforce comparison
    ax1 = axes[0, 0]
    for setup_name, setup in SETUPS.items():
        downforce = [calculate_downforce(s, setup['sCz']) for s in speeds]
        downforce_kg = [d / 9.81 for d in downforce]
        ax1.plot(speeds, downforce_kg, linewidth=2, label=f"{setup['name']} (sCz={setup['sCz']})")
    
    ax1.axhline(y=800, color='white', linestyle='--', alpha=0.5, label='Car Weight (~800kg)')
    ax1.set_xlabel('Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Downforce (kg)', fontsize=12)
    ax1.set_title('Downforce vs Speed by Setup', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drag comparison
    ax2 = axes[0, 1]
    for setup_name, setup in SETUPS.items():
        drag = [calculate_drag(s, setup['sCx']) for s in speeds]
        ax2.plot(speeds, drag, linewidth=2, label=f"{setup['name']} (sCx={setup['sCx']})")
    
    ax2.set_xlabel('Speed (km/h)', fontsize=12)
    ax2.set_ylabel('Drag Force (N)', fontsize=12)
    ax2.set_title('Drag vs Speed by Setup', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: L/D Efficiency
    ax3 = axes[1, 0]
    for setup_name, setup in SETUPS.items():
        ld = setup['sCz'] / setup['sCx']
        ax3.bar(setup['name'], ld, color=plt.cm.viridis(list(SETUPS.keys()).index(setup_name) / 3))
        ax3.text(setup['name'], ld + 0.05, f'{ld:.2f}', ha='center', fontsize=11)
    
    ax3.set_ylabel('L/D Ratio (Efficiency)', fontsize=12)
    ax3.set_title('Aerodynamic Efficiency by Setup', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Power required
    ax4 = axes[1, 1]
    for setup_name, setup in SETUPS.items():
        power = [calculate_power_to_overcome_drag(s, setup['sCx']) for s in speeds]
        ax4.plot(speeds, power, linewidth=2, label=f"{setup['name']}")
    
    ax4.axhline(y=750, color='#ff0000', linestyle='--', alpha=0.7, label='~1000 HP limit')
    ax4.set_xlabel('Speed (km/h)', fontsize=12)
    ax4.set_ylabel('Power to Overcome Drag (kW)', fontsize=12)
    ax4.set_title('Power Required for Drag', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('F1 Aerodynamic Calculator\nBased on PERRINN 2017 CFD Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'aero_calculator_plots.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'aero_calculator_plots.png'}")


def main():
    print_aero_table()
    plot_downforce_vs_speed()
    
    print("\n" + "=" * 80)
    print("QUICK CALCULATIONS")
    print("=" * 80)
    
    print("\n250 km/h (typical corner entry speed):")
    print(f"  Downforce: {calculate_downforce(250):.0f} N ({calculate_downforce(250)/9.81:.0f} kg)")
    print(f"  Drag: {calculate_drag(250):.0f} N")
    
    print("\n350 km/h (Monza top speed):")
    print(f"  Downforce: {calculate_downforce(350, SETUPS['low_downforce']['sCz']):.0f} N")
    print(f"  Drag: {calculate_drag(350, SETUPS['low_downforce']['sCx']):.0f} N")
    print(f"  Power needed: {calculate_power_to_overcome_drag(350, SETUPS['low_downforce']['sCx']):.0f} kW")


if __name__ == "__main__":
    main()
