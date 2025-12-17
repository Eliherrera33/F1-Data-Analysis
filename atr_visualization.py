# -*- coding: utf-8 -*-
"""
FIA Aerodynamic Testing Restrictions (ATR) Visualization
Shows wind tunnel hour allocations based on constructor standings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use('dark_background')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'windtunnel_data'
OUTPUT_DIR = DATA_DIR / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 2024 Constructor Standings (approximate final)
CONSTRUCTORS_2024 = {
    1: {'team': 'McLaren', 'color': '#FF8700'},
    2: {'team': 'Ferrari', 'color': '#DC0000'},
    3: {'team': 'Red Bull', 'color': '#0600EF'},
    4: {'team': 'Mercedes', 'color': '#00D2BE'},
    5: {'team': 'Aston Martin', 'color': '#006F62'},
    6: {'team': 'Alpine', 'color': '#0090FF'},
    7: {'team': 'Haas', 'color': '#FFFFFF'},
    8: {'team': 'RB', 'color': '#2B4562'},
    9: {'team': 'Williams', 'color': '#005AFF'},
    10: {'team': 'Sauber', 'color': '#00E701'},
}

# ATR percentages by position
ATR_PERCENTAGES = {
    1: 70, 2: 75, 3: 80, 4: 85, 5: 90,
    6: 95, 7: 100, 8: 105, 9: 110, 10: 115
}

BASELINE_HOURS = 320  # Wind tunnel hours per year baseline


def load_atr_data():
    """Load ATR allocations from CSV or use defaults."""
    csv_path = DATA_DIR / 'atr_allocations.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        # Generate from constants
        data = []
        for pos in range(1, 11):
            data.append({
                'Position': pos,
                'Team': CONSTRUCTORS_2024[pos]['team'],
                'Percentage': ATR_PERCENTAGES[pos],
                'Hours': int(BASELINE_HOURS * ATR_PERCENTAGES[pos] / 100)
            })
        return pd.DataFrame(data)


def plot_atr_bar_chart():
    """Bar chart of wind tunnel hours by position."""
    df = load_atr_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    teams = [CONSTRUCTORS_2024[i]['team'] for i in range(1, 11)]
    hours = [int(BASELINE_HOURS * ATR_PERCENTAGES[i] / 100) for i in range(1, 11)]
    colors = [CONSTRUCTORS_2024[i]['color'] for i in range(1, 11)]
    
    bars = ax.bar(teams, hours, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, h, pct in zip(bars, hours, [ATR_PERCENTAGES[i] for i in range(1, 11)]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{h}h\n({pct}%)', ha='center', va='bottom', fontsize=10)
    
    # Baseline line
    ax.axhline(y=BASELINE_HOURS, color='#ff00ff', linestyle='--', linewidth=2, 
               label=f'Baseline: {BASELINE_HOURS} hours')
    
    ax.set_ylabel('Wind Tunnel Hours per Year', fontsize=12)
    ax.set_xlabel('Constructor (by 2024 Standing)', fontsize=12)
    ax.set_title('FIA Aerodynamic Testing Restrictions (ATR)\n2025 Wind Tunnel Allocations', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'atr_wind_tunnel_hours.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'atr_wind_tunnel_hours.png'}")


def plot_atr_advantage():
    """Show the development time advantage between positions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate advantage vs P1
    positions = list(range(1, 11))
    hours = [int(BASELINE_HOURS * ATR_PERCENTAGES[i] / 100) for i in positions]
    p1_hours = hours[0]
    advantage_vs_p1 = [h - p1_hours for h in hours]
    
    # Left: Advantage vs P1
    ax1 = axes[0]
    teams = [CONSTRUCTORS_2024[i]['team'] for i in positions]
    colors = [CONSTRUCTORS_2024[i]['color'] for i in positions]
    
    bars = ax1.barh(teams, advantage_vs_p1, color=colors, edgecolor='white')
    
    for bar, adv in zip(bars, advantage_vs_p1):
        x_pos = bar.get_width() + 5 if adv >= 0 else bar.get_width() - 5
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'+{adv}h' if adv >= 0 else f'{adv}h', 
                va='center', fontsize=10)
    
    ax1.axvline(x=0, color='white', linewidth=2)
    ax1.set_xlabel('Extra Hours vs P1 Constructor', fontsize=12)
    ax1.set_title('Wind Tunnel Advantage vs Championship Leader', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Right: Cumulative development over 3 years
    ax2 = axes[1]
    
    years = [1, 2, 3]
    for pos in [1, 5, 10]:
        team = CONSTRUCTORS_2024[pos]['team']
        color = CONSTRUCTORS_2024[pos]['color']
        cumulative = [hours[pos-1] * y for y in years]
        ax2.plot(years, cumulative, 'o-', color=color, linewidth=3, markersize=10, label=f'P{pos}: {team}')
    
    ax2.set_xlabel('Years', fontsize=12)
    ax2.set_ylabel('Cumulative Wind Tunnel Hours', fontsize=12)
    ax2.set_title('Development Time Over 3 Years\n(Assuming Constant Position)', fontsize=14, fontweight='bold')
    ax2.set_xticks(years)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('ATR: Leveling the Playing Field', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'atr_advantage_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'atr_advantage_analysis.png'}")


def plot_atr_infographic():
    """Create an infographic explaining ATR."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    title_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FIA AERODYNAMIC TESTING RESTRICTIONS (ATR)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  PURPOSE: Level the playing field by giving slower teams more development    ║
║           time to catch up with the championship leaders.                    ║
║                                                                               ║
║  ═══════════════════════════ HOW IT WORKS ══════════════════════════════     ║
║                                                                               ║
║  1. Baseline: 320 wind tunnel hours + equivalent CFD (per year)              ║
║                                                                               ║
║  2. Allocations based on PREVIOUS season's constructor standings:            ║
║                                                                               ║
║     P1:  70%  = 224 hours      P6:  95%  = 304 hours                         ║
║     P2:  75%  = 240 hours      P7:  100% = 320 hours                         ║
║     P3:  80%  = 256 hours      P8:  105% = 336 hours                         ║
║     P4:  85%  = 272 hours      P9:  110% = 352 hours                         ║
║     P5:  90%  = 288 hours      P10: 115% = 368 hours                         ║
║                                                                               ║
║  3. Wind tunnel rules: 60% scale model, max 50 m/s airspeed                  ║
║                                                                               ║
║  ═════════════════════════════ IMPACT ══════════════════════════════════     ║
║                                                                               ║
║  P10 (Sauber) gets 144 MORE HOURS than P1 (McLaren)                          ║
║  That's 64% more development time!                                           ║
║                                                                               ║
║  Over 3 years:                                                                ║
║    • P1 team: 672 hours total                                                 ║
║    • P10 team: 1,104 hours total                                              ║
║    • Difference: 432 hours (~18 extra days of testing)                        ║
║                                                                               ║
║  ═══════════════════════════ CRITICISM ═════════════════════════════════     ║
║                                                                               ║
║  • Some say 70% is too restrictive for winning teams                         ║
║  • Others say it's not restrictive enough to close the gap                   ║
║  • CFD correlation with wind tunnel varies by team resources                 ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.5, 0.5, title_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', color='#00ffff',
            bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#00ffff'))
    
    plt.savefig(OUTPUT_DIR / 'atr_infographic.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'atr_infographic.png'}")


def main():
    print("=" * 60)
    print("ATR VISUALIZATION")
    print("=" * 60)
    
    plot_atr_bar_chart()
    plot_atr_advantage()
    plot_atr_infographic()
    
    print("\n" + "=" * 60)
    print("ATR VISUALIZATIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
