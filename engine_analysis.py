# -*- coding: utf-8 -*-
"""
F1 Engine & Power Unit Analysis
Visualizations for engine specifications, manufacturers, and evolution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "reference"
OUTPUT_DIR = BASE_DIR / "engine_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_engine_data():
    """Load engine and manufacturer data."""
    engines_path = DATA_DIR / "engines.csv"
    manufacturers_path = DATA_DIR / "engine_manufacturers.csv"
    
    engines_df = pd.read_csv(engines_path) if engines_path.exists() else pd.DataFrame()
    manufacturers_df = pd.read_csv(manufacturers_path) if manufacturers_path.exists() else pd.DataFrame()
    
    return engines_df, manufacturers_df


# =============================================================================
# ENGINE CONFIGURATION ANALYSIS
# =============================================================================

def plot_engine_configurations(engines_df, save_name="engine_configurations"):
    """
    Pie chart showing distribution of engine configurations (V6, V8, V10, etc.)
    """
    if engines_df.empty or 'configuration' not in engines_df.columns:
        print("  No engine configuration data available")
        return
    
    config_counts = engines_df['configuration'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(config_counts)))
    
    wedges, texts, autotexts = ax.pie(config_counts.values, 
                                       labels=config_counts.index,
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       explode=[0.02] * len(config_counts),
                                       pctdistance=0.8)
    
    ax.set_title('F1 Engine Configurations (All-Time)\nFrom V16 to V6 Hybrid', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_aspiration_types(engines_df, save_name="aspiration_types"):
    """
    Bar chart of engine aspiration types (NA, Turbo, Hybrid, Supercharged).
    """
    if engines_df.empty or 'aspiration' not in engines_df.columns:
        print("  No aspiration data available")
        return
    
    # Clean up aspiration names
    asp_counts = engines_df['aspiration'].value_counts()
    
    labels = {
        'NATURALLY_ASPIRATED': 'Naturally Aspirated',
        'TURBOCHARGED': 'Turbocharged',
        'TURBOCHARGED_HYBRID': 'Turbo Hybrid (2014+)',
        'SUPERCHARGED': 'Supercharged'
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#00ff00', '#ff6600', '#ff00ff', '#ffff00']
    
    bars = ax.bar([labels.get(x, x) for x in asp_counts.index], 
                  asp_counts.values, color=colors[:len(asp_counts)])
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Number of Engine Types', fontsize=12)
    ax.set_xlabel('Aspiration Type', fontsize=12)
    ax.set_title('F1 Engine Aspiration Types (All-Time)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_engine_capacity_distribution(engines_df, save_name="engine_capacity"):
    """
    Histogram of engine capacities across F1 history.
    """
    if engines_df.empty or 'capacity' not in engines_df.columns:
        print("  No capacity data available")
        return
    
    # Filter valid capacities
    capacities = engines_df['capacity'].dropna()
    capacities = pd.to_numeric(capacities, errors='coerce').dropna()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.hist(capacities, bins=20, color='#00ffff', edgecolor='black', alpha=0.7)
    
    ax.axvline(x=1.6, color='#ff00ff', linewidth=2, linestyle='--', label='Current (1.6L V6 Hybrid)')
    ax.axvline(x=2.4, color='#ff6600', linewidth=2, linestyle='--', label='V8 Era (2.4L)')
    ax.axvline(x=3.0, color='#00ff00', linewidth=2, linestyle='--', label='V10 Era (3.0L)')
    
    ax.set_xlabel('Engine Capacity (Liters)', fontsize=12)
    ax.set_ylabel('Number of Engines', fontsize=12)
    ax.set_title('F1 Engine Capacity Distribution', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# MANUFACTURER ANALYSIS
# =============================================================================

def plot_engines_by_manufacturer(engines_df, save_name="engines_by_manufacturer"):
    """
    Bar chart showing number of engine models per manufacturer.
    """
    if engines_df.empty or 'engineManufacturerId' not in engines_df.columns:
        print("  No manufacturer data available")
        return
    
    mfr_counts = engines_df['engineManufacturerId'].value_counts().head(20)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(mfr_counts)))
    
    bars = ax.barh(mfr_counts.index[::-1], mfr_counts.values[::-1], color=colors[::-1])
    
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(bar.get_width())), va='center', fontsize=10)
    
    ax.set_xlabel('Number of Engine Models', fontsize=12)
    ax.set_ylabel('Manufacturer', fontsize=12)
    ax.set_title('F1 Engine Models by Manufacturer (Top 20)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_manufacturer_countries(manufacturers_df, save_name="manufacturer_countries"):
    """
    Pie chart showing engine manufacturers by country.
    """
    if manufacturers_df.empty or 'countryId' not in manufacturers_df.columns:
        print("  No country data available")
        return
    
    country_counts = manufacturers_df['countryId'].value_counts()
    
    # Pretty names
    country_names = {
        'united-kingdom': 'United Kingdom',
        'italy': 'Italy',
        'japan': 'Japan',
        'france': 'France',
        'germany': 'Germany',
        'united-states-of-america': 'USA',
        'switzerland': 'Switzerland',
        'australia': 'Australia',
        'malaysia': 'Malaysia',
        'taiwan': 'Taiwan',
        'netherlands': 'Netherlands',
        'luxembourg': 'Luxembourg',
        'austria': 'Austria'
    }
    
    labels = [country_names.get(c, c) for c in country_counts.index]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab20(range(len(country_counts)))
    
    wedges, texts, autotexts = ax.pie(country_counts.values,
                                       labels=labels,
                                       autopct='%1.0f%%',
                                       colors=colors,
                                       pctdistance=0.8)
    
    ax.set_title('F1 Engine Manufacturers by Country', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# HYBRID ERA ANALYSIS (2014-2025)
# =============================================================================

def plot_hybrid_era_engines(engines_df, save_name="hybrid_era_engines"):
    """
    Focus on the current turbo hybrid power units.
    """
    if engines_df.empty:
        print("  No engine data available")
        return
    
    hybrid_engines = engines_df[engines_df['aspiration'] == 'TURBOCHARGED_HYBRID'].copy()
    
    if hybrid_engines.empty:
        print("  No hybrid engine data available")
        return
    
    # Group by manufacturer
    mfr_counts = hybrid_engines['engineManufacturerId'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Engines per manufacturer
    ax1 = axes[0]
    colors_map = {
        'mercedes': '#00d2be',
        'ferrari': '#dc0000',
        'honda': '#e40046',
        'renault': '#fff500',
        'honda-rbpt': '#0600ef',
        'rbpt': '#0600ef',
        'tag-heuer': '#ff8700',
        'bwt-mercedes': '#ff69b4',
        'toro-rosso': '#469BFF'
    }
    
    colors = [colors_map.get(m, '#888888') for m in mfr_counts.index]
    
    bars = ax1.bar(mfr_counts.index, mfr_counts.values, color=colors)
    
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=11)
    
    ax1.set_ylabel('Number of Engine Versions', fontsize=12)
    ax1.set_xlabel('Manufacturer', fontsize=12)
    ax1.set_title('Turbo Hybrid Power Units (2014-2025)\nVersions by Manufacturer', 
                 fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Engine name examples
    ax2 = axes[1]
    
    # Get latest engines per manufacturer
    latest_engines = []
    for mfr in ['mercedes', 'ferrari', 'honda', 'renault', 'honda-rbpt']:
        mfr_engines = hybrid_engines[hybrid_engines['engineManufacturerId'] == mfr]
        if not mfr_engines.empty:
            latest = mfr_engines.iloc[-1]
            latest_engines.append({
                'Manufacturer': mfr.replace('-', ' ').title(),
                'Engine': latest['name'],
                'Full Name': latest['fullName']
            })
    
    if latest_engines:
        df = pd.DataFrame(latest_engines)
        ax2.axis('off')
        table = ax2.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='left',
                         loc='center',
                         colColours=['#333333']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        ax2.set_title('Current Power Units (2024-2025)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_engine_evolution_timeline(engines_df, save_name="engine_evolution"):
    """
    Timeline showing engine evolution through F1 eras.
    """
    if engines_df.empty:
        print("  No engine data available")
        return
    
    # Define F1 engine eras
    eras = [
        ('1950-1953', 'Supercharged Era', '1.5L S / 4.5L NA', 'SUPERCHARGED'),
        ('1954-1960', 'Front Engine Era', '2.5L NA', 'NATURALLY_ASPIRATED'),
        ('1961-1965', '1.5L Era', '1.5L NA', 'NATURALLY_ASPIRATED'),
        ('1966-1988', '3.0L Era', '3.0L NA / 1.5L Turbo', 'NATURALLY_ASPIRATED'),
        ('1989-1994', '3.5L NA Era', '3.5L NA V10/V12', 'NATURALLY_ASPIRATED'),
        ('1995-2005', '3.0L V10 Era', '3.0L NA V10', 'NATURALLY_ASPIRATED'),
        ('2006-2013', '2.4L V8 Era', '2.4L NA V8', 'NATURALLY_ASPIRATED'),
        ('2014-2025', 'Turbo Hybrid Era', '1.6L V6 Turbo Hybrid', 'TURBOCHARGED_HYBRID'),
    ]
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(eras)))
    
    y_positions = range(len(eras))
    
    for i, (period, name, spec, asp_type) in enumerate(eras):
        # Count engines of this type
        if asp_type == 'TURBOCHARGED_HYBRID':
            count = len(engines_df[engines_df['aspiration'] == asp_type])
        elif '3.0' in spec and 'V10' in spec:
            count = len(engines_df[(engines_df['capacity'] == 3.0) & (engines_df['configuration'] == 'V10')])
        elif '2.4' in spec:
            count = len(engines_df[engines_df['capacity'] == 2.4])
        elif '3.5' in spec:
            count = len(engines_df[engines_df['capacity'] == 3.5])
        else:
            count = 20  # Approximate for historical eras
        
        bar = ax.barh(i, count, color=colors[i], height=0.7)
        
        ax.text(-2, i, f"{period}", va='center', ha='right', fontsize=11, fontweight='bold')
        ax.text(count + 1, i, f"{name}\n{spec}", va='center', ha='left', fontsize=10)
    
    ax.set_yticks([])
    ax.set_xlabel('Number of Engine Models', fontsize=12)
    ax.set_title('F1 Engine Evolution Through the Eras', fontsize=16, fontweight='bold')
    ax.set_xlim(-30, engines_df['aspiration'].value_counts().max() + 50)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_config_by_era(engines_df, save_name="config_by_era"):
    """
    Stacked bar chart showing engine configurations over different capacity eras.
    """
    if engines_df.empty:
        print("  No engine data available")
        return
    
    # Group by capacity and configuration
    capacity_config = engines_df.groupby(['capacity', 'configuration']).size().unstack(fill_value=0)
    
    # Select most common capacities
    top_capacities = engines_df['capacity'].value_counts().head(8).index.tolist()
    capacity_config = capacity_config.loc[capacity_config.index.isin(top_capacities)]
    capacity_config = capacity_config.sort_index()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    capacity_config.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_xlabel('Engine Capacity (Liters)', fontsize=12)
    ax.set_ylabel('Number of Engines', fontsize=12)
    ax.set_title('F1 Engine Configurations by Capacity', fontsize=16, fontweight='bold')
    ax.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_current_power_unit_specs(save_name="current_pu_specs"):
    """
    Infographic showing current power unit specifications.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Current PU specs
    specs = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    F1 POWER UNIT SPECIFICATIONS                   ║
    ║                         (2014-2025 Era)                           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  ENGINE TYPE:           1.6L V6 Turbocharged Hybrid               ║
    ║  CONFIGURATION:         90° V6 with single turbocharger           ║
    ║  RPM LIMIT:             15,000 RPM (actual ~12,500 used)          ║
    ║  FUEL FLOW LIMIT:       100 kg/hour max                           ║
    ║                                                                    ║
    ║  ═══════════════════════ POWER OUTPUT ═══════════════════════    ║
    ║                                                                    ║
    ║  ICE (Internal Combustion):     ~550 HP                           ║
    ║  MGU-K (Kinetic):               ~160 HP (120kW max)               ║
    ║  MGU-H (Heat):                  Energy recovery only              ║
    ║  ─────────────────────────────────────────────                    ║
    ║  TOTAL POWER:                   ~1,000+ HP                        ║
    ║                                                                    ║
    ║  ═══════════════════ ENERGY RECOVERY ════════════════════════    ║
    ║                                                                    ║
    ║  ES (Energy Store):             4 MJ max capacity                 ║
    ║  MGU-K Recovery:                2 MJ per lap                      ║
    ║  Thermal Efficiency:            50%+ (road cars: ~30%)            ║
    ║                                                                    ║
    ║  ══════════════════ CURRENT SUPPLIERS ═══════════════════════    ║
    ║                                                                    ║
    ║  🔴 FERRARI    │  066/12 (2024), 066/13 (2025)                   ║
    ║  🔵 MERCEDES   │  M15 (2024), M16 (2025)                         ║
    ║  🟠 HONDA RBPT │  RBPTH002 (2024-2025)                           ║
    ║  🟡 RENAULT    │  E-Tech RE24 (2024), RE25 (2025)                ║
    ║                                                                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, specs, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', color='#00ffff',
            bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#00ffff'))
    
    ax.set_title('Current F1 Power Unit Technical Specifications', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


def plot_v_engine_comparison(engines_df, save_name="v_engine_comparison"):
    """
    Compare V6, V8, V10, V12 engines.
    """
    if engines_df.empty:
        print("  No engine data available")
        return
    
    v_configs = ['V6', 'V8', 'V10', 'V12']
    v_engines = engines_df[engines_df['configuration'].isin(v_configs)]
    
    config_counts = v_engines['configuration'].value_counts().reindex(v_configs, fill_value=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Count comparison
    ax1 = axes[0]
    colors = ['#ff00ff', '#00ff00', '#00ffff', '#ffff00']
    bars = ax1.bar(config_counts.index, config_counts.values, color=colors)
    
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=14)
    
    ax1.set_ylabel('Number of Engine Models', fontsize=12)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_title('V-Engine Configurations in F1', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Manufacturers per config
    ax2 = axes[1]
    
    config_mfrs = v_engines.groupby('configuration')['engineManufacturerId'].nunique()
    config_mfrs = config_mfrs.reindex(v_configs, fill_value=0)
    
    bars2 = ax2.bar(config_mfrs.index, config_mfrs.values, color=colors)
    
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=14)
    
    ax2.set_ylabel('Number of Manufacturers', fontsize=12)
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_title('Manufacturers Who Built Each Config', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('F1 V-Engine Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("F1 ENGINE & POWER UNIT ANALYSIS")
    print("=" * 60)
    
    print("\nLoading engine data...")
    engines_df, manufacturers_df = load_engine_data()
    
    print(f"Loaded {len(engines_df)} engine records")
    print(f"Loaded {len(manufacturers_df)} manufacturer records")
    
    print("\n" + "=" * 60)
    print("GENERATING ENGINE VISUALIZATIONS")
    print("=" * 60)
    
    print("\nEngine Configuration Analysis...")
    plot_engine_configurations(engines_df)
    plot_aspiration_types(engines_df)
    plot_engine_capacity_distribution(engines_df)
    
    print("\nManufacturer Analysis...")
    plot_engines_by_manufacturer(engines_df)
    plot_manufacturer_countries(manufacturers_df)
    
    print("\nHybrid Era Analysis...")
    plot_hybrid_era_engines(engines_df)
    plot_current_power_unit_specs()
    
    print("\nEvolution & Comparison...")
    plot_engine_evolution_timeline(engines_df)
    plot_config_by_era(engines_df)
    plot_v_engine_comparison(engines_df)
    
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
