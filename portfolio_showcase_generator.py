# -*- coding: utf-8 -*-
"""
F1 Data Analysis Video/Image Compiler
Creates a visual summary of all project visualizations
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np

plt.style.use('dark_background')

OUTPUT_DIR = Path(__file__).parent / 'portfolio_showcase'
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECT_DIR = Path(__file__).parent


def find_all_visualizations():
    """Find all visualization files in the project."""
    viz_dirs = [
        'telemetry_visualizations',
        'gforce_visualizations',
        'tire_pitstop_visualizations',
        'windtunnel_data/visualizations',
        'engine_visualizations',
        'aero_visualizations',
        'cfd_visualizations',
        'race_simulations',
    ]
    
    all_files = []
    for dir_name in viz_dirs:
        dir_path = PROJECT_DIR / dir_name
        if dir_path.exists():
            for ext in ['*.png', '*.gif']:
                files = list(dir_path.glob(ext))
                all_files.extend(files)
    
    return all_files


def create_portfolio_montage():
    """Create a montage of key visualizations."""
    print("Creating portfolio montage...")
    
    # Key visualizations to include
    key_images = [
        ('telemetry_visualizations', '2024_Monaco_Grand_Prix_track_map.png', 'Track Map'),
        ('gforce_visualizations', 'gforce_track_gauge_comparison.png', 'G-Force Analysis'),
        ('cfd_visualizations', 'pressure_velocity_fields.png', 'CFD Pressure Fields'),
        ('cfd_visualizations', 'streamlines_visualization.png', 'Streamlines'),
        ('cfd_visualizations', 'force_vectors.png', 'Force Vectors'),
        ('cfd_visualizations', 'component_breakdown.png', 'Aero Components'),
        ('windtunnel_data/visualizations', 'aero_calculator_plots.png', 'Aero Calculator'),
        ('tire_pitstop_visualizations', 'pit_stop_distribution_2024.png', 'Pit Stops'),
    ]
    
    # Filter to existing files
    existing_images = []
    for dir_name, file_name, label in key_images:
        path = PROJECT_DIR / dir_name / file_name
        if path.exists():
            existing_images.append((path, label))
    
    if len(existing_images) < 4:
        print("  Not enough images found for montage")
        return
    
    # Create montage
    n_images = min(len(existing_images), 8)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(24, 6 * n_rows))
    
    # Title
    fig.suptitle('F1 DATA ANALYSIS PROJECT SHOWCASE',
                fontsize=28, fontweight='bold', color='#e10600', y=0.98)
    
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, (img_path, label) in enumerate(existing_images[:n_images]):
        row = i // n_cols
        col = i % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(label, fontsize=12, fontweight='bold', 
                        color='#00d2be', pad=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{img_path.name}',
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    # Footer
    fig.text(0.5, 0.02, 
            'Python | FastF1 | OpenFOAM | Matplotlib | NumPy | Pandas',
            ha='center', fontsize=14, color='#666666')
    
    plt.savefig(OUTPUT_DIR / 'portfolio_montage.png', dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'portfolio_montage.png'}")


def create_cfd_showcase():
    """Create a CFD-focused showcase image."""
    print("Creating CFD showcase...")
    
    cfd_images = [
        ('cfd_visualizations/streamlines_visualization.png', 'Streamlines'),
        ('cfd_visualizations/pressure_velocity_fields.png', 'Pressure & Velocity'),
        ('cfd_visualizations/force_vectors.png', 'Force Vectors'),
        ('cfd_visualizations/ground_effect.png', 'Ground Effect'),
        ('cfd_visualizations/speed_comparison.png', 'Speed Comparison'),
        ('cfd_visualizations/cfd_infographic.png', 'CFD Summary'),
    ]
    
    existing = []
    for path, label in cfd_images:
        full_path = PROJECT_DIR / path
        if full_path.exists():
            existing.append((full_path, label))
    
    if len(existing) < 4:
        print("  Not enough CFD images found")
        return
    
    fig = plt.figure(figsize=(20, 15))
    
    fig.suptitle('CFD AERODYNAMIC SIMULATION SHOWCASE\nBased on PERRINN F1 Data',
                fontsize=24, fontweight='bold', color='#e10600', y=0.98)
    
    n_images = min(len(existing), 6)
    n_cols = 3
    n_rows = 2
    
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.1)
    
    for i, (img_path, label) in enumerate(existing[:n_images]):
        row = i // n_cols
        col = i % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(label, fontsize=14, fontweight='bold', color='#00d2be')
        except:
            pass
        
        ax.axis('off')
    
    plt.savefig(OUTPUT_DIR / 'cfd_showcase.png', dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'cfd_showcase.png'}")


def create_race_simulation_showcase():
    """Create a race simulation showcase."""
    print("Creating race simulation showcase...")
    
    # Find race simulation images
    race_dir = PROJECT_DIR / 'race_simulations'
    if not race_dir.exists():
        print("  Race simulations directory not found")
        return
    
    track_comparison = race_dir / 'track_comparison.png'
    
    if track_comparison.exists():
        fig = plt.figure(figsize=(20, 12))
        
        fig.suptitle('F1 RACE SIMULATION - TELEMETRY DRIVEN',
                    fontsize=24, fontweight='bold', color='#e10600', y=0.98)
        
        ax = fig.add_subplot(111)
        img = mpimg.imread(str(track_comparison))
        ax.imshow(img)
        ax.axis('off')
        
        # Add info text
        fig.text(0.5, 0.02,
                'Monaco | Spa | Monza - Real FastF1 Telemetry Data',
                ha='center', fontsize=16, color='#888888')
        
        plt.savefig(OUTPUT_DIR / 'race_showcase.png', dpi=150,
                    bbox_inches='tight', facecolor='black')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'race_showcase.png'}")


def create_project_summary():
    """Create a comprehensive project summary image."""
    print("Creating project summary...")
    
    fig = plt.figure(figsize=(20, 24))
    
    # Title
    fig.suptitle('F1 DATA ANALYSIS & CFD SIMULATION PROJECT',
                fontsize=28, fontweight='bold', color='#e10600', y=0.97)
    
    gs = GridSpec(4, 3, figure=fig, hspace=0.25, wspace=0.2,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Project stats
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis('off')
    
    stats_text = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                              PROJECT STATISTICS                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║   📊 15+ Python Analysis Scripts        🏎️ 3 Tracks Analyzed (Monaco, Spa, Monza)    ║
    ║   🖼️ 50+ Visualizations Generated      💨 CFD Simulations with PERRINN Data         ║
    ║   📈 Real FastF1 Telemetry Data        🔧 OpenFOAM Case Generator                   ║
    ║   🌪️ Streamline & Pressure Analysis   🏁 Race Simulations with Full Telemetry      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center',
                 fontsize=12, fontfamily='monospace', color='#00d2be')
    
    # Add category images
    categories = [
        ('Telemetry Analysis', 'telemetry_visualizations/2024_Monaco_Grand_Prix_track_map.png'),
        ('G-Force Mapping', 'gforce_visualizations/gforce_track_gauge_comparison.png'),
        ('CFD Streamlines', 'cfd_visualizations/streamlines_visualization.png'),
        ('Pressure Fields', 'cfd_visualizations/pressure_velocity_fields.png'),
        ('Force Analysis', 'cfd_visualizations/force_vectors.png'),
        ('Aero Calculator', 'windtunnel_data/visualizations/aero_calculator_plots.png'),
        ('Ground Effect', 'cfd_visualizations/ground_effect.png'),
        ('Tire Strategy', 'tire_pitstop_visualizations/pit_stop_distribution_2024.png'),
        ('Track Comparison', 'race_simulations/track_comparison.png'),
    ]
    
    for i, (label, path) in enumerate(categories[:9]):
        row = 1 + i // 3
        col = i % 3
        
        ax = fig.add_subplot(gs[row, col])
        
        full_path = PROJECT_DIR / path
        if full_path.exists():
            try:
                img = mpimg.imread(str(full_path))
                ax.imshow(img)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                       transform=ax.transAxes, color='#666')
        else:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                   transform=ax.transAxes, color='#666')
        
        ax.set_title(label, fontsize=12, fontweight='bold', color='#00d2be')
        ax.axis('off')
    
    # Tech stack footer
    fig.text(0.5, 0.02,
            'Python | FastF1 | OpenFOAM | NumPy | Pandas | Matplotlib | ParaView',
            ha='center', fontsize=14, color='#888888')
    
    plt.savefig(OUTPUT_DIR / 'project_summary.png', dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'project_summary.png'}")


def list_all_gifs():
    """List all GIF animations in the project."""
    print("\nAnimated GIFs in project:")
    
    gifs = []
    for root, dirs, files in os.walk(PROJECT_DIR):
        # Skip cache and venv
        if 'cache' in root or 'venv' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.gif'):
                full_path = Path(root) / file
                size_mb = full_path.stat().st_size / (1024 * 1024)
                gifs.append((full_path.relative_to(PROJECT_DIR), size_mb))
    
    for path, size in sorted(gifs):
        print(f"  {path} ({size:.1f} MB)")
    
    return gifs


def main():
    print("=" * 60)
    print("F1 DATA ANALYSIS - PORTFOLIO SHOWCASE GENERATOR")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Find all visualizations
    all_viz = find_all_visualizations()
    print(f"Found {len(all_viz)} visualization files\n")
    
    # Create showcase images
    create_portfolio_montage()
    create_cfd_showcase()
    create_race_simulation_showcase()
    create_project_summary()
    
    # List GIFs
    gifs = list_all_gifs()
    
    print("\n" + "=" * 60)
    print("SHOWCASE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"""
Generated files in {OUTPUT_DIR}:
  - portfolio_montage.png    (8-panel overview)
  - cfd_showcase.png         (CFD visualizations)
  - race_showcase.png        (Race simulations)
  - project_summary.png      (Complete summary)

Found {len(gifs)} animated GIFs for demo purposes.

To create a video from GIFs, use:
  ffmpeg -i animation.gif -movflags faststart output.mp4
""")


if __name__ == "__main__":
    main()
