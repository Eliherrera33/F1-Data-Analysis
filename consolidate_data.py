# -*- coding: utf-8 -*-
"""
F1 Data Consolidator
Consolidates all F1 data from different sources into a unified format.
"""

import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(__file__).parent
F1DB_CSV_DIR = BASE_DIR / "data" / "f1db_csv"
FORMULA1_DATASETS_DIR = BASE_DIR / "formula1-datasets"
OUTPUT_DIR = BASE_DIR / "data" / "consolidated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_f1db_data():
    """Load all f1db CSV data."""
    data = {
        'race_results': [],
        'qualifying': [],
        'fastest_laps': [],
        'pit_stops': [],
        'driver_standings': [],
        'constructor_standings': []
    }
    
    for data_type in data.keys():
        for csv_file in sorted(F1DB_CSV_DIR.glob(f"*_{data_type}.csv")):
            df = pd.read_csv(csv_file)
            data[data_type].append(df)
    
    # Combine into DataFrames
    result = {}
    for key, dfs in data.items():
        if dfs:
            result[key] = pd.concat(dfs, ignore_index=True)
        else:
            result[key] = pd.DataFrame()
    
    return result


def create_master_lists():
    """Create comprehensive lists of all drivers, constructors, etc."""
    data = load_f1db_data()
    
    lists = {}
    
    # =========================================================================
    # DRIVER LISTS
    # =========================================================================
    
    # All unique drivers
    if not data['race_results'].empty:
        drivers = data['race_results'].groupby('driverId').agg({
            'year': ['min', 'max'],
            'race': 'count'
        }).reset_index()
        drivers.columns = ['driver_id', 'first_year', 'last_year', 'total_races']
        drivers = drivers.sort_values('total_races', ascending=False)
        lists['all_drivers'] = drivers
        
        # Drivers by total wins
        df_copy = data['race_results'].copy()
        df_copy['pos'] = pd.to_numeric(df_copy['position'], errors='coerce')
        wins = df_copy[df_copy['pos'] == 1].groupby('driverId').size().reset_index(name='wins')
        wins = wins.sort_values('wins', ascending=False)
        lists['drivers_by_wins'] = wins
        
        # Drivers by podiums
        podiums = df_copy[df_copy['pos'] <= 3].groupby('driverId').size().reset_index(name='podiums')
        podiums = podiums.sort_values('podiums', ascending=False)
        lists['drivers_by_podiums'] = podiums
        
        # Drivers by total points
        points = data['race_results'].groupby('driverId')['points'].sum().reset_index()
        points = points.sort_values('points', ascending=False)
        lists['drivers_by_points'] = points
        
        # Drivers by season (who raced each year)
        drivers_by_year = data['race_results'].groupby(['year', 'driverId']).size().reset_index(name='races')
        lists['drivers_by_year'] = drivers_by_year
    
    # =========================================================================
    # CONSTRUCTOR LISTS
    # =========================================================================
    
    if not data['race_results'].empty:
        # All constructors
        constructors = data['race_results'].groupby('constructorId').agg({
            'year': ['min', 'max'],
            'race': 'count'
        }).reset_index()
        constructors.columns = ['constructor_id', 'first_year', 'last_year', 'total_entries']
        constructors = constructors.sort_values('total_entries', ascending=False)
        lists['all_constructors'] = constructors
        
        # Constructors by wins
        df_copy = data['race_results'].copy()
        df_copy['pos'] = pd.to_numeric(df_copy['position'], errors='coerce')
        constructor_wins = df_copy[df_copy['pos'] == 1].groupby('constructorId').size().reset_index(name='wins')
        constructor_wins = constructor_wins.sort_values('wins', ascending=False)
        lists['constructors_by_wins'] = constructor_wins
        
        # Constructors by points
        constructor_points = data['race_results'].groupby('constructorId')['points'].sum().reset_index()
        constructor_points = constructor_points.sort_values('points', ascending=False)
        lists['constructors_by_points'] = constructor_points
    
    # =========================================================================
    # RACE LISTS
    # =========================================================================
    
    if not data['race_results'].empty:
        # All races
        races = data['race_results'].groupby(['year', 'race']).agg({
            'driverId': 'count'
        }).reset_index()
        races.columns = ['year', 'race', 'entries']
        races = races.sort_values(['year', 'race'])
        lists['all_races'] = races
        
        # Races per year
        races_per_year = data['race_results'].groupby('year')['race'].nunique().reset_index(name='races')
        lists['races_per_year'] = races_per_year
    
    # =========================================================================
    # CHAMPIONSHIP WINNERS
    # =========================================================================
    
    if not data['driver_standings'].empty:
        # Get final standings for each year
        final_standings = data['driver_standings'].groupby('year').apply(
            lambda x: x[x['race'] == x['race'].max()]
        ).reset_index(drop=True)
        
        # World champions
        champions = final_standings[final_standings['position'] == 1][['year', 'driverId', 'points']].copy()
        champions.columns = ['year', 'champion', 'points']
        lists['world_champions'] = champions
    
    if not data['constructor_standings'].empty:
        # Constructor champions
        final_constructor = data['constructor_standings'].groupby('year').apply(
            lambda x: x[x['race'] == x['race'].max()]
        ).reset_index(drop=True)
        
        constructor_champions = final_constructor[final_constructor['position'] == 1][['year', 'constructorId', 'points']].copy()
        constructor_champions.columns = ['year', 'champion', 'points']
        lists['constructor_champions'] = constructor_champions
    
    # =========================================================================
    # PERFORMANCE COMPARISONS
    # =========================================================================
    
    if not data['race_results'].empty:
        # Driver head-to-head (same team comparisons)
        h2h = data['race_results'].groupby(['year', 'constructorId', 'driverId']).agg({
            'points': 'sum',
            'race': 'count'
        }).reset_index()
        h2h.columns = ['year', 'constructor', 'driver', 'points', 'races']
        lists['driver_season_performance'] = h2h
    
    if not data['fastest_laps'].empty:
        # Fastest laps by driver
        fastest_laps = data['fastest_laps'].groupby('driverId').size().reset_index(name='fastest_laps')
        fastest_laps = fastest_laps.sort_values('fastest_laps', ascending=False)
        lists['drivers_by_fastest_laps'] = fastest_laps
    
    if not data['pit_stops'].empty:
        # Average pit stop time by constructor (recent years)
        pit_data = data['pit_stops'].copy()
        # Convert duration to numeric
        pit_data['duration_sec'] = pd.to_numeric(pit_data.get('duration', 0), errors='coerce')
        avg_pits = pit_data.groupby(['year', 'constructorId'])['duration_sec'].mean().reset_index()
        avg_pits.columns = ['year', 'constructor', 'avg_pit_duration']
        lists['avg_pit_stops_by_constructor'] = avg_pits
    
    return lists


def save_lists(lists):
    """Save all lists to CSV files."""
    for name, df in lists.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            output_path = OUTPUT_DIR / f"{name}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Saved: {name}.csv ({len(df)} rows)")


def create_comprehensive_summary():
    """Create a comprehensive summary DataFrame."""
    data = load_f1db_data()
    
    if data['race_results'].empty:
        return
    
    df = data['race_results'].copy()
    df['pos'] = pd.to_numeric(df['position'], errors='coerce')
    
    summary = df.groupby(['year', 'driverId', 'constructorId']).agg({
        'points': 'sum',
        'race': 'count',
        'pos': ['mean', 'min']
    }).reset_index()
    
    summary.columns = ['year', 'driver', 'constructor', 'total_points', 'races', 'avg_finish', 'best_finish']
    
    # Add wins and podiums
    wins = df[df['pos'] == 1].groupby(['year', 'driverId']).size().reset_index(name='wins')
    podiums = df[df['pos'] <= 3].groupby(['year', 'driverId']).size().reset_index(name='podiums')
    
    summary = summary.merge(wins, left_on=['year', 'driver'], right_on=['year', 'driverId'], how='left')
    summary = summary.merge(podiums, left_on=['year', 'driver'], right_on=['year', 'driverId'], how='left')
    
    summary['wins'] = summary['wins'].fillna(0).astype(int)
    summary['podiums'] = summary['podiums'].fillna(0).astype(int)
    
    # Clean up
    summary = summary.drop(columns=['driverId_x', 'driverId_y'], errors='ignore')
    summary = summary.sort_values(['year', 'total_points'], ascending=[True, False])
    
    # Save
    summary.to_csv(OUTPUT_DIR / 'driver_season_summary.csv', index=False)
    print(f"  Saved: driver_season_summary.csv ({len(summary)} rows)")
    
    return summary


def main():
    print("=" * 60)
    print("F1 DATA CONSOLIDATOR")
    print("=" * 60)
    
    print("\nCreating master lists...")
    lists = create_master_lists()
    save_lists(lists)
    
    print("\nCreating comprehensive summary...")
    create_comprehensive_summary()
    
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE!")
    print(f"Consolidated data saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print summary of created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
