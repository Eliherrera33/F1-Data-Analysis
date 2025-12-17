# -*- coding: utf-8 -*-
"""
F1 Data Downloader
Downloads F1 session data from 2018 to 2025 using FastF1.

Note: FastF1 v3.x provides telemetry data from 2018 onwards.
For 2012-2017, use the f1db or formula1-datasets repositories included in this folder.
"""

import fastf1
import os
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up cache directory for faster subsequent loads
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Data output directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_event_schedule(year):
    """Get the event schedule for a given year."""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        print(f"Error getting schedule for {year}: {e}")
        return None


def download_session_data(year, event, session_type='R'):
    """
    Download data for a specific session.
    
    Parameters:
    - year: int, the year (2018-2025)
    - event: str or int, event name or round number
    - session_type: str, one of 'FP1', 'FP2', 'FP3', 'Q', 'S' (Sprint), 'R' (Race)
    
    Returns:
    - session object with loaded data
    """
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"    Error loading {year} {event} {session_type}: {e}")
        return None


def download_all_races(start_year=2018, end_year=2025):
    """
    Download race data for all years from start_year to end_year.
    FastF1 supports 2018 onwards for telemetry data.
    """
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*50}")
        print(f"Processing Year: {year}")
        print(f"{'='*50}")
        
        schedule = get_event_schedule(year)
        if schedule is None:
            continue
        
        # Filter for conventional events (exclude testing)
        try:
            race_events = schedule[schedule['EventFormat'].isin(['conventional', 'sprint', 'sprint_shootout', 'sprint_qualifying'])]
        except:
            race_events = schedule
        
        for idx, event in race_events.iterrows():
            event_name = event.get('EventName', f'Round {idx}')
            round_num = event.get('RoundNumber', idx)
            
            print(f"\n  [{round_num}] Processing: {event_name}")
            
            # Try to load race session
            session = download_session_data(year, round_num, 'R')
            if session is not None:
                try:
                    lap_count = len(session.laps)
                    print(f"    [OK] Race data loaded: {lap_count} laps")
                    
                    # Save lap data to CSV
                    year_dir = DATA_DIR / str(year)
                    year_dir.mkdir(exist_ok=True)
                    
                    safe_event_name = "".join(c for c in event_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_event_name = safe_event_name.replace(' ', '_')
                    
                    csv_path = year_dir / f"{safe_event_name}_laps.csv"
                    session.laps.to_csv(csv_path, index=False)
                    print(f"    [OK] Saved to {csv_path.name}")
                    
                    # Also save results
                    try:
                        results_path = year_dir / f"{safe_event_name}_results.csv"
                        session.results.to_csv(results_path, index=False)
                        print(f"    [OK] Saved results to {results_path.name}")
                    except Exception as e:
                        print(f"    [WARN] Could not save results: {e}")
                        
                except Exception as e:
                    print(f"    [ERROR] Failed to process: {e}")


def download_single_race(year, event, include_telemetry=False):
    """
    Download a single race with optional telemetry data.
    
    Example:
        download_single_race(2023, 'Monaco', include_telemetry=True)
    """
    session = download_session_data(year, event, 'R')
    
    if session is None:
        return None
    
    print(f"Session: {session.event['EventName']} {year}")
    print(f"Total Laps: {len(session.laps)}")
    
    # Get race results
    print("\nRace Results:")
    results = session.results
    if 'Position' in results.columns:
        display_cols = [c for c in ['Position', 'Abbreviation', 'TeamName', 'Status'] if c in results.columns]
        print(results[display_cols].head(10).to_string())
    
    if include_telemetry:
        # Get fastest lap telemetry
        fastest_lap = session.laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry()
        print(f"\nFastest Lap Telemetry Points: {len(telemetry)}")
    
    return session


if __name__ == "__main__":
    print("=" * 60)
    print("F1 DATA DOWNLOADER")
    print("=" * 60)
    print("\nThis script uses FastF1 to download F1 data.")
    print("\nAvailable data:")
    print("  - FastF1 (2018-2025): Full telemetry, lap times, car data")
    print("  - f1db repo: Historical data from 1950 onwards")
    print("  - formula1-datasets repo: 2019-2025 CSVs with analysis notebooks")
    
    # Download all races from 2018-2025
    print("\n" + "="*60)
    print("DOWNLOADING ALL F1 TELEMETRY DATA FROM 2018-2025")
    print("="*60)
    download_all_races(2018, 2025)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nData locations:")
    print(f"  FastF1 Cache: {CACHE_DIR}")
    print(f"  CSV Exports: {DATA_DIR}")
    print(f"  f1db Database: ./f1db")
    print(f"  Formula1 Datasets: ./formula1-datasets")
