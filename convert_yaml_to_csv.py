# -*- coding: utf-8 -*-
"""
F1DB YAML to CSV Converter
Converts f1db YAML data (2012-2017) to CSV format for easier analysis.
"""

import yaml
import csv
import os
from pathlib import Path

# Directories
F1DB_DIR = Path(__file__).parent / "f1db" / "src" / "data" / "seasons"
OUTPUT_DIR = Path(__file__).parent / "data" / "f1db_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(file_path):
    """Load a YAML file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def convert_race_results(year, race_dir, race_name):
    """Convert race results YAML to list of dicts."""
    results_file = race_dir / "race-results.yml"
    if not results_file.exists():
        return []
    
    data = load_yaml(results_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def convert_qualifying_results(year, race_dir, race_name):
    """Convert qualifying results YAML to list of dicts."""
    qual_file = race_dir / "qualifying-results.yml"
    if not qual_file.exists():
        return []
    
    data = load_yaml(qual_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def convert_fastest_laps(year, race_dir, race_name):
    """Convert fastest laps YAML to list of dicts."""
    laps_file = race_dir / "fastest-laps.yml"
    if not laps_file.exists():
        return []
    
    data = load_yaml(laps_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def convert_pit_stops(year, race_dir, race_name):
    """Convert pit stops YAML to list of dicts."""
    pits_file = race_dir / "pit-stops.yml"
    if not pits_file.exists():
        return []
    
    data = load_yaml(pits_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def convert_driver_standings(year, race_dir, race_name):
    """Convert driver standings YAML to list of dicts."""
    standings_file = race_dir / "driver-standings.yml"
    if not standings_file.exists():
        return []
    
    data = load_yaml(standings_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def convert_constructor_standings(year, race_dir, race_name):
    """Convert constructor standings YAML to list of dicts."""
    standings_file = race_dir / "constructor-standings.yml"
    if not standings_file.exists():
        return []
    
    data = load_yaml(standings_file)
    if not data:
        return []
    
    results = []
    for entry in data:
        entry['year'] = year
        entry['race'] = race_name
        results.append(entry)
    return results


def save_to_csv(data, filename, fieldnames=None):
    """Save list of dicts to CSV."""
    if not data:
        print(f"  No data for {filename}")
        return
    
    if fieldnames is None:
        # Get all unique keys from all entries
        fieldnames = set()
        for entry in data:
            fieldnames.update(entry.keys())
        fieldnames = sorted(list(fieldnames))
    
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"  Saved {len(data)} rows to {filename}")


def convert_year(year):
    """Convert all data for a given year."""
    print(f"\nConverting {year}...")
    
    year_dir = F1DB_DIR / str(year)
    if not year_dir.exists():
        print(f"  Year directory not found: {year_dir}")
        return
    
    races_dir = year_dir / "races"
    if not races_dir.exists():
        print(f"  Races directory not found")
        return
    
    all_race_results = []
    all_qualifying = []
    all_fastest_laps = []
    all_pit_stops = []
    all_driver_standings = []
    all_constructor_standings = []
    
    # Process each race
    for race_dir in sorted(races_dir.iterdir()):
        if not race_dir.is_dir():
            continue
        
        race_name = race_dir.name
        
        all_race_results.extend(convert_race_results(year, race_dir, race_name))
        all_qualifying.extend(convert_qualifying_results(year, race_dir, race_name))
        all_fastest_laps.extend(convert_fastest_laps(year, race_dir, race_name))
        all_pit_stops.extend(convert_pit_stops(year, race_dir, race_name))
        all_driver_standings.extend(convert_driver_standings(year, race_dir, race_name))
        all_constructor_standings.extend(convert_constructor_standings(year, race_dir, race_name))
    
    # Save to CSVs
    save_to_csv(all_race_results, f"{year}_race_results.csv")
    save_to_csv(all_qualifying, f"{year}_qualifying.csv")
    save_to_csv(all_fastest_laps, f"{year}_fastest_laps.csv")
    save_to_csv(all_pit_stops, f"{year}_pit_stops.csv")
    save_to_csv(all_driver_standings, f"{year}_driver_standings.csv")
    save_to_csv(all_constructor_standings, f"{year}_constructor_standings.csv")


def convert_all_years(start_year=2012, end_year=2025):
    """Convert all years in the specified range."""
    print("=" * 60)
    print("F1DB YAML TO CSV CONVERTER")
    print("=" * 60)
    
    for year in range(start_year, end_year + 1):
        convert_year(year)
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print(f"CSV files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    convert_all_years(2012, 2025)
