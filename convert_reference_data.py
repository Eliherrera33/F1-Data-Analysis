# -*- coding: utf-8 -*-
"""
F1 Additional Data Converter
Converts all remaining f1db data (circuits, drivers, constructors, etc.) to CSV format.
"""

import yaml
import csv
from pathlib import Path

# Directories
F1DB_DIR = Path(__file__).parent / "f1db" / "src" / "data"
OUTPUT_DIR = Path(__file__).parent / "data" / "reference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(file_path):
    """Load a YAML file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def convert_circuits():
    """Convert all circuit data to CSV."""
    circuits_dir = F1DB_DIR / "circuits"
    all_circuits = []
    
    for yml_file in sorted(circuits_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_circuits.append(data)
    
    if all_circuits:
        # Get all unique keys
        fieldnames = set()
        for circuit in all_circuits:
            fieldnames.update(circuit.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "circuits.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for circuit in all_circuits:
                writer.writerow(circuit)
        
        print(f"  Saved: circuits.csv ({len(all_circuits)} circuits)")


def convert_drivers():
    """Convert all driver data to CSV."""
    drivers_dir = F1DB_DIR / "drivers"
    all_drivers = []
    
    for yml_file in sorted(drivers_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            # Flatten nested data
            flat_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_data[f"{key}_{subkey}"] = subvalue
                elif isinstance(value, list):
                    flat_data[key] = ', '.join(str(v) for v in value)
                else:
                    flat_data[key] = value
            all_drivers.append(flat_data)
    
    if all_drivers:
        fieldnames = set()
        for driver in all_drivers:
            fieldnames.update(driver.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "drivers.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for driver in all_drivers:
                writer.writerow(driver)
        
        print(f"  Saved: drivers.csv ({len(all_drivers)} drivers)")


def convert_constructors():
    """Convert all constructor data to CSV."""
    constructors_dir = F1DB_DIR / "constructors"
    all_constructors = []
    
    for yml_file in sorted(constructors_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            flat_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_data[f"{key}_{subkey}"] = subvalue
                elif isinstance(value, list):
                    flat_data[key] = ', '.join(str(v) for v in value)
                else:
                    flat_data[key] = value
            all_constructors.append(flat_data)
    
    if all_constructors:
        fieldnames = set()
        for constructor in all_constructors:
            fieldnames.update(constructor.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "constructors.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for constructor in all_constructors:
                writer.writerow(constructor)
        
        print(f"  Saved: constructors.csv ({len(all_constructors)} constructors)")


def convert_engines():
    """Convert engine data to CSV."""
    engines_dir = F1DB_DIR / "engines"
    all_engines = []
    
    for yml_file in sorted(engines_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_engines.append(data)
    
    if all_engines:
        fieldnames = set()
        for engine in all_engines:
            fieldnames.update(engine.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "engines.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for engine in all_engines:
                writer.writerow(engine)
        
        print(f"  Saved: engines.csv ({len(all_engines)} engines)")


def convert_engine_manufacturers():
    """Convert engine manufacturer data to CSV."""
    mfg_dir = F1DB_DIR / "engine-manufacturers"
    all_mfgs = []
    
    for yml_file in sorted(mfg_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_mfgs.append(data)
    
    if all_mfgs:
        fieldnames = set()
        for mfg in all_mfgs:
            fieldnames.update(mfg.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "engine_manufacturers.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for mfg in all_mfgs:
                writer.writerow(mfg)
        
        print(f"  Saved: engine_manufacturers.csv ({len(all_mfgs)} manufacturers)")


def convert_tyre_manufacturers():
    """Convert tyre manufacturer data to CSV."""
    tyre_dir = F1DB_DIR / "tyre-manufacturers"
    all_tyres = []
    
    for yml_file in sorted(tyre_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_tyres.append(data)
    
    if all_tyres:
        fieldnames = set()
        for tyre in all_tyres:
            fieldnames.update(tyre.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "tyre_manufacturers.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tyre in all_tyres:
                writer.writerow(tyre)
        
        print(f"  Saved: tyre_manufacturers.csv ({len(all_tyres)} manufacturers)")


def convert_grands_prix():
    """Convert grands prix data to CSV."""
    gp_dir = F1DB_DIR / "grands-prix"
    all_gps = []
    
    for yml_file in sorted(gp_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_gps.append(data)
    
    if all_gps:
        fieldnames = set()
        for gp in all_gps:
            fieldnames.update(gp.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "grands_prix.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gp in all_gps:
                writer.writerow(gp)
        
        print(f"  Saved: grands_prix.csv ({len(all_gps)} Grands Prix)")


def convert_countries():
    """Convert country data to CSV."""
    countries_dir = F1DB_DIR / "countries"
    all_countries = []
    
    for yml_file in sorted(countries_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_countries.append(data)
    
    if all_countries:
        fieldnames = set()
        for country in all_countries:
            fieldnames.update(country.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "countries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for country in all_countries:
                writer.writerow(country)
        
        print(f"  Saved: countries.csv ({len(all_countries)} countries)")


def convert_chassis():
    """Convert chassis data to CSV."""
    chassis_dir = F1DB_DIR / "chassis"
    all_chassis = []
    
    for yml_file in sorted(chassis_dir.glob("*.yml")):
        data = load_yaml(yml_file)
        if data:
            all_chassis.append(data)
    
    if all_chassis:
        fieldnames = set()
        for chassis in all_chassis:
            fieldnames.update(chassis.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_DIR / "chassis.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for chassis in all_chassis:
                writer.writerow(chassis)
        
        print(f"  Saved: chassis.csv ({len(all_chassis)} chassis)")


def main():
    print("=" * 60)
    print("F1 ADDITIONAL DATA CONVERTER")
    print("=" * 60)
    
    print("\nConverting reference data...")
    convert_circuits()
    convert_drivers()
    convert_constructors()
    convert_engines()
    convert_engine_manufacturers()
    convert_tyre_manufacturers()
    convert_grands_prix()
    convert_countries()
    convert_chassis()
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print(f"Reference data saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
