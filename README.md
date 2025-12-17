# F1 Data Analysis Repository

A comprehensive Formula 1 data repository covering 1950-2025 with detailed telemetry, race results, driver/constructor statistics, and visualizations.

## Data Sources

| Source | Years | Content |
|--------|-------|---------|
| **f1db** | 1950-2025 | Complete historical F1 database (race results, qualifying, standings, pit stops) |
| **formula1-datasets** | 2013-2025 | Clean CSVs with race results, qualifying, sprint races, calendars |
| **FastF1** | 2018-2025 | Detailed telemetry data (speed, throttle, brake, gear, DRS, position) |

## Quick Start

```powershell
# Activate virtual environment
cd "e:\Repose E\F1 Data"
.\venv\Scripts\activate

# Run any analysis script
python f1_analysis.py           # Driver/constructor performance charts
python telemetry_analysis.py    # Track maps, throttle/brake traces, speed data
python circuit_analysis.py      # Circuit comparisons
python consolidate_data.py      # Generate consolidated data lists
```

## Scripts

| Script | Description |
|--------|-------------|
| `f1_analysis.py` | Driver/constructor wins, podiums, points progression, heatmaps |
| `telemetry_analysis.py` | Track maps, speed traces, throttle/brake usage, driver comparisons |
| `circuit_analysis.py` | Circuit length/turns comparisons, world map, street vs permanent |
| `tire_pitstop_analysis.py` | Tire compounds, pit stop times, strategies, degradation analysis |
| `car_aero_analysis.py` | DRS usage, car speed profiles, chassis data, downforce comparison |
| `gforce_analysis.py` | G-force calculation and visualization from telemetry |
| `engine_analysis.py` | Engine specifications, manufacturers, power unit evolution |
| `consolidate_data.py` | Create unified data lists (winners, points, fastest laps) |
| `convert_yaml_to_csv.py` | Convert f1db YAML to CSV format |
| `convert_reference_data.py` | Convert circuits, drivers, constructors to CSV |
| `download_f1_data.py` | Download race telemetry using FastF1 |

## Visualizations

### Performance Charts (`visualizations/`)
- Driver wins bar chart
- Constructor wins bar chart
- Podium finishes
- Points progression (line chart)
- Wins heatmap by year
- Championship dominance
- DNF rates
- Individual driver position distributions

### Telemetry Charts (`telemetry_visualizations/`)
- **Track Maps**: Colored by speed, gear
- **Speed Traces**: Speed over lap distance
- **Throttle/Brake**: Input traces over lap
- **RPM & Gear**: Engine data traces
- **DRS Usage**: DRS zones highlighted
- **Weather**: Air/track temperature, humidity, wind
- **Driver Comparisons**: Side-by-side telemetry
- **Racing Lines**: Track position comparison
- **Tire Strategy**: Compound usage per driver
- **Lap Evolution**: Lap times through race

### Circuit Charts (`circuit_visualizations/`)
- Circuit lengths comparison
- Number of turns
- Circuit types (street, permanent, road)
- Circuits by country
- Clockwise vs anti-clockwise
- Length vs turns scatter plot
- World map of circuit locations
- Street vs permanent comparison

### Tire & Pit Stop Charts (`tire_pitstop_visualizations/`)
- **Tire Strategy**: Compound usage per driver per lap (race visualization)
- **Tire Degradation**: Lap time vs tire age scatter plot
- **Compound Usage**: Pie chart of total laps per compound
- **Stint Analysis**: Visual breakdown of stint lengths per driver
- **Average Stint Length**: Bar chart per compound
- **Fastest Pit Stops**: Fastest pit stop per team
- **Pit Stop Distribution**: Box plot of pit stop times per team
- **Pit Stops per Race**: Total pit stops per race in season

### Car & Aero Charts (`car_aero_visualizations/`)
- **DRS Zones**: Track map showing DRS activation zones
- **DRS Usage by Driver**: Percentage of time each driver uses DRS
- **DRS Speed Analysis**: Speed comparison DRS open vs closed
- **Downforce Comparison**: Monaco (high DF) vs Monza (low DF) speed profiles
- **Speed Trace Comparison**: Normalized lap overlay between tracks
- **Cornering Speed**: Average corner speeds per driver (downforce indicator)
- **Chassis by Team**: Historical chassis models per constructor
- **Chassis Timeline**: Evolution of chassis per top team

### G-Force Charts (`gforce_visualizations/`)
- **G-Force Trace**: Longitudinal (braking/accel) and lateral (cornering) G over lap distance
- **G-Force Histogram**: Distribution of G-forces experienced during fastest lap
- **G-Force Comparison**: Comparison of max G-forces across all drivers
- **G-Force Map**: Track map colored by G-force intensity (braking, lateral, total)
- **Track Comparison**: Cross-track comparison of G-forces (Monaco vs Spa vs Monza)

### Engine & Power Unit Charts (`engine_visualizations/`)
- **Engine Configurations**: Pie chart of all F1 engine configurations (V6, V8, V10, V12, etc.)
- **Aspiration Types**: Bar chart comparing NA, Turbo, Turbo Hybrid, Supercharged
- **Engine Capacity**: Histogram of engine displacements through F1 history
- **Engines by Manufacturer**: Top 20 manufacturers by number of engine models
- **Manufacturer Countries**: Pie chart of engine manufacturers by country
- **Hybrid Era Engines**: Analysis of 2014-2025 turbo hybrid power units
- **Current PU Specs**: Technical specifications infographic for current era
- **Engine Evolution**: Timeline of F1 engine eras from 1950-2025
- **Config by Era**: Stacked bar chart of configurations per capacity era
- **V-Engine Comparison**: Detailed comparison of V6, V8, V10, V12 engines

## Data Files

### Consolidated Lists (`data/consolidated/`)
- `world_champions.csv` - WDC winners 2012-2025
- `constructor_champions.csv` - WCC winners
- `drivers_by_wins.csv` - Career wins ranking
- `drivers_by_podiums.csv` - Career podiums ranking
- `drivers_by_points.csv` - Career points ranking
- `constructors_by_wins.csv` - Team wins ranking
- `all_drivers.csv` - Driver career spans
- `all_races.csv` - All race entries

### Reference Data (`data/reference/`)
- `circuits.csv` - 77 circuits with location, length, turns
- `drivers.csv` - 915 drivers with nationalities, DOB
- `constructors.csv` - 185 constructors
- `engines.csv` - 419 engine specifications
- `chassis.csv` - 1138 chassis models

### Season Data (`data/f1db_csv/`)
- `{year}_race_results.csv` - Complete race results
- `{year}_qualifying.csv` - Qualifying results
- `{year}_fastest_laps.csv` - Fastest lap data
- `{year}_pit_stops.csv` - Pit stop times
- `{year}_driver_standings.csv` - Championship standings
- `{year}_constructor_standings.csv` - Constructor standings

## Key Statistics (2012-2025)

| Metric | Value |
|--------|-------|
| Total Races | 291 |
| Unique Drivers | 73 |
| Unique Constructors | 23 |
| Most Driver Wins | Lewis Hamilton (88) |
| Most Constructor Wins | Mercedes (122) |

## Telemetry Data Available

For 2018-2025 races (via FastF1):
- **Car Telemetry**: Speed, RPM, Gear, Throttle, Brake, DRS
- **Position Data**: X, Y, Z coordinates (track position)
- **Timing**: Lap times, sector times, gap data
- **Weather**: Air temp, track temp, humidity, wind speed/direction
- **Race Control**: Flags, penalties, safety car periods

## Installation

The virtual environment includes:
- FastF1 (telemetry access)
- Pandas (data manipulation)
- Matplotlib (static charts)
- Seaborn (statistical visualizations)
- Plotly (interactive charts)
- PyYAML (data conversion)

Re-install dependencies:
```powershell
pip install -r requirements.txt
```

## Adding More Data

### Download telemetry for a specific race:
```python
import fastf1

session = fastf1.get_session(2024, 'Silverstone', 'R')
session.load()

# Access telemetry for a driver
lap = session.laps.pick_driver('VER').pick_fastest()
telemetry = lap.get_telemetry()
```

### Analyze a different race:
Edit `telemetry_analysis.py` and change:
```python
analyze_session(2024, 'Monaco', 'R')  # Change to any race
```

## License

Data sourced from f1db (open source), formula1-datasets (GitHub), and FastF1 (official F1 timing data).
