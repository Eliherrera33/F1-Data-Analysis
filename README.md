<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,5,30&height=300&section=header&text=F1%20DATA%20ANALYTICS&fontSize=60&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Pushing%20the%20Limits%20of%20Motorsport%20Data%20Science&descAlignY=55&descAlign=50" width="100%"/>

<!-- Animated F1 Car -->
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="500">

<!-- Typing Animation -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=600&size=24&duration=3000&pause=1000&color=E10600&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=100&lines=🏎️+Telemetry+Analysis+%7C+G-Force+Mapping+%7C+Aerodynamics;📊+50%2B+Visualizations+%7C+12+Analysis+Scripts;⚡+Real-time+Data+%7C+Physics-based+Calculations)](https://git.io/typing-svg)

<!-- Badges -->
<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastF1-E10600?style=for-the-badge&logo=f1&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
</p>

<!-- Live Demo Button -->
<a href="https://eliherrera33.github.io/F1-Data-Analysis/">
<img src="https://img.shields.io/badge/🏁_LIVE_DEMO-View_Dashboard-E10600?style=for-the-badge&logoColor=white"/>
</a>

</div>

---

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🏎️ About This Project

> **A comprehensive Formula 1 data analysis suite** that processes telemetry, calculates G-forces, analyzes aerodynamics, and visualizes race strategy. Built for those who want to understand the science behind the fastest motorsport on Earth.

<table>
<tr>
<td width="50%">

### 📊 Key Features
- **Real-time Telemetry Analysis** - Speed traces, throttle/brake, gear shifts
- **G-Force Calculations** - Longitudinal & lateral G with driver-style gauges
- **Aerodynamic Modeling** - CFD-based downforce/drag calculations
- **Tire Strategy Analysis** - Compound degradation & pit stop optimization
- **Engine/Power Unit Data** - 1.6L V6 Turbo Hybrid specifications
- **Multi-Track Comparison** - Monaco vs Spa vs Monza analysis

</td>
<td width="50%">

### 📈 By The Numbers
| Metric | Value |
|--------|-------|
| 🎨 Visualizations | **50+** |
| 🐍 Python Scripts | **12** |
| 🏁 Tracks Analyzed | **3** |
| 👨‍✈️ Drivers Compared | **20** |
| 🔧 Engine Records | **419** |
| ⚡ Data Points | **1M+** |

</td>
</tr>
</table>

---

## 🎯 Analysis Modules

<details>
<summary><b>🔬 TELEMETRY ANALYSIS</b> - Click to expand</summary>
<br>

Real-time car data processing including speed traces, throttle/brake application, DRS usage, and driver comparison.

| Visualization | Description |
|--------------|-------------|
| 🗺️ Track Maps | GPS-based circuit visualization with speed coloring |
| 📈 Speed Traces | Lap-by-lap speed comparison |
| 🎮 Driver Inputs | Throttle, brake, and gear shift patterns |
| 🏎️ Racing Lines | Corner-by-corner trajectory analysis |

```python
# Example: Load Monaco GP telemetry
import fastf1
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()
lap = session.laps.pick_fastest()
tel = lap.get_telemetry()
```

</details>

<details>
<summary><b>💪 G-FORCE MAPPING</b> - Click to expand</summary>
<br>

Calculate and visualize the extreme forces drivers experience - up to **6G** in corners and under braking!

| G-Force Type | Max Value | Description |
|--------------|-----------|-------------|
| 🔴 Braking | **5.5G** | Deceleration force |
| 🟢 Acceleration | **1.5G** | Acceleration force |
| 🟡 Lateral | **5.8G** | Cornering force |

**Physics Behind It:**
```
Longitudinal G = (Δv / Δt) / 9.81
Lateral G = (v² × κ) / 9.81
Total G = √(G_long² + G_lat²)
```

Features F1 TV-style circular and diamond G-force gauges!

</details>

<details>
<summary><b>🌪️ AERODYNAMICS</b> - Click to expand</summary>
<br>

Wind tunnel data analysis using PERRINN F1 CFD coefficients.

| Setup | sCx (Drag) | sCz (Downforce) | L/D Ratio |
|-------|------------|-----------------|-----------|
| 🔴 Monaco (High DF) | 1.35 | 4.0 | 2.96 |
| 🟡 Silverstone (Med) | 1.16 | 3.25 | 2.80 |
| 🟢 Monza (Low DF) | 0.95 | 2.5 | 2.63 |

**Key Insight:** At 250 km/h, downforce exceeds car weight - an F1 car could theoretically drive upside down! 🤯

</details>

<details>
<summary><b>🛞 TIRE STRATEGY</b> - Click to expand</summary>
<br>

Compound analysis, degradation modeling, and pit stop optimization.

| Compound | Grip | Durability | Ideal For |
|----------|------|------------|-----------|
| 🔴 SOFT | ⭐⭐⭐⭐⭐ | ⭐⭐ | Qualifying, short stints |
| 🟡 MEDIUM | ⭐⭐⭐⭐ | ⭐⭐⭐ | Race balance |
| ⚪ HARD | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Long stints, hot conditions |

</details>

<details>
<summary><b>⚡ POWER UNITS</b> - Click to expand</summary>
<br>

Current F1 Power Unit specifications (2014-present era):

| Component | Specification |
|-----------|--------------|
| Configuration | 1.6L V6 Turbo Hybrid |
| Total Power | ~1,000 HP |
| ICE Output | ~550 HP |
| MGU-K | 120 kW (160 HP) |
| Thermal Efficiency | 50%+ |
| RPM Limit | 15,000 |

</details>

---

## 🖼️ Visualization Gallery

<div align="center">

### Telemetry & Track Analysis
<img src="telemetry_visualizations/2024_Monaco_Grand_Prix_track_map.png" width="45%"/>
<img src="telemetry_visualizations/2024_Monaco_Grand_Prix_speed_trace.png" width="45%"/>

### G-Force Gauges
<img src="gforce_visualizations/gforce_circular_meter.png" width="45%"/>
<img src="gforce_visualizations/gforce_diamond_cross.png" width="45%"/>

### Aerodynamics
<img src="windtunnel_data/visualizations/aero_calculator_plots.png" width="45%"/>
<img src="windtunnel_data/visualizations/atr_wind_tunnel_hours.png" width="45%"/>

### Strategy & Engines
<img src="tire_pitstop_visualizations/2024_Monaco_Grand_Prix_tire_strategy.png" width="45%"/>
<img src="engine_visualizations/engine_evolution.png" width="45%"/>

</div>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Eliherrera33/F1-Data-Analysis.git
cd F1-Data-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install fastf1 matplotlib numpy pandas scipy seaborn

# Run any analysis script
python telemetry_analysis.py
python gforce_analysis.py
python aero_calculator.py
```

---

## 📁 Project Structure

```
F1-Data-Analysis/
├── 🌐 index.html                    # Portfolio website
├── 🎨 styles.css                    # Website styling
├── ⚡ script.js                     # Interactive features
│
├── 🔬 Analysis Scripts
│   ├── f1_analysis.py               # Core F1 data analysis
│   ├── telemetry_analysis.py        # Telemetry processing
│   ├── gforce_analysis.py           # G-force calculations
│   ├── gforce_driver_comparison.py  # Driver G-force comparison
│   ├── car_aero_analysis.py         # DRS & aero analysis
│   ├── aero_calculator.py           # Downforce/drag calculator
│   ├── atr_visualization.py         # Wind tunnel restrictions
│   ├── downforce_vs_speed.py        # Speed-based aero analysis
│   ├── estimated_aero_analysis.py   # Telemetry-based aero inference
│   ├── tire_pitstop_analysis.py     # Tire & pit strategy
│   ├── engine_analysis.py           # Power unit analysis
│   └── circuit_analysis.py          # Track comparison
│
├── 📊 Visualizations
│   ├── telemetry_visualizations/    # Track maps, speed traces
│   ├── gforce_visualizations/       # G-force gauges (22 files)
│   ├── engine_visualizations/       # Power unit charts
│   ├── tire_pitstop_visualizations/ # Strategy charts
│   ├── car_aero_visualizations/     # DRS, chassis
│   └── windtunnel_data/             # CFD data & ATR
│
└── 📚 Data
    └── data/reference/              # Historical F1 data
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data** | ![FastF1](https://img.shields.io/badge/FastF1-E10600?style=flat-square) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square) |
| **Science** | ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |
| **Web** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) |

</div>

---

## 📫 Connect

<div align="center">

[![Portfolio](https://img.shields.io/badge/🏁_Portfolio-View_Dashboard-E10600?style=for-the-badge)](https://eliherrera33.github.io/F1-Data-Analysis/)
[![GitHub](https://img.shields.io/badge/GitHub-Eliherrera33-181717?style=for-the-badge&logo=github)](https://github.com/Eliherrera33)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/)

</div>

---

<div align="center">

### ⭐ Star this repo if you found it useful!

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,5,30&height=120&section=footer&animation=fadeIn" width="100%"/>

<sub>Built with 🏎️ and Python | Data sourced via FastF1 API</sub>

</div>
