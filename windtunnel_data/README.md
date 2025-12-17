# F1 Wind Tunnel & CFD Aerodynamic Data

## Overview

This folder contains compiled research and data from publicly available F1 aerodynamic studies. 
The primary source is the **PERRINN 2017 F1 Car** - the only open-source F1 car CAD model available for CFD analysis.

> **Note**: Actual F1 team wind tunnel data is proprietary and never released publicly. The data here comes from academic research and open-source CFD simulations.

---

## PERRINN 2017 F1 Car

### About PERRINN
PERRINN (Platform for Electrification and Renewable Racing Innovation) released their 2017 F1 car design as an open-source project, allowing engineers and researchers to download, modify, and perform CFD analysis.

### CAD Model Sources
- **Official Project**: [PERRINN on F1Technical.net](https://www.f1technical.net/forum/viewtopic.php?t=22689)
- **SimScale**: Cloud CFD platform with PERRINN model available
- **GrabCAD**: 3D model downloads

---

## CFD Simulation Results

### Published Aerodynamic Coefficients

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| Frontal Area | S | ~1.5 | m² | Reference area for coefficients |
| Drag Coefficient × Area | sCx | 1.15 - 1.17 | m² | At 50mm rear ride height |
| Downforce Coefficient × Area | sCz | 3.23 - 3.27 | m² | At 50mm rear ride height |
| Aerodynamic Efficiency | L/D | ~2.8 | - | Downforce/Drag ratio |

### Ride Height Study Results

| Front RH (mm) | Rear RH (mm) | sCx (m²) | sCz (m²) | Efficiency |
|---------------|--------------|----------|----------|------------|
| 30 | 50 | 1.15 | 3.23 | 2.81 |
| 40 | 50 | 1.16 | 3.25 | 2.80 |
| 50 | 50 | 1.17 | 3.27 | 2.79 |

*Source: CAE Conference proceedings, OpenFOAM simulations*

### Component Contribution to Downforce

| Component | % of Total Downforce |
|-----------|---------------------|
| Underbody/Floor | ~40-45% |
| Rear Wing | ~25-30% |
| Front Wing | ~20-25% |
| Diffuser | ~10-15% |
| Bargeboard Area | ~5-10% |

### Component Contribution to Drag

| Component | % of Total Drag |
|-----------|-----------------|
| Rear Wing (induced) | ~30-35% |
| Front Wheels | ~15-20% |
| Rear Wheels | ~15-20% |
| Body/Sidepods | ~10-15% |
| Front Wing | ~10-12% |
| Other | ~10-15% |

---

## CFD Methodology

### Software Used
- **OpenFOAM** - Primary open-source CFD solver
- **snappyHexMesh** - Mesh generation utility
- **ParaView** - Post-processing visualization

### Turbulence Models

| Model | Drag Accuracy | Downforce Accuracy | Notes |
|-------|---------------|-------------------|-------|
| k-ω SST | -11% | -20% | Predicts premature separation |
| Spalart-Allmaras (S-A) | Good | Good | Best agreement with reference |
| DDES | Good | Good | Detached Eddy Simulation |

### Simulation Parameters
- **Mesh Size**: ~40 million cells (half-car)
- **Inlet Velocity**: 50 m/s (180 km/h typical)
- **Ground**: Moving wall (simulates track motion)
- **Wheels**: Rotating with MRF (Multiple Reference Frame)

---

## Typical F1 Aerodynamic Values (2017-2025)

These are estimated ranges based on published research and industry knowledge:

| Parameter | Low Downforce (Monza) | Medium | High Downforce (Monaco) |
|-----------|----------------------|--------|------------------------|
| Downforce @ 250 km/h | ~15,000 N | ~22,000 N | ~30,000+ N |
| Drag @ 250 km/h | ~8,000 N | ~10,000 N | ~14,000 N |
| L/D Ratio | 2.0-2.5 | 2.5-3.0 | 2.0-2.5 |
| Top Speed | ~370 km/h | ~340 km/h | ~300 km/h |

---

## Research Papers and Sources

### Academic Publications

1. **"CFD Analysis of a 2017 F1 Car Using OpenFOAM"**
   - Source: David Publisher
   - Key findings: k-ω SST model underestimates downforce by ~20%

2. **"External Aerodynamics Investigation of a Modern Formula One Car"**
   - Source: University of Bergamo
   - Focus: Workflow development with OpenFOAM

3. **"Ride Height Sensitivity Study"**
   - Source: CAE Conference
   - Key findings: sCx/sCz variations with ride height

### Online Resources

- **luke.cfd** - OpenFOAM F1 tutorials
- **SimScale** - Cloud CFD F1 examples
- **CFD-Online Forum** - PERRINN discussions

---

## Limitations of Public Data

| What's Available | What's NOT Available |
|------------------|---------------------|
| PERRINN 2017 geometry | Current 2024/2025 team data |
| Academic CFD results | Actual wind tunnel measurements |
| Estimated coefficients | Proprietary team correlations |
| General trends | DRS effect quantification |
| Open-source tools | Real-time aero adjustments |

---

## How to Run Your Own CFD

### Prerequisites
- OpenFOAM (v2106 or later recommended)
- 32GB+ RAM
- Multi-core CPU (16+ cores recommended)

### Basic Workflow
1. Download PERRINN CAD model
2. Clean/repair geometry in CAD software
3. Generate mesh using snappyHexMesh
4. Set boundary conditions (inlet, outlet, ground, wheels)
5. Run simulation (8-24 hours typical)
6. Post-process with ParaView

### Example Commands
```bash
# Mesh generation
blockMesh
snappyHexMesh -overwrite

# Run solver
simpleFoam

# Calculate forces
postProcess -func forces
```

---

## ATR (Aerodynamic Testing Restrictions) - 2024

The FIA limits wind tunnel time based on constructor standings:

| Position | % of Baseline | Approx Hours/Year |
|----------|--------------|-------------------|
| 1st | 70% | ~224 hours |
| 2nd | 75% | ~240 hours |
| 3rd | 80% | ~256 hours |
| 4th | 85% | ~272 hours |
| 5th | 90% | ~288 hours |
| 6th | 95% | ~304 hours |
| 7th | 100% | ~320 hours |
| 8th | 105% | ~336 hours |
| 9th | 110% | ~352 hours |
| 10th | 115% | ~368 hours |

*Baseline: 320 hours wind tunnel time per year*

---

## File Index

| File | Description |
|------|-------------|
| `README.md` | This documentation |
| `perrinn_cfd_data.csv` | Compiled CFD results data |
| `atr_allocations.csv` | ATR wind tunnel hour allocations |

---

*Last Updated: December 2024*
*Compiled from publicly available research and open-source projects*
