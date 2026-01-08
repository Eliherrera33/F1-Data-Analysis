# F1 Wind Tunnel / CFD Simulation Resources

## Open Source F1 CFD Workflow Guide

This guide covers setting up a wind tunnel CFD simulation for an F1 car using open-source tools.

---

## 1. Required Software

### Core CFD Solver
| Software | Description | Install |
|----------|-------------|---------|
| **OpenFOAM** | Industry-standard open-source CFD | `sudo apt install openfoam` (Linux) or WSL on Windows |
| **ParaView** | Post-processing visualization | `sudo apt install paraview` |
| **FreeCAD** | 3D CAD modeling | [freecad.org](https://freecad.org) |
| **Blender** | STL editing and geometry cleanup | [blender.org](https://blender.org) |

### Alternative Solvers
- **SU2** - Stanford University solver (good for optimization)
- **SimScale** - Cloud-based CFD (free tier available)
- **AirShaper** - Automated CFD built on OpenFOAM

---

## 2. F1 Car 3D Models (Free/Open Source)

### Recommended Sources:
| Source | Models Available | Format |
|--------|------------------|--------|
| **[FetchCFD](https://fetchcfd.com)** | 2022 F1 Car, 2026 Concept, Aston Martin AMR23 | STL, GLB |
| **[GrabCAD](https://grabcad.com)** | PERRINN F1, Formula car for CFD | STEP, STL, Parasolid |
| **[PERRINN](https://perrinn.com)** | Open-source 2017 F1 car (full geometry) | STEP, STL |
| **[Cad Crowd](https://cadcrowd.com)** | F1 aerodynamic parts | Various |

### PERRINN Open Source F1 Car
The PERRINN F1 car is the most complete open-source F1 geometry available:
- Full aero package (front wing, rear wing, floor, diffuser)
- Body geometry
- Wheel geometry
- Used in academic CFD studies

---

## 3. OpenFOAM Simulation Workflow

### Step 1: Geometry Preparation
```bash
# Convert CAD to STL (if needed)
# Use FreeCAD or Blender to export clean STL

# Clean geometry (remove small features, close gaps)
surfaceCheck car.stl
surfaceClean car.stl car_clean.stl
```

### Step 2: Create Wind Tunnel Domain
```bash
# Create background mesh (wind tunnel box)
blockMesh

# Domain dimensions (typical):
# Length: 5 car lengths upstream, 10 downstream
# Width: 5 car widths
# Height: 5 car heights
```

### Step 3: Generate Mesh
```bash
# Use snappyHexMesh for complex geometry
snappyHexMesh -overwrite

# Key parameters in snappyHexMeshDict:
# - castellatedMeshControls: feature edges, surface refinement
# - snapControls: mesh snapping to STL
# - addLayersControls: boundary layer mesh
```

### Step 4: Boundary Conditions
```bash
# 0/ directory setup:
# - inlet: fixedValue (freestream velocity, e.g., 50 m/s)
# - outlet: zeroGradient
# - ground: movingWallVelocity (same as inlet)
# - car: noSlip wall
# - top/sides: slip
```

### Step 5: Run Simulation
```bash
# Steady-state RANS simulation
simpleFoam

# Or for transient
pimpleFoam
```

### Step 6: Post-Processing
```bash
# Generate force coefficients
postProcess -func forceCoeffs

# Open in ParaView
paraFoam
```

---

## 4. Typical F1 CFD Parameters

### Flow Conditions
| Parameter | Value | Notes |
|-----------|-------|-------|
| Freestream Velocity | 50 m/s (180 km/h) | Typical wind tunnel speed |
| Reynolds Number | ~10^7 | Based on car length |
| Reference Area | 1.5 m² | Frontal area |
| Reference Length | 5.7 m | Car length |

### Turbulence Model
- **k-omega SST**: Best for automotive external flow
- **Spalart-Allmaras**: Simpler, good for attached flows
- **k-epsilon**: Less accurate for separation

### Mesh Requirements
| Region | Cell Size | Notes |
|--------|-----------|-------|
| Far-field | 500mm | Coarse background |
| Near-body | 10-20mm | Surface refinement |
| Wake region | 50mm | Refinement box |
| Boundary layer | y+ < 1 or ~30-50 | Wall functions |

---

## 5. GitHub Repositories

### OpenFOAM F1/Race Car Examples:
| Repository | Description |
|------------|-------------|
| [JulianSchutsch/FOAMCAR](https://github.com/JulianSchutsch/FOAMCAR) | Race car CFD, multiple wind angles |
| [openFoamHub/openFoamTutorials](https://github.com/openFoamHub/openFoamTutorials) | Car flow tutorials |
| [mjennings2/open-foam-adjoint-analysis](https://github.com/mjennings2/open-foam-adjoint-analysis) | FSAE wing optimization |
| [Interfluo/OpenFOAM-Cases-Interfluo](https://github.com/Interfluo/OpenFOAM-Cases-Interfluo) | Various aero cases |

---

## 6. Quick Start Commands

```bash
# 1. Clone a tutorial case
git clone https://github.com/JulianSchutsch/FOAMCAR.git
cd FOAMCAR

# 2. Source OpenFOAM environment
source /opt/openfoam10/etc/bashrc

# 3. Generate mesh
blockMesh
surfaceFeatures
snappyHexMesh -overwrite

# 4. Initialize fields
setFields

# 5. Run solver
simpleFoam | tee log.simpleFoam

# 6. Post-process
postProcess -func forceCoeffs
paraFoam
```

---

## 7. Expected Results

### Force Coefficients (typical F1 car at 50 m/s)
| Coefficient | Value | Force @ 50 m/s |
|-------------|-------|----------------|
| Cd (Drag) | 0.7-0.9 | ~1,600-2,000 N |
| Cl (Downforce) | 2.5-4.0 | ~5,700-9,000 N |
| L/D Ratio | 2.5-4.0 | - |

### Visualization Outputs
- Pressure contours on car surface
- Streamlines showing flow patterns
- Velocity magnitude slices
- Vortex structures (Q-criterion)
- Wake analysis

---

## 8. Cloud CFD Options (No Install Required)

| Service | Free Tier | Description |
|---------|-----------|-------------|
| **[SimScale](https://simscale.com)** | 3000 core-hours | Full CFD, CAD import |
| **[AirShaper](https://airshaper.com)** | Limited | Automated aero |
| **[OnScale](https://onscale.com)** | Trial | Cloud HPC |

---

## Resources

### Tutorials
- [OpenFOAM F1 Front Wing (YouTube)](https://youtube.com)
- [Tensor Engineering F1 CFD](https://tensorengineering.us)
- [Formula Careers CFD Guide](https://formulacareers.com)

### Documentation
- [OpenFOAM User Guide](https://openfoam.com/documentation)
- [snappyHexMesh Tutorial](https://openfoamwiki.net/index.php/SnappyHexMesh)
- [PERRINN CFD Data](https://perrinn.com)

---

*This guide is part of the F1 Data Analysis project.*
