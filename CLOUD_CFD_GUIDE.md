# Cloud CFD Guide: SimScale & AirShaper

## Run Real F1 CFD Simulations for Free (No Installation Required)

This guide walks you through running professional CFD simulations using cloud platforms.

---

## Option 1: SimScale (Recommended)

### What is SimScale?
- Cloud-based engineering simulation platform
- **Free tier: 3,000 core-hours** (enough for several F1 car simulations)
- Full CFD capabilities with OpenFOAM backend
- Real pressure maps, force coefficients, streamlines

### Step 1: Create Account
1. Go to [simscale.com](https://www.simscale.com)
2. Sign up for free Community account
3. Verify email

### Step 2: Upload CAD Model
1. Create new project: "F1 Aerodynamics"
2. Import geometry:
   - Download F1 car STL from [FetchCFD](https://fetchcfd.com) or [GrabCAD](https://grabcad.com)
   - Upload to SimScale (supports STEP, STL, OBJ, Parasolid)
3. Check geometry for issues

### Step 3: Set Up Simulation
1. **Create Simulation > Incompressible (CFD)**
2. **Turbulence Model:** k-omega SST
3. **Domain:** 
   - Create external flow region
   - Wind tunnel: 5 car lengths upstream, 10 downstream
4. **Boundary Conditions:**
   - Inlet: Velocity inlet (50 m/s or 180 km/h)
   - Outlet: Pressure outlet (0 Pa gauge)
   - Ground: Moving wall (50 m/s)
   - Car surface: No-slip wall
   - Top/sides: Slip
5. **Mesh:** 
   - Use automatic meshing
   - Add refinements around car body
   - Target: 5-10 million cells for good accuracy

### Step 4: Run Simulation
1. Click "Run"
2. Select compute resources (use free cores)
3. Wait for convergence (typically 1-2 hours)

### Step 5: Post-Processing
1. **Pressure contours:** Visualize surface pressure
2. **Streamlines:** See airflow patterns
3. **Force coefficients:** Extract Cd, Cl values
4. **Export images/videos** for portfolio

### SimScale Tips:
- Start with coarse mesh (faster) then refine
- Check residuals for convergence
- Export results as images for GitHub

---

## Option 2: AirShaper

### What is AirShaper?
- Automated aerodynamics platform
- Built on OpenFOAM
- **Free tier available** with limited simulations
- Simplified workflow - great for beginners

### Step 1: Create Account
1. Go to [airshaper.com](https://www.airshaper.com)
2. Sign up for free account

### Step 2: Upload Model
1. Create new project
2. Upload STL file (max size limits on free tier)
3. Set scale and orientation

### Step 3: Configure Simulation
1. **Freestream velocity:** 50 m/s
2. **Ground simulation:** Enable moving ground
3. **Mesh quality:** Medium (for free tier)

### Step 4: Run and Analyze
1. Submit simulation
2. Results typically in 1-2 hours
3. View:
   - Drag coefficient
   - Lift coefficient
   - Pressure map
   - Streamlines
   - Turbulence visualization

### AirShaper Output:
- 3D interactive viewer
- Downloadable reports
- Comparison between designs

---

## Option 3: OnScale (Advanced)

### Features:
- High-performance cloud HPC
- More complex simulations
- Trial credits available

### Use for:
- Transient simulations
- Multi-physics (thermal + aero)
- Very fine meshes (50M+ cells)

---

## Where to Get F1 Car Models

### Free CFD-Ready Models:

| Source | Model | Format |
|--------|-------|--------|
| [FetchCFD](https://fetchcfd.com) | F1 2022 Car | STL, GLB |
| [GrabCAD](https://grabcad.com) | PERRINN F1 | STEP, STL |
| [GrabCAD](https://grabcad.com) | F1 CFD Study Car | Solidworks |
| [Free3D](https://free3d.com) | Various F1 models | OBJ, 3DS |

### PERRINN Open Source Car:
- Most complete open-source F1 geometry
- Full aerodynamic package
- Used in academic studies
- Search "PERRINN F1" on GrabCAD

---

## Expected Results

### Typical F1 Car at 50 m/s (180 km/h):

| Parameter | Low DF (Monza) | Medium (Reference) | High DF (Monaco) |
|-----------|----------------|-------------------|------------------|
| Cd | 0.90 - 1.00 | 1.10 - 1.20 | 1.30 - 1.50 |
| Cl | 2.20 - 2.80 | 3.00 - 3.50 | 3.80 - 4.50 |
| Downforce | 4,000 - 5,000 N | 5,500 - 6,500 N | 7,000 - 8,500 N |
| Drag | 1,600 - 1,900 N | 2,000 - 2,200 N | 2,400 - 2,800 N |
| L/D Ratio | 2.5 - 3.0 | 2.7 - 3.2 | 2.8 - 3.3 |

---

## Exporting Results for Portfolio

### What to Export:

1. **Pressure contour images** (top, side, front view)
2. **Streamline visualizations**
3. **Force coefficient plots** (convergence history)
4. **Comparison table** (different setups)
5. **Report PDF** (SimScale generates these)

### For GitHub:
```markdown
## CFD Results

| Setup | Cd | Cl | Downforce @ 250 km/h |
|-------|----|----|---------------------|
| Low DF | 0.95 | 2.5 | 8,500 N |
| Medium | 1.16 | 3.25 | 11,000 N |
| High DF | 1.35 | 4.0 | 13,600 N |

![Pressure Map](cfd_results/pressure_map.png)
![Streamlines](cfd_results/streamlines.png)
```

---

## Quick Comparison

| Platform | Free Tier | Ease of Use | Accuracy | Best For |
|----------|-----------|-------------|----------|----------|
| SimScale | 3000 core-hours | Medium | High | Serious analysis |
| AirShaper | Limited runs | Easy | Medium | Quick studies |
| OnScale | Trial credits | Medium | Very High | Advanced work |

---

## Workflow Summary

```
1. Get F1 CAD model (FetchCFD/GrabCAD)
           ↓
2. Upload to SimScale/AirShaper
           ↓
3. Configure wind tunnel domain
           ↓
4. Set boundary conditions
           ↓
5. Generate mesh
           ↓
6. Run simulation (1-2 hours)
           ↓
7. Post-process results
           ↓
8. Export images for portfolio
           ↓
9. Add to GitHub with analysis
```

---

## Resources

- [SimScale Learning Center](https://www.simscale.com/docs/)
- [SimScale External Aerodynamics Tutorial](https://www.simscale.com/docs/tutorials/)
- [AirShaper YouTube Channel](https://www.youtube.com/@AirShaper)
- [F1 Aerodynamics Explained](https://www.youtube.com/results?search_query=f1+cfd+aerodynamics)

---

*Part of the F1 Data Analysis Project*
