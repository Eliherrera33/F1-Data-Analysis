# WSL + OpenFOAM Installation Guide

## Complete Setup for F1 CFD Simulation on Windows

This guide walks you through setting up a full CFD environment on Windows using WSL2 and OpenFOAM.

---

## Step 1: Enable WSL2

### Open PowerShell as Administrator and run:

```powershell
# Enable WSL
wsl --install

# If already enabled, update to WSL2
wsl --set-default-version 2

# Restart your computer after this
```

### Install Ubuntu from Microsoft Store:
1. Open Microsoft Store
2. Search for "Ubuntu 22.04 LTS"
3. Click Install
4. Launch and create username/password

---

## Step 2: Install OpenFOAM

### In Ubuntu terminal:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Add OpenFOAM repository
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu

# Install OpenFOAM v10
sudo apt update
sudo apt install openfoam10 -y

# Add to bashrc
echo "source /opt/openfoam10/etc/bashrc" >> ~/.bashrc
source ~/.bashrc

# Verify installation
simpleFoam -help
```

---

## Step 3: Install ParaView (Visualization)

```bash
# Install ParaView
sudo apt install paraview -y

# For GUI, you need X server on Windows:
# Option A: Install VcXsrv from https://sourceforge.net/projects/vcxsrv/
# Option B: Use WSLg (Windows 11 built-in)
```

### For Windows 10 (VcXsrv):
1. Download and install VcXsrv
2. Launch XLaunch with:
   - Multiple windows
   - Start no client
   - Check "Disable access control"
3. In WSL, add to ~/.bashrc:
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

---

## Step 4: Test Installation

```bash
# Copy tutorial case
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/motorBike ~/test_case

# Run test
cd ~/test_case
./Allrun

# Check results
ls -la processor*/

# View in ParaView (if GUI works)
paraFoam
```

---

## Step 5: Run F1 CFD Simulation

### Copy case from Windows to WSL:

```bash
# Access Windows files (replace with your path)
cd /mnt/e/Repose\ E/F1\ Data/openfoam_f1_case

# Or copy to home directory
cp -r /mnt/e/Repose\ E/F1\ Data/openfoam_f1_case ~/f1_cfd
cd ~/f1_cfd
```

### Download F1 Car STL:

```bash
# Create directory for STL
mkdir -p constant/triSurface

# Download PERRINN F1 car (if available)
# Or use a simplified car model from tutorials
cp $FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz constant/triSurface/
gunzip constant/triSurface/motorBike.obj.gz

# Convert to STL if needed
surfaceMeshConvert motorBike.obj car.stl
```

### Run Simulation:

```bash
# Make run script executable
chmod +x Allrun

# Run full simulation
./Allrun

# Or run step by step:
blockMesh
surfaceFeatures
snappyHexMesh -overwrite
checkMesh
simpleFoam

# Post-process
postProcess -func forceCoeffs
```

---

## Step 6: View Results

### Check force coefficients:

```bash
# View drag and lift coefficients
cat postProcessing/forceCoeffs/0/coefficient.dat
```

### Open in ParaView:

```bash
# Create ParaView file
touch case.foam

# Open ParaView
paraFoam
```

### ParaView Visualization Tips:
1. **Pressure contours:** Add "Contour" filter on p field
2. **Streamlines:** Add "Stream Tracer" filter
3. **Surface pressure:** Select car patch, color by p
4. **Q-criterion vortices:** Add "Contour" on Q field

---

## Troubleshooting

### Common Issues:

**1. "simpleFoam: command not found"**
```bash
source /opt/openfoam10/etc/bashrc
```

**2. Mesh quality errors**
- Reduce refinement levels in snappyHexMeshDict
- Check STL geometry for holes/issues:
```bash
surfaceCheck constant/triSurface/car.stl
```

**3. Simulation diverging**
- Reduce relaxation factors in fvSolution
- Increase mesh resolution
- Check boundary conditions

**4. GUI not working**
```bash
# Test X server
xeyes
# If xeyes doesn't show, check DISPLAY variable
echo $DISPLAY
```

---

## Performance Tips

### For faster simulations:

```bash
# Run in parallel (4 cores)
decomposePar
mpirun -np 4 simpleFoam -parallel
reconstructPar

# Reduce mesh size for testing
# Edit system/blockMeshDict - reduce cell counts
```

### Recommended Hardware:
- **CPU:** 8+ cores (more = faster)
- **RAM:** 32GB+ for detailed meshes
- **Storage:** SSD recommended

---

## Resources

- [OpenFOAM Documentation](https://openfoam.org/documentation/)
- [CFD Online Forum](https://www.cfd-online.com/Forums/openfoam/)
- [OpenFOAM Wiki](https://openfoamwiki.net/)
- [ParaView Tutorials](https://www.paraview.org/tutorials/)

---

## Quick Reference Commands

```bash
# Mesh generation
blockMesh                    # Background mesh
snappyHexMesh               # Car mesh
checkMesh                   # Quality check

# Solvers
simpleFoam                  # Steady-state RANS
pimpleFoam                  # Transient
potentialFoam              # Initialize fields

# Post-processing
postProcess -func forceCoeffs    # Forces
postProcess -func yPlus          # Wall y+
paraFoam                         # Visualization

# Parallel
decomposePar                     # Split mesh
mpirun -np 4 simpleFoam -parallel
reconstructPar                   # Merge results
```

---

*Part of the F1 Data Analysis Project*
