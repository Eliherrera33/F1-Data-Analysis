# -*- coding: utf-8 -*-
"""
F1 OpenFOAM Case Generator
Prepares case files for OpenFOAM CFD simulation
"""

import os
from pathlib import Path


class OpenFOAMCaseGenerator:
    """Generate OpenFOAM case structure for F1 car CFD."""
    
    def __init__(self, case_dir, car_stl="car.stl"):
        self.case_dir = Path(case_dir)
        self.car_stl = car_stl
        
        # Wind tunnel dimensions (meters)
        self.domain = {
            'x_min': -10,  # 2 car lengths upstream
            'x_max': 25,   # 5 car lengths downstream
            'y_min': -5,   # Half width
            'y_max': 5,    # Half width
            'z_min': 0,    # Ground
            'z_max': 8,    # Height
        }
        
        # Flow conditions
        self.velocity = 50.0  # m/s (180 km/h)
        self.nu = 1.5e-5      # kinematic viscosity (air at 20C)
        
    def create_structure(self):
        """Create OpenFOAM case directory structure."""
        dirs = ['0', 'constant', 'system', 'constant/triSurface']
        for d in dirs:
            (self.case_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"Created case structure in: {self.case_dir}")
        
    def write_control_dict(self):
        """Write system/controlDict."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;

deltaT          1;
writeControl    timeStep;
writeInterval   100;

purgeWrite      3;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        
        patches         (car);
        rho             rhoInf;
        rhoInf          1.225;
        
        CofR            (2.5 0 0.3);  // Center of car
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        pitchAxis       (0 1 0);
        
        magUInf         {self.velocity};
        lRef            5.7;   // Car length
        Aref            1.5;   // Frontal area
    }}
}}
"""
        with open(self.case_dir / 'system/controlDict', 'w') as f:
            f.write(content)
        print("  Written: system/controlDict")
        
    def write_fv_schemes(self):
        """Write system/fvSchemes."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
        with open(self.case_dir / 'system/fvSchemes', 'w') as f:
            f.write(content)
        print("  Written: system/fvSchemes")
        
    def write_fv_solution(self):
        """Write system/fvSolution."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-5;
        U               1e-5;
        "(k|omega)"     1e-5;
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        k               0.5;
        omega           0.5;
    }
    fields
    {
        p               0.3;
    }
}
"""
        with open(self.case_dir / 'system/fvSolution', 'w') as f:
            f.write(content)
        print("  Written: system/fvSolution")
        
    def write_block_mesh_dict(self):
        """Write system/blockMeshDict for wind tunnel domain."""
        d = self.domain
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({d['x_min']} {d['y_min']} {d['z_min']})  // 0
    ({d['x_max']} {d['y_min']} {d['z_min']})  // 1
    ({d['x_max']} {d['y_max']} {d['z_min']})  // 2
    ({d['x_min']} {d['y_max']} {d['z_min']})  // 3
    ({d['x_min']} {d['y_min']} {d['z_max']})  // 4
    ({d['x_max']} {d['y_min']} {d['z_max']})  // 5
    ({d['x_max']} {d['y_max']} {d['z_max']})  // 6
    ({d['x_min']} {d['y_max']} {d['z_max']})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (70 20 16) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    ground
    {{
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
    sides
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
        );
    }}
);
"""
        with open(self.case_dir / 'system/blockMeshDict', 'w') as f:
            f.write(content)
        print("  Written: system/blockMeshDict")
        
    def write_snappy_hex_mesh_dict(self):
        """Write system/snappyHexMeshDict for car mesh."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    car.stl
    {{
        type triSurfaceMesh;
        name car;
    }}
    
    refinementBox
    {{
        type searchableBox;
        min (-2 -1.5 0);
        max (8 1.5 2);
    }}
    
    wakeBox
    {{
        type searchableBox;
        min (5 -1 0);
        max (15 1 1.5);
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 100000000;
    maxGlobalCells 200000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    
    features
    (
        {{
            file "car.eMesh";
            level 6;
        }}
    );
    
    refinementSurfaces
    {{
        car
        {{
            level (5 6);
            patchInfo
            {{
                type wall;
            }}
        }}
    }}
    
    resolveFeatureAngle 30;
    
    refinementRegions
    {{
        refinementBox
        {{
            mode inside;
            levels ((1E15 4));
        }}
        wakeBox
        {{
            mode inside;
            levels ((1E15 3));
        }}
    }}
    
    locationInMesh (2.5 0 1.5);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    
    layers
    {{
        car
        {{
            nSurfaceLayers 5;
        }}
        ground
        {{
            nSurfaceLayers 3;
        }}
    }}
    
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.001;
    nGrow 0;
    featureAngle 180;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;
"""
        with open(self.case_dir / 'system/snappyHexMeshDict', 'w') as f:
            f.write(content)
        print("  Written: system/snappyHexMeshDict")
        
    def write_velocity_bc(self):
        """Write 0/U boundary conditions."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({self.velocity} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({self.velocity} 0 0);
    }}
    
    outlet
    {{
        type            zeroGradient;
    }}
    
    ground
    {{
        type            fixedValue;
        value           uniform ({self.velocity} 0 0);  // Moving ground
    }}
    
    top
    {{
        type            slip;
    }}
    
    sides
    {{
        type            slip;
    }}
    
    car
    {{
        type            noSlip;
    }}
}}
"""
        with open(self.case_dir / '0/U', 'w') as f:
            f.write(content)
        print("  Written: 0/U")
        
    def write_pressure_bc(self):
        """Write 0/p boundary conditions."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    ground
    {
        type            zeroGradient;
    }
    
    top
    {
        type            slip;
    }
    
    sides
    {
        type            slip;
    }
    
    car
    {
        type            zeroGradient;
    }
}
"""
        with open(self.case_dir / '0/p', 'w') as f:
            f.write(content)
        print("  Written: 0/p")
        
    def write_turbulence_bc(self):
        """Write turbulence boundary conditions (k-omega SST)."""
        # Turbulence intensity and length scale
        I = 0.01  # 1% turbulence intensity
        L = 0.1   # 10cm turbulent length scale
        
        k_inlet = 1.5 * (self.velocity * I)**2
        omega_inlet = k_inlet**0.5 / (0.09**0.25 * L)
        
        # k file
        k_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_inlet:.4f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_inlet:.4f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    ground
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.4f};
    }}
    top
    {{
        type            slip;
    }}
    sides
    {{
        type            slip;
    }}
    car
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.4f};
    }}
}}
"""
        with open(self.case_dir / '0/k', 'w') as f:
            f.write(k_content)
        print("  Written: 0/k")
        
        # omega file
        omega_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega_inlet:.2f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.2f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    ground
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.2f};
    }}
    top
    {{
        type            slip;
    }}
    sides
    {{
        type            slip;
    }}
    car
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.2f};
    }}
}}
"""
        with open(self.case_dir / '0/omega', 'w') as f:
            f.write(omega_content)
        print("  Written: 0/omega")
        
        # nut file
        nut_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    ground
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
    top
    {
        type            calculated;
        value           uniform 0;
    }
    sides
    {
        type            calculated;
        value           uniform 0;
    }
    car
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
}
"""
        with open(self.case_dir / '0/nut', 'w') as f:
            f.write(nut_content)
        print("  Written: 0/nut")
        
    def write_transport_properties(self):
        """Write constant/transportProperties."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] {self.nu};
"""
        with open(self.case_dir / 'constant/transportProperties', 'w') as f:
            f.write(content)
        print("  Written: constant/transportProperties")
        
    def write_turbulence_properties(self):
        """Write constant/turbulenceProperties."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""
        with open(self.case_dir / 'constant/turbulenceProperties', 'w') as f:
            f.write(content)
        print("  Written: constant/turbulenceProperties")
        
    def write_run_script(self):
        """Write Allrun script."""
        content = """#!/bin/bash
# OpenFOAM F1 Car CFD Run Script

# Source OpenFOAM environment
source /opt/openfoam10/etc/bashrc

echo "=== F1 Car CFD Simulation ==="
echo ""

# 1. Create background mesh
echo "Step 1: Creating background mesh..."
blockMesh | tee log.blockMesh

# 2. Extract surface features
echo "Step 2: Extracting surface features..."
surfaceFeatures | tee log.surfaceFeatures

# 3. Create mesh around car
echo "Step 3: Creating mesh with snappyHexMesh..."
snappyHexMesh -overwrite | tee log.snappyHexMesh

# 4. Check mesh quality
echo "Step 4: Checking mesh..."
checkMesh | tee log.checkMesh

# 5. Run solver
echo "Step 5: Running simpleFoam..."
simpleFoam | tee log.simpleFoam

# 6. Post-process
echo "Step 6: Post-processing..."
postProcess -func forceCoeffs

echo ""
echo "=== Simulation Complete ==="
echo "View results with: paraFoam"
"""
        script_path = self.case_dir / 'Allrun'
        with open(script_path, 'w') as f:
            f.write(content)
        print("  Written: Allrun")
        
    def generate_case(self):
        """Generate complete OpenFOAM case."""
        print("=" * 50)
        print("F1 OpenFOAM Case Generator")
        print("=" * 50)
        print(f"\nVelocity: {self.velocity} m/s ({self.velocity * 3.6:.0f} km/h)")
        print(f"Domain: {self.domain['x_max'] - self.domain['x_min']}m x "
              f"{self.domain['y_max'] - self.domain['y_min']}m x "
              f"{self.domain['z_max'] - self.domain['z_min']}m")
        print()
        
        self.create_structure()
        print("\nWriting system files...")
        self.write_control_dict()
        self.write_fv_schemes()
        self.write_fv_solution()
        self.write_block_mesh_dict()
        self.write_snappy_hex_mesh_dict()
        
        print("\nWriting boundary conditions...")
        self.write_velocity_bc()
        self.write_pressure_bc()
        self.write_turbulence_bc()
        
        print("\nWriting constant files...")
        self.write_transport_properties()
        self.write_turbulence_properties()
        
        print("\nWriting run script...")
        self.write_run_script()
        
        print("\n" + "=" * 50)
        print("OpenFOAM case generated successfully!")
        print("=" * 50)
        print(f"""
Next steps:
1. Place your F1 car STL file in: {self.case_dir / 'constant/triSurface/car.stl'}
2. On Linux/WSL with OpenFOAM installed:
   cd {self.case_dir}
   chmod +x Allrun
   ./Allrun
3. View results: paraFoam
""")


def main():
    # Generate case in current directory
    output_dir = Path(__file__).parent / 'openfoam_f1_case'
    
    generator = OpenFOAMCaseGenerator(output_dir)
    generator.generate_case()


if __name__ == "__main__":
    main()
