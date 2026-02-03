import numpy as np 
import matplotlib.pyplot as plt
import sys
from petsc4py import PETSc

'''
Linear pure advection equation
u_t + a u_x = v*u_xx

Implicit UPWIND scheme (Backward Euler)
central scheme for diffusion term
Dirichlet boundary condition (Left side)
Gaussian initial condition

Solver: GMRES
Preconditioner: jacobi
'''

# init parameters
a = 0.23
v = 0.1
n = 1000
dt = 0.001

# --- FIX 1: Parentheses for correct order of operations ---
dx = 1.0 / (n - 1)  

k = (a * dt) / dx
b = (v * dt) / (dx**2)

# dmda grid
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0)

# initial conditions
u_initial = da.createGlobalVec()
(xs, xe) = da.getRanges()[0] # Unpacking for 1D

with da.getVecArray(u_initial) as arr:
    # Use global indices 'i' to calculate physical coordinate 'x'
    i = np.arange(xs, xe)
    x = i * dx
    arr[:] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))

# matrix A init
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

coeff_diag  = 1.0 + k + 2.0 * b 
coeff_left  = -(k + b)
coeff_right = -b

for i in range(xs, xe):
    row.index = (i,)
    
    # Boundary Conditions (Dirichlet u=0 at ends)
    if i == 0 or i == n - 1:
        # --- FIX 2: Use assignment '=' instead of function call '()' ---
        col.index = (i,)
        A.setValueStencil(row, col, 1.0)
        
    else:
        # Diagonal: (1 + k + 2b)
        col.index = (i,)
        A.setValueStencil(row, col, coeff_diag)

        # Left Neighbor: -(k + b)
        col.index = (i - 1,)
        A.setValueStencil(row, col, coeff_left)

        # Right Neighbor: -b
        col.index = (i + 1,)
        A.setValueStencil(row, col, coeff_right)

A.assemblyBegin()
A.assemblyEnd()

# Visualization check
# A.view() 
A_dense = A.convert("dense")
A_np = A_dense.getDenseArray()
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# Print just the top-left corner to verify the structure without flooding the screen
print("Top-left 5x5 corner of Matrix A:")
print(A_np)

b = da.createGlobalVec()

with da.getVecArray(u_initial) as u, da.getVecArray(b) as b_arr:
    for i in range(xs, xe):
        b_arr[i] = u[i]

ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

# NOTE: If you use LU, this will solve in 1 iteration (Direct Solve).
# To see a convergence graph, try changing this to PETSc.PC.Type.JACOBI
ksp.getPC().setType(PETSc.PC.Type.JACOBI) 

ksp.setFromOptions()

# --- NEW: Define the Monitor ---
residuals = []
def monitor(ksp, it, rnorm):
    residuals.append(rnorm)
    # Optional: Print live progress
    # print(f"  Iteration {it}: Residual = {rnorm:.4e}")

# Register the monitor with the KSP
ksp.setMonitor(monitor)

x_sol = da.createGlobalVec()
ksp.solve(b, x_sol)

print(f"\nSolver Complete in {ksp.getIterationNumber()} iterations.")

