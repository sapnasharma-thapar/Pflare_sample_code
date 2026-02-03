import numpy as np 
import sys
import petsc4py
import matplotlib.pyplot as plt

# Initialize PETSc
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Try to load PFLARE
try:
    import pflare 
    print("PFLARE library loaded successfully.")
except ImportError:
    print("Warning: PFLARE module not found. Check PYTHONPATH. Using standard PETSc.")

# ---------------------------------------------------------
# 1. PARAMETERS
# ---------------------------------------------------------
a = 1.0       # Advection Velocity
v = 0.001     # Diffusion Coefficient (Reduced so pulse doesn't vanish instantly)
n = 200       # Grid points
dt = 0.001    # Time step

dx = 1.0 / (n - 1)  

# Coefficients
k = (a * dt) / dx           # Courant Number
beta = (v * dt) / (dx**2)   # Diffusion Number

print(f"Grid: N={n}, dx={dx:.4f}")
print(f"Coeffs: CFL={k:.4f}, Beta={beta:.4f}")
print("------------------------------------------------")

# ---------------------------------------------------------
# 2. SETUP GRID & FIELDS
# ---------------------------------------------------------
da = PETSc.DMDA().create(sizes=[n], dof=1, stencil_width=1, boundary_type=PETSc.DM.BoundaryType.NONE)
da.setUniformCoordinates(0.0, 1.0)

# Create Vectors
u_initial = da.createGlobalVec()
b_rhs = da.createGlobalVec()
x_sol = da.createGlobalVec()
u_prev = da.createGlobalVec()

# Set Initial Condition (Gaussian Pulse)
(xs, xe) = da.getRanges()[0]
with da.getVecArray(u_initial) as arr:
    i = np.arange(xs, xe)
    x = i * dx
    arr[:] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))

# Copy initial condition to previous step storage
u_initial.copy(u_prev)

# ---------------------------------------------------------
# 3. MATRIX ASSEMBLY (Implicit Upwind)
# ---------------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

val_diag  = 1.0 + k + 2.0 * beta 
val_left  = -(k + beta)
val_right = -beta

for i in range(xs, xe):
    row.index = (i,)
    if i == 0 or i == n - 1:
        col.index = (i,)
        A.setValueStencil(row, col, 1.0)
    else:
        col.index = (i,)
        A.setValueStencil(row, col, val_diag)
        col.index = (i - 1,)
        A.setValueStencil(row, col, val_left)
        col.index = (i + 1,)
        A.setValueStencil(row, col, val_right)

A.assemblyBegin()
A.assemblyEnd()

# ---------------------------------------------------------
# 4. SOLVER SETUP
# ---------------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES) 

pc = ksp.getPC()
# Try to use AIR if PFLARE is loaded, otherwise fallback to LU
if 'pflare' in sys.modules:
    pc.setType("air")
else:
    pc.setType(PETSc.PC.Type.LU) # Robust fallback
    
ksp.setFromOptions()

# ---------------------------------------------------------
# 5. TIME STEPPING LOOP
# ---------------------------------------------------------
num_steps = 100 
current_time = 0.0  # <--- Defined here to fix NameError

print(f"Starting simulation for {num_steps} steps...")

for step in range(num_steps):
    # RHS is simply the previous solution
    u_prev.copy(b_rhs)
    
    # Solve
    ksp.solve(b_rhs, x_sol)
    
    # Update
    x_sol.copy(u_prev)
    current_time += dt

print(f"Simulation finished at t = {current_time:.4f}")

# ---------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------
# Gather data to processor 0
scatter_ctx, u_init_global = PETSc.Scatter.toAll(u_initial)
scatter_ctx.scatter(u_initial, u_init_global, mode=PETSc.ScatterMode.FORWARD)

# We need a new vector for the final result
u_final_global = u_init_global.duplicate() 
scatter_ctx.scatter(x_sol, u_final_global, mode=PETSc.ScatterMode.FORWARD)

# Convert to Numpy
arr_init = u_init_global.getArray()
arr_final = u_final_global.getArray()
x_grid = np.linspace(0, 1, n)

# Plot
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x_grid, arr_init, 'k--', label='Initial (t=0)', alpha=0.6)
plt.plot(x_grid, arr_final, 'r-', linewidth=2, label=f'Final (t={current_time:.3f})')

plt.title(f"Advection-Diffusion Result\nCFL={k:.2f}, Beta={beta:.2f}, Steps={num_steps}")
plt.xlabel("Position x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True, alpha=0.5)

plt.savefig("advection_result.png")
print("Graph saved as 'advection_result.png'")