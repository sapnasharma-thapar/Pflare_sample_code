import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    # PFLARE bindings (required)
    import pflare
except ModuleNotFoundError:
    print("petsc4py or pflare not found")
    sys.exit()

'''
=====================================================================
Steady 1D Convection–Diffusion
a*u_x - nu*u_xx = 0

Solver:
- KSP: GMRES
- PC : IAIR
- AIR algorithm: PFLARE
=====================================================================
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
a = 2.0
nu = 0.05
n = 100
dx = 1.0 / (n - 1)

phi_left  = 1.0
phi_right = 0.0

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)

# --------------------------------------------------
# Matrix Assembly 
# --------------------------------------------------
A = da.createMatrix()
(xs, xe) = da.getRanges()[0]

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    if i == 0:
        A.setValueStencil(row, row, 1.0)

    elif i == n - 1:
        A.setValueStencil(row, row, 1.0)

    else:
        # Diagonal
        col.index = (i,)
        A.setValueStencil(
            row, col,
            (a / dx) + (2.0 * nu / dx**2)
        )

        # Left (upwind)
        col.index = (i - 1,)
        A.setValueStencil(
            row, col,
            (-a / dx) - (nu / dx**2)
        )

        # Right
        col.index = (i + 1,)
        A.setValueStencil(
            row, col,
            -nu / dx**2
        )

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# RHS
# --------------------------------------------------
b = da.createGlobalVec()
b.set(0.0)

with da.getVecArray(b) as b_arr:
    if xs <= 0 < xe:
        b_arr[0] = phi_left
    if xs <= n-1 < xe:
        b_arr[n-1] = phi_right

# ---------------------------------------------------------
# 4. SOLVER SETUP (The "Nuclear Option")
# ---------------------------------------------------------
u_sol = da.createGlobalVec()

# 1. Force Matrix to AIJ Format
# DMDA creates a stencil matrix by default. We convert it to standard AIJ
# so the algebraic multigrid (AIR) can actually read the graph.
A.convert("aij") 

ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

# 2. Set a specific prefix to bind options tightly
# This tells PETSc: "Only apply options starting with 'my_solver_' to this KSP"
ksp.setOptionsPrefix("my_solver_")

pc = ksp.getPC()
pc.setType("air")
# 3. Configure via Global Options Database using the Prefix
opts = PETSc.Options()
opts["my_solver_pc_air_type"] = "iair"
opts["my_solver_pc_air_strong_threshold"] = 0.7  # High threshold = Safer (less aggressive)
opts["my_solver_pc_air_z_type"] = "product"
opts["my_solver_pc_air_poly_degree"] = 1
opts["my_solver_pc_air_reuse_interpolation"] = True

# 4. Trigger setup
ksp.setFromOptions()

# --------------------------------------------------
# Solve
# --------------------------------------------------
print("Solving...")
ksp.solve(b, u_sol)

reason = ksp.getConvergedReason()
iters = ksp.getIterationNumber()
print(f"Convergence Reason = {reason}")
print(f"Iterations = {iters}")

if reason < 0:
    print(f"\n[FAILURE] Solver diverged with reason {reason}.")
# --------------------------------------------------
# Visualization
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
u_final = u_sol.getArray()

plt.figure(figsize=(8, 5))
plt.plot(x_axis, u_final, 'o-', markersize=3, label='PFLARE + AIR')
plt.title("Steady 1D Convection–Diffusion (PFLARE)")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("steady_state_pflare_air.png")
