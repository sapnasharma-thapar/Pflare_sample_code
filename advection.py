import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

'''
Linear pure advection equation
u_t + a u_x = 0

Implicit UPWIND scheme (Backward Euler)
Dirichlet boundary condition
Gaussian initial condition

Solver: GMRES
Preconditioner: Jacobi
'''

#parameters
a = 0.23          # Advection speed (a > 0)
n = 1000          # Number of grid points

dx = 1.0 / (n - 1)
dt = 0.01
lam = a * dt / dx    # <-- TRUE CFL for upwind

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
(xs, xe) = da.getRanges()[0]

with da.getVecArray(u_initial) as arr:
    i = np.arange(xs, xe)
    x = i * dx
    arr[:] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))

#grid init
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    # Diagonal: (1 + lambda)
    col.index = (i,)
    A.setValueStencil(row, col, 1.0 + lam)

    # Left neighbor: -lambda (upwind, a > 0)
    if i - 1 >= 0:
        col.index = (i - 1,)
        A.setValueStencil(row, col, -lam)

A.assemblyBegin()
A.assemblyEnd()


# print("\n===== Matrix A (PETSc view) =====")
A.view()

A_dense = A.convert("dense")
A_np = A_dense.getDenseArray()

# print("\n===== Matrix A (Dense NumPy) =====")
np.set_printoptions(precision=3, suppress=True)
print(A_np)


b = da.createGlobalVec()

with da.getVecArray(u_initial) as u, da.getVecArray(b) as b_arr:
    for i in range(xs, xe):
        b_arr[i] = u[i]   # RHS is simply u^n


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


# 1. Gather distributed vectors to Rank 0
scatter, u_seq = PETSc.Scatter.toZero(u_initial)
scatter, x_seq = PETSc.Scatter.toZero(x_sol)

scatter.scatter(u_initial, u_seq, False, PETSc.Scatter.Mode.FORWARD)
scatter.scatter(x_sol, x_seq, False, PETSc.Scatter.Mode.FORWARD)

# 2. Plotting (Only on Rank 0)
comm = PETSc.COMM_WORLD
rank = comm.getRank()

if rank == 0:
    u_in_data = u_seq.getArray()
    u_out_data = x_seq.getArray()
    X_axis = np.linspace(0.0, 1.0, n)
    
    # Create 2 subplots: Left=Physics, Right=Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Graph 1: The Physics ---
    ax1.plot(X_axis, u_in_data, 'b--', linewidth=1.5, label='Initial (t=0)')
    ax1.plot(X_axis, u_out_data, 'r-', linewidth=2.0, label=f'Solution (t={dt})')
    ax1.set_title(f"1D Linear Advection\nCFL: {lam:.2f}")
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("u(x)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Graph 2: The Residuals (Convergence) ---
    if len(residuals) > 0:
        ax2.semilogy(range(len(residuals)), residuals, 'k-o', markersize=5)
        ax2.set_title(f"Solver Convergence\nTotal Iterations: {len(residuals)-1}")
        ax2.set_xlabel("Iteration Number")
        ax2.set_ylabel("Residual Norm (Log Scale)")
        ax2.grid(True, which="both", linestyle='-', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "Converged too fast to plot!", ha='center')

    plt.tight_layout()
    print("Displaying graphs...")
    plt.show()