import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

import pflare   # <--- PFLARE ENABLED

'''
Linear Pure Advection Equation
u_t + a u_x = 0

Implicit Scheme (Backward Euler)
- Advection: Upwind (a > 0)
- Diffusion: NONE

Solver: GMRES
Preconditioner: AIR (PFLARE)
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
a = 1.0
n = 100
dx = 1.0 / (n - 1)
dt = 0.01

lam = a * dt / dx
nt = 100

# --------------------------------------------------
# DMDA grid (DIRICHLET / NON-PERIODIC)
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0)

# --------------------------------------------------
# Initial condition (Gaussian)
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe) = da.getRanges()[0]

with da.getVecArray(u_initial) as arr:
    i = np.arange(xs, xe)
    x = i * dx
    arr[:] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))

# --------------------------------------------------
# Matrix A assembly (IMPLICIT PURE ADVECTION)
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    col.index = (i,)
    A.setValueStencil(row, col, 1.0 + lam)

    if i - 1 >= 0:
        col.index = (i - 1,)
        A.setValueStencil(row, col, -lam)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Linear solver (GMRES + PFLARE AIR)
# --------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

pc = ksp.getPC()
pc.setType("air")     # <--- PFLARE AIR
ksp.setPC(pc)

pc.setFromOptions()
ksp.setFromOptions()

# --------------------------------------------------
# Time stepping
# --------------------------------------------------
u = u_initial.copy()
u_new = da.createGlobalVec()
b = da.createGlobalVec()

solution_history = []

for step in range(nt):
    with da.getVecArray(u) as u_arr, da.getVecArray(b) as b_arr:
        b_arr[:] = u_arr[:]

    ksp.solve(b, u_new)
    u.copy(u_new)
    solution_history.append(u.getArray().copy())

# --------------------------------------------------
# Animation
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot(x_axis, solution_history[0], lw=2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")

def update(frame):
    line.set_ydata(solution_history[frame])
    ax.set_title(f"Implicit Pure Advection | Time = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=60, blit=True)
ani.save("implicit_pure_advection_air.gif", dpi=150)
plt.close()

print(f"Final Iteration Count: {ksp.getIterationNumber()}")
print("Animation saved as implicit_pure_advection_air.gif")
