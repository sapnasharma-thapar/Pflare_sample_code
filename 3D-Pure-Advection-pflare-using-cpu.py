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

import pflare

'''
3D Steady Pure Advection Operator

3u(i,j,k) - u(i-1,j,k) - u(i,j-1,k) - u(i,j,k-1)

Raw operator matrix (no identity BC rows)
Solve Au = b and visualize
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------

nx = 100
ny = 100
nz = 100

length_x = 1.0
length_y = 1.0
length_z = 1.0

# --------------------------------------------------
# Grid spacing
# --------------------------------------------------

dx = length_x / (nx - 1)
dy = length_y / (ny - 1)
dz = length_z / (nz - 1)

# --------------------------------------------------
# DMDA grid (3D)
# --------------------------------------------------

da = PETSc.DMDA().create(
    sizes=[nx, ny, nz],
    dof=1,
    stencil_width=1,
    boundary_type=(
        PETSc.DM.BoundaryType.NONE,
        PETSc.DM.BoundaryType.NONE,
        PETSc.DM.BoundaryType.NONE,
    )
)

da.setUniformCoordinates(
    0.0, length_x,
    0.0, length_y,
    0.0, length_z
)

# --------------------------------------------------
# Matrix and vectors
# --------------------------------------------------

A = da.createMatrix()
b = da.createGlobalVec()
u = da.createGlobalVec()

(xs, xe), (ys, ye), (zs, ze) = da.getRanges()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

# --------------------------------------------------
# Assemble matrix
# --------------------------------------------------

with da.getVecArray(b) as b_arr:
    for k in range(zs, ze):
        for j in range(ys, ye):
            for i in range(xs, xe):

                row.index = (i, j, k)

                # diagonal
                A.setValueStencil(row, row, 3.0)

                # left neighbor
                if i > 0:
                    col.index = (i-1, j, k)
                    A.setValueStencil(row, col, -1.0)

                # bottom neighbor
                if j > 0:
                    col.index = (i, j-1, k)
                    A.setValueStencil(row, col, -1.0)

                # back neighbor
                if k > 0:
                    col.index = (i, j, k-1)
                    A.setValueStencil(row, col, -1.0)

                # RHS source
                b_arr[i, j, k] = 1.0

A.assemblyBegin()
A.assemblyEnd()

b.assemblyBegin()
b.assemblyEnd()


# --------------------------------------------------
# Solve
# --------------------------------------------------

ksp = PETSc.KSP().create(comm=da.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

pc = ksp.getPC()
pc.setType("air")
   
opts = PETSc.Options()

opts["my_solver_pc_air_type"] = "iair"
opts["my_solver_pc_air_strong_threshold"] = 0.7  
opts["my_solver_pc_air_z_type"] = "product"
opts["my_solver_pc_air_poly_degree"] = 1
opts["my_solver_pc_air_reuse_interpolation"] = True


ksp.setFromOptions()
ksp.solve(b, u)

print("\nSolver statistics")
print("Iterations:", ksp.getIterationNumber())
print("Residual:", ksp.getResidualNorm())
print("Converged:", ksp.getConvergedReason())

# --------------------------------------------------
# Convert to 3D numpy array
# --------------------------------------------------

u_arr = u.getArray().reshape((nz, ny, nx))

# --------------------------------------------------
# 2D slice heatmap (middle z slice)
# --------------------------------------------------

mid_k = nz // 2
slice_2d = u_arr[mid_k]

plt.figure(figsize=(6, 5))
plt.imshow(
    slice_2d,
    origin="lower",
    extent=[0.0, length_x, 0.0, length_y],
    cmap="viridis"
)

plt.colorbar(label="u")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"3D Solution Slice (z={mid_k*dz:.2f})")
plt.tight_layout()
plt.savefig("3d-steady-advection-slice.png", dpi=150)
plt.close()

print("Saved: 3d-steady-advection-slice.png")

# --------------------------------------------------
# 3D surface of middle slice
# --------------------------------------------------

X, Y = np.meshgrid(
    np.linspace(0.0, length_x, nx),
    np.linspace(0.0, length_y, ny),
    indexing="xy"
)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, slice_2d,
    cmap="magma",
    edgecolor="none"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("Surface of middle z-slice")

fig.colorbar(surf, ax=ax, shrink=0.6)

plt.tight_layout()
plt.savefig("3d-steady-advection-surface.png", dpi=150)
plt.close()

print("Saved: 3d-steady-advection-surface.png")

# --------------------------------------------------
# Centerline plot
# --------------------------------------------------

mid_j = ny // 2
centerline = u_arr[mid_k, mid_j, :]

plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0.0, length_x, nx), centerline, lw=2)

plt.xlabel("x")
plt.ylabel("u")
plt.title("3D centerline")
plt.grid(True)
plt.tight_layout()
plt.savefig("3d-steady-advection-centerline.png", dpi=150)
plt.close()

print("Saved: 3d-steady-advection-centerline.png")
