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

# --------------------------------------------------
# Parameters
# --------------------------------------------------
ax, ay, az = 2.0, 1.0, 0.5   # Advection velocities
nu = 0.05                   # Diffusion

nx, ny, nz = 32, 32, 32
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
dz = 1.0 / (nz - 1)

phi_left = 1.0

# --------------------------------------------------
# DMDA grid (3D)
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[nx, ny, nz],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)

# --------------------------------------------------
# Matrix Assembly
# --------------------------------------------------
A = da.createMatrix()
(xs, xe), (ys, ye), (zs, ze) = da.getRanges()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for k in range(zs, ze):
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j, k)

            # Dirichlet boundaries
            if i == 0:
                A.setValueStencil(row, row, 1.0)

            elif (i == nx - 1 or
                  j == 0 or j == ny - 1 or
                  k == 0 or k == nz - 1):
                A.setValueStencil(row, row, 1.0)

            else:
                col.index = (i, j, k)
                val_diag = (
                    ax / dx + ay / dy + az / dz
                    + 2.0 * nu * (
                        1.0 / dx**2 +
                        1.0 / dy**2 +
                        1.0 / dz**2
                    )
                )
                A.setValueStencil(row, col, val_diag)

                # West (upwind x)
                col.index = (i - 1, j, k)
                A.setValueStencil(row, col, -(ax / dx + nu / dx**2))

                # East
                col.index = (i + 1, j, k)
                A.setValueStencil(row, col, -nu / dx**2)

                # South (upwind y)
                col.index = (i, j - 1, k)
                A.setValueStencil(row, col, -(ay / dy + nu / dy**2))

                # North
                col.index = (i, j + 1, k)
                A.setValueStencil(row, col, -nu / dy**2)

                # Bottom (upwind z)
                col.index = (i, j, k - 1)
                A.setValueStencil(row, col, -(az / dz + nu / dz**2))

                # Top
                col.index = (i, j, k + 1)
                A.setValueStencil(row, col, -nu / dz**2)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# RHS Assembly
# --------------------------------------------------
b = da.createGlobalVec()
b.set(0.0)

(xs, xe), (ys, ye), (zs, ze) = da.getRanges()
with da.getVecArray(b) as b_arr:
    for k in range(zs, ze):
        for j in range(ys, ye):
            if xs <= 0 < xe:
                b_arr[0, j, k] = phi_left

# --------------------------------------------------
# Solver
# --------------------------------------------------
u_sol = da.createGlobalVec()

ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

pc = ksp.getPC()
pc.setType("air")
opts = PETSc.Options()
opts["my_solver_pc_air_type"] = "iair"
opts["my_solver_pc_air_strong_threshold"] = 0.7  # High threshold = Safer (less aggressive)
opts["my_solver_pc_air_z_type"] = "product"
opts["my_solver_pc_air_poly_degree"] = 1
opts["my_solver_pc_air_reuse_interpolation"] = True

ksp.setFromOptions()
ksp.solve(b, u_sol)

print(f"Convergence Reason: {ksp.getConvergedReason()}")
print(f"Iterations = {ksp.getIterationNumber()}")

# --------------------------------------------------
# Visualization (2D slice + 1D line)
# --------------------------------------------------
solution = u_sol.getArray().reshape(nx, ny, nz).transpose(2, 1, 0)

# ---- 2D slice at z = 0.5 ----
mid_z = nz // 2
slice_xy = solution[mid_z, :, :]

plt.figure(figsize=(6, 5))
plt.imshow(slice_xy, origin='lower',
           extent=[0, 1, 0, 1], cmap='magma')
plt.colorbar(label='u')
plt.title("3D Steady Convection–Diffusion (z = 0.5)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("steady_3d_slice.png")

# ---- 1D centerline ----
mid_y = ny // 2
line_x = slice_xy[mid_y, :]
x_axis = np.linspace(0.0, 1.0, nx)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, line_x, 'o-', markersize=3, label='y=z=0.5')
plt.title("1D Slice from 3D Steady Convection–Diffusion")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("steady_3d_centerline.png")
