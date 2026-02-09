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
2D Steady Pure Advection Operator

2u(i,j) - u(i-1,j) - u(i,j-1)

Raw operator matrix (no identity BC rows)
Solve Au = b and visualize
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------

nx = 400
ny = 400
length_x = 1.0
length_y = 1.0

# --------------------------------------------------
# Grid spacing
# --------------------------------------------------

dx = length_x / (nx - 1)
dy = length_y / (ny - 1)

# --------------------------------------------------
# DMDA grid (2D)
# --------------------------------------------------

da = PETSc.DMDA().create(
    sizes=[nx, ny],
    dof=1,
    stencil_width=1,
    boundary_type=(
        PETSc.DM.BoundaryType.NONE,
        PETSc.DM.BoundaryType.NONE,
    )
)

da.setUniformCoordinates(0.0, length_x, 0.0, length_y)

# --------------------------------------------------
# Matrix and vectors
# --------------------------------------------------

A = da.createMatrix()
b = da.createGlobalVec()
u = da.createGlobalVec()

(xs, xe), (ys, ye) = da.getRanges()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

# --------------------------------------------------
# Matrix and vectors (GPU)
# --------------------------------------------------

# A = da.createMatrix()
# A.setType(PETSc.Mat.Type.AIJCUSPARSE)

# b = da.createGlobalVec()
# u = da.createGlobalVec()

# b.setType(PETSc.Vec.Type.SEQCUDA)
# u.setType(PETSc.Vec.Type.SEQCUDA)

# (xs, xe), (ys, ye) = da.getRanges()

# row = PETSc.Mat.Stencil()
# col = PETSc.Mat.Stencil()

# --------------------------------------------------
# Assemble matrix: 2u - left - bottom
# --------------------------------------------------

with da.getVecArray(b) as b_arr:
    for j in range(ys, ye):
        for i in range(xs, xe):

            row.index = (i, j)

            # diagonal
            A.setValueStencil(row, row, 2.0)

            # left neighbor
            if i > 0:
                col.index = (i - 1, j)
                A.setValueStencil(row, col, -1.0)

            # bottom neighbor
            if j > 0:
                col.index = (i, j - 1)
                A.setValueStencil(row, col, -1.0)

            # RHS (example source)
            b_arr[i, j] = 1.0

A.assemblyBegin()
A.assemblyEnd()

b.assemblyBegin()
b.assemblyEnd()

# # --- Print the Matrix Visibly ---
size = A.getSize()[0]
dense_A = np.zeros((size, size))

# Extract values into a numpy array for clean printing
for i in range(size):
    cols, vals = A.getRow(i)
    dense_A[i, cols] = vals

print(f"Full Matrix A ({size}x{size}) for a 4x4 grid:\n")
np.set_printoptions(precision=1, suppress=True, linewidth=120)
print(dense_A)

# --------------------------------------------------
# Matrix preview
# --------------------------------------------------

def print_matrix_preview(mat, name, max_rows=8, max_cols=8):
    nrows, ncols = mat.getSize()
    print(f"\n{name} (preview {max_rows}x{max_cols}, size {nrows}x{ncols})")

    if nrows <= max_rows:
        row_idx = list(range(nrows))
        row_break = None
    else:
        half = max_rows // 2
        row_idx = list(range(half)) + list(range(nrows - (max_rows - half), nrows))
        row_break = half

    if ncols <= max_cols:
        col_idx = list(range(ncols))
        col_break = None
    else:
        half = max_cols // 2
        col_idx = list(range(half)) + list(range(ncols - (max_cols - half), ncols))
        col_break = half

    vals = mat.getValues(row_idx, col_idx)

    def fmt_row(r):
        cells = [f"{v:10.3g}" for v in r]
        if col_break is not None:
            cells.insert(col_break, "   ...   ")
        return " ".join(cells)

    for i, row in enumerate(vals):
        if row_break is not None and i == row_break:
            print("   ...")
        print(fmt_row(row))

print_matrix_preview(A, "Matrix A (2D Operator)")

# --------------------------------------------------
# Solve
# --------------------------------------------------

ksp = PETSc.KSP().create(comm=da.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

ksp.setTolerances(rtol=0.0, atol=1e-5)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.JACOBI)

ksp.solve(b, u)

print("\nSolver statistics")
print("Iterations:", ksp.getIterationNumber())
print("Residual:", ksp.getResidualNorm())
print("Converged:", ksp.getConvergedReason())

# --------------------------------------------------
# Convert solution to 2D array
# --------------------------------------------------

u_arr = u.getArray().reshape((ny, nx))

# --------------------------------------------------
# Heatmap plot
# --------------------------------------------------

plt.figure(figsize=(6, 5))
plt.imshow(
    u_arr,
    origin="lower",
    extent=[0.0, length_x, 0.0, length_y],
    aspect="auto"
)
plt.colorbar(label="u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Steady Advection Solution")
plt.tight_layout()
plt.savefig("2d-steady-advection.png", dpi=150)
plt.close()

print("Saved: 2d-steady-advection.png")

# --------------------------------------------------
# 3D surface plot
# --------------------------------------------------

X, Y = np.meshgrid(
    np.linspace(0.0, length_x, nx),
    np.linspace(0.0, length_y, ny),
    indexing="xy"
)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, u_arr,
    cmap="magma",
    edgecolor="none",
    alpha=0.9
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("2D Advection Surface")

fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)

plt.tight_layout()
plt.savefig("2d-steady-advection-surface.png", dpi=150)
plt.close()

print("Saved: 2d-steady-advection-surface.png")

# --------------------------------------------------
# Centerline plot
# --------------------------------------------------

mid_j = ny // 2
centerline = u_arr[mid_j, :]

plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0.0, length_x, nx), centerline, lw=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Centerline y = {mid_j * dy:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2d-steady-advection-centerline.png", dpi=150)
plt.close()

print("Saved: 2d-steady-advection-centerline.png")
