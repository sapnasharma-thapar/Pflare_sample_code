import numpy as np
from petsc4py import PETSc
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------- Parameters (IDENTICAL) ----------------
nx, ny = 256, 64
dx = 1.0
dy = 1.0
dt = 0.005
nu = 0.1
ntime = 2000
n_check = 200
ub = -1.0

N = nx * ny

def idx(i, j):
    return j * nx + i

# ------------------------------------------------
# Laplacian (pressure Poisson)
# Right boundary Dirichlet
# Elsewhere standard 5-point
# ------------------------------------------------
def build_laplacian():
    A = PETSc.Mat().createAIJ([N, N])
    A.setPreallocationNNZ(5)
    A.setUp()

    for j in range(ny):
        for i in range(nx):

            row = idx(i, j)

            # Pressure Dirichlet at right boundary
            if i == nx - 1:
                A.setValue(row, row, 1.0)
                continue

            diag = -2/(dx*dx) - 2/(dy*dy)
            A.setValue(row, row, diag)

            if i > 0:
                A.setValue(row, idx(i-1,j), 1/(dx*dx))
            if i < nx-1:
                A.setValue(row, idx(i+1,j), 1/(dx*dx))
            if j > 0:
                A.setValue(row, idx(i,j-1), 1/(dy*dy))
            if j < ny-1:
                A.setValue(row, idx(i,j+1), 1/(dy*dy))

    A.assemble()
    return A

# ------------------------------------------------
# Diffusion matrix
# ------------------------------------------------
def build_diffusion(L):
    I = PETSc.Mat().createAIJ([N, N])
    I.setPreallocationNNZ(1)
    I.setUp()

    for i in range(N):
        I.setValue(i, i, 1.0)

    I.assemble()

    A = I.copy()
    A.axpy(-nu*dt, L)
    A.assemble()
    return A

def build_solver(A):
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('gmres')
    ksp.getPC().setType('gamg')
    ksp.setTolerances(rtol=1e-8)
    return ksp

# ------------------------------------------------
# Build matrices
# ------------------------------------------------
L = build_laplacian()
A_diff = build_diffusion(L)

ksp_p = build_solver(L)
ksp_diff = build_solver(A_diff)

# ------------------------------------------------
# Initialize fields (same as notebook)
# ------------------------------------------------
u = PETSc.Vec().createSeq(N)
v = PETSc.Vec().createSeq(N)
p = PETSc.Vec().createSeq(N)

u.set(-ub)
v.set(0.0)
p.set(0.0)

# ------------------------------------------------
# Solid body (32x32 same placement)
# ------------------------------------------------
sigma = np.zeros((ny, nx))
cor_x = nx // 4
cor_y = ny // 2 + 5
size = 32

for j in range(cor_y - size//2, cor_y + size//2):
    for i in range(cor_x - size//2, cor_x + size//2):
        if 0 <= i < nx and 0 <= j < ny:
            sigma[j,i] = 1e5

sigma_flat = sigma.flatten()

# print("Starting solver (Notebook-matching BCs)...")
start = time.time()

# ==========================================================
# TIME LOOP
# ==========================================================
frames = []

print("Starting solver with animation...")
start = time.time()

for t in range(1, ntime+1):

    u_np = np.array(u.getArray()).reshape(ny, nx)
    v_np = np.array(v.getArray()).reshape(ny, nx)

    # -------------------------------
    # Boundary Conditions (MATCH NOTEBOOK)
    # -------------------------------

    # Inlet (left)
    u_np[:,0] = -ub
    v_np[:,0] = 0.0

    # Top & Bottom
    # v = 0
    v_np[0,:] = 0.0
    v_np[-1,:] = 0.0

    # du/dy = 0  (copy interior value)
    u_np[0,:] = u_np[1,:]
    u_np[-1,:] = u_np[-2,:]

    # ------------------------------------------------
    # Advection (central difference - same as before)
    # ------------------------------------------------
    u_adv = u_np.copy()
    v_adv = v_np.copy()

    for j in range(1, ny-1):
        for i in range(1, nx-1):

            dudx = (u_np[j,i+1] - u_np[j,i-1])/(2*dx)
            dudy = (u_np[j+1,i] - u_np[j-1,i])/(2*dy)

            dvdx = (v_np[j,i+1] - v_np[j,i-1])/(2*dx)
            dvdy = (v_np[j+1,i] - v_np[j-1,i])/(2*dy)

            u_adv[j,i] = u_np[j,i] - dt*(u_np[j,i]*dudx + v_np[j,i]*dudy)
            v_adv[j,i] = v_np[j,i] - dt*(u_np[j,i]*dvdx + v_np[j,i]*dvdy)

    # ------------------------------------------------
    # Diffusion
    # ------------------------------------------------
    u_tmp = PETSc.Vec().createSeq(N)
    v_tmp = PETSc.Vec().createSeq(N)
    u_tmp.setArray(u_adv.flatten())
    v_tmp.setArray(v_adv.flatten())

    u_star = u.duplicate()
    v_star = v.duplicate()

    ksp_diff.solve(u_tmp, u_star)
    ksp_diff.solve(v_tmp, v_star)

    # ------------------------------------------------
    # Penalization
    # ------------------------------------------------
    for k in range(N):
        u_star[k] /= (1 + dt*sigma_flat[k])
        v_star[k] /= (1 + dt*sigma_flat[k])

    # ------------------------------------------------
    # Pressure Projection
    # ------------------------------------------------
    div = PETSc.Vec().createSeq(N)
    div.set(0.0)

    u_star_np = np.array(u_star.getArray()).reshape(ny, nx)
    v_star_np = np.array(v_star.getArray()).reshape(ny, nx)

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            dudx = (u_star_np[j,i+1] - u_star_np[j,i-1])/(2*dx)
            dvdy = (v_star_np[j+1,i] - v_star_np[j-1,i])/(2*dy)
            div[idx(i,j)] = (dudx + dvdy)/dt

    # Pressure Dirichlet at outlet
    for j in range(ny):
        div[idx(nx-1,j)] = 0.0

    ksp_p.solve(div, p)

    p_np = np.array(p.getArray()).reshape(ny, nx)

    for j in range(1, ny-1):
        for i in range(1, nx-1):

            dpdx = (p_np[j,i+1] - p_np[j,i-1])/(2*dx)
            dpdy = (p_np[j+1,i] - p_np[j-1,i])/(2*dy)

            u_np[j,i] = u_star_np[j,i] - dt*dpdx
            v_np[j,i] = v_star_np[j,i] - dt*dpdy

    # Penalize again
    u_np[sigma > 0] = 0.0
    v_np[sigma > 0] = 0.0

    u.setArray(u_np.flatten())
    v.setArray(v_np.flatten())
    # Save every 20th frame
    if t % 20 == 0:
        vel_mag = np.sqrt(u_np**2 + v_np**2)
        frames.append(vel_mag.copy())

    if t % n_check == 0:
        its = ksp_p.getIterationNumber()
        res = ksp_p.getResidualNorm()
        print(f"Step {t} | Iter {its} | Residual {res:.3e}")

print(f"Finished in {time.time()-start:.2f} sec")

# ============================================
# CREATE ANIMATION
# ============================================
fig, ax = plt.subplots(figsize=(12,4))
im = ax.imshow(frames[0], origin='lower', cmap='viridis')
plt.colorbar(im)

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=50)

ani.save("flow_animation.gif", writer="pillow", fps=20)

print("Animation saved as flow_animation.mp4")


# ==========================================================
# VISUALIZATION (Same plots as notebook)
# ==========================================================

u_np = np.array(u.getArray()).reshape(ny, nx)
v_np = np.array(v.getArray()).reshape(ny, nx)

vel_mag = np.sqrt(u_np**2 + v_np**2)

omega = np.zeros_like(u_np)
for j in range(1, ny-1):
    for i in range(1, nx-1):
        dvdx = (v_np[j,i+1] - v_np[j,i-1])/(2*dx)
        dudy = (u_np[j+1,i] - u_np[j-1,i])/(2*dy)
        omega[j,i] = dvdx - dudy

plt.figure(figsize=(12,4))
plt.imshow(u_np, origin='lower')
plt.colorbar()
plt.title("u velocity")
plt.savefig("u_vel.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
plt.imshow(v_np, origin='lower')
plt.colorbar()
plt.title("v velocity")
plt.savefig("v_vel.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
plt.imshow(vel_mag, origin='lower')
plt.colorbar()
plt.title("Velocity magnitude")
plt.savefig("velocity_magnitude.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
plt.imshow(omega, origin='lower')
plt.colorbar()
plt.title("Vorticity")
plt.savefig("vorticity.png", dpi=300)
plt.close()

print("All plots saved.")

