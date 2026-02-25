import os
import numpy as np
import time
import torch
import torch.nn as nn
from petsc4py import PETSc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from AI4PDEs_utils import create_tensors_2D, get_weights_linear_2D, create_solid_body_2D
from AI4PDEs_bounds import boundary_condition_2D_u
from AI4PDEs_bounds import boundary_condition_2D_v
from AI4PDEs_bounds import boundary_condition_2D_p


# ---------------- Parameters ----------------
dt = 0.005
dx = 1.0
nu = 0.1
ub = -1.0
nx = 256
ny = 64
ntime = 5000
n_check = 100

[w1, w2, w3, wA, _, _] = get_weights_linear_2D(dx)


# ============================================================
#                       MODEL
# ============================================================

class AI4CFD(nn.Module):

    def __init__(self):
        super().__init__()

        self.dt = dt
        self.nu = nu

        # CNN operators
        self.xadv = nn.Conv2d(1,1,3,1,0,bias=False).to(device)
        self.yadv = nn.Conv2d(1,1,3,1,0,bias=False).to(device)
        self.diff = nn.Conv2d(1,1,3,1,0,bias=False).to(device)

        self.xadv.weight.data = w2.to(device)
        self.yadv.weight.data = w3.to(device)
        self.diff.weight.data = w1.to(device)

        # PETSc solver (robust choice)
        self.ksp = PETSc.KSP().create()
        self.ksp.setType('gmres')
        self.ksp.getPC().setType('gamg')
        self.ksp.setTolerances(rtol=1e-6)

        self._build_matrix()


    # --------------------------------------------------------
    # Build Laplacian matrix A (no zeroRows)
    # --------------------------------------------------------
    def _build_matrix(self):

        N = nx * ny
        stencil = wA.detach().cpu().numpy()[0,0]

        A = PETSc.Mat().createAIJ([N, N])
        A.setPreallocationNNZ(9)
        A.setUp()

        for j in range(ny):
            for i in range(nx):

                row = j * nx + i

                # Dirichlet pressure at right boundary
                if i == nx - 1:
                    A.setValue(row, row, 1.0)
                    continue

                for sj in range(-1,2):
                    for si in range(-1,2):

                        ii = i + si
                        jj = j + sj
                        val = stencil[sj+1, si+1]

                        if val == 0:
                            continue

                        # Neumann boundaries (simple reflection)
                        if ii < 0: ii = 1
                        if jj < 0: jj = 1
                        if jj >= ny: jj = ny - 2
                        if ii >= nx: continue

                        col = jj * nx + ii
                        A.setValue(row, col, val)

        A.assemble()
        self.A = A
        self.ksp.setOperators(A)


    # --------------------------------------------------------
    # Solid body penalization
    # --------------------------------------------------------
    def solid_body(self, u, v, sigma):
        u = u / (1 + self.dt * sigma)
        v = v / (1 + self.dt * sigma)
        return u, v
    
        # --------------------------------------------------------
    # Print PETSc Matrix A
    # --------------------------------------------------------
    def print_matrix(self, rows_to_print=5, save_to_file=False, filename="A_matrix.txt"):
        """
        Print PETSc sparse matrix information and selected rows.
        """

        print("\n================ MATRIX INFO ================")
        print("Matrix size:", self.A.getSize())
        print("Total nonzeros:", int(self.A.getInfo()['nz_used']))
        print("============================================\n")

        nrows = self.A.getSize()[0]
        rows_to_print = min(rows_to_print, nrows)

        for row in range(rows_to_print):
            cols, vals = self.A.getRow(row)

            print(f"Row {row}:")
            for c, v in zip(cols, vals):
                print(f"  col {c:6d}  value {v: .6e}")

            print("--------------------------------------------")

        # Optional: Save full matrix
        if save_to_file:
            viewer = PETSc.Viewer().createASCII(filename)
            self.A.view(viewer)
            viewer.destroy()
            print(f"\nFull matrix saved to {filename}")


    # --------------------------------------------------------
    # Pressure Solve
    # --------------------------------------------------------
    def solve_pressure(self, uu, vv, p):

        N = nx * ny

        div_u = self.xadv(uu) + self.yadv(vv)
        b = -div_u / self.dt

        b_np = b.detach().cpu().numpy().reshape(N)
        p_np = p.detach().cpu().numpy().reshape(N)

        # Enforce RHS = 0 at Dirichlet boundary
        for j in range(ny):
            row = j * nx + (nx - 1)
            b_np[row] = 0.0

        b_vec = PETSc.Vec().createWithArray(b_np)
        p_vec = PETSc.Vec().createWithArray(p_np)

        self.ksp.solve(b_vec, p_vec)

        p_new = p_vec.getArray().reshape(ny,nx)

        return torch.from_numpy(p_new)\
                    .float()\
                    .to(device)\
                    .unsqueeze(0)\
                    .unsqueeze(0)


    # --------------------------------------------------------
    # Forward Step
    # --------------------------------------------------------
    def forward(self, u, uu, v, vv, p, pp, sigma):

        # Apply velocity BC
        uu = boundary_condition_2D_u(u, uu, ub)
        vv = boundary_condition_2D_v(v, vv, ub)

        # Advection-diffusion
        ADx_u = self.xadv(uu)
        ADy_u = self.yadv(uu)
        AD2_u = self.diff(uu)

        ADx_v = self.xadv(vv)
        ADy_v = self.yadv(vv)
        AD2_v = self.diff(vv)

        u_star = u + self.nu*AD2_u*self.dt - u*ADx_u*self.dt - v*ADy_u*self.dt
        v_star = v + self.nu*AD2_v*self.dt - u*ADx_v*self.dt - v*ADy_v*self.dt

        # Solid body
        u_star, v_star = self.solid_body(u_star, v_star, sigma)

        # Reapply BC
        uu = boundary_condition_2D_u(u_star, uu, ub)
        vv = boundary_condition_2D_v(v_star, vv, ub)

        # Pressure solve
        p = self.solve_pressure(uu, vv, p)

        # Ghost pressure for gradient
        pp = boundary_condition_2D_p(p, pp)

        # Velocity correction
        u = u_star - self.xadv(pp)*self.dt
        v = v_star - self.yadv(pp)*self.dt

        # Solid body again
        u, v = self.solid_body(u, v, sigma)

        return u, v, p


# ============================================================
# INITIALIZATION
# ============================================================

model = AI4CFD().to(device)
model.print_matrix(rows_to_print=10)
model.print_matrix(rows_to_print=5, save_to_file=True)

u, v, p, uu, vv, pp, _, _ = create_tensors_2D(nx, ny)

u = u.to(device)
v = v.to(device)
p = p.to(device)
uu = uu.to(device)
vv = vv.to(device)
pp = pp.to(device)

# Laminar inlet
u.fill_(-ub)
v.fill_(0.0)
p.fill_(0.0)

# Bluff body
cor_x = int(nx/4)
cor_y = int(ny/2) + 5
size_x = int(ny/4)
size_y = int(ny/4)

sigma = create_solid_body_2D(nx, ny, cor_x, cor_y, size_x, size_y).to(device)

print("Starting solver...")
start = time.time()

with torch.no_grad():
    for t in range(1, ntime+1):

        u, v, p = model(u, uu, v, vv, p, pp, sigma)

        if t % n_check == 0:
            its = model.ksp.getIterationNumber()
            res = model.ksp.getResidualNorm()
            print(f"Step {t} | Iterations: {its} | Residual: {res:.3e}")

print(f"Simulation complete in {time.time()-start:.2f} seconds.")


dvdx = model.xadv(boundary_condition_2D_v(v, vv, ub))
dudy = model.yadv(boundary_condition_2D_u(u, uu, ub))

omega = dvdx - dudy
omega_np = omega.detach().squeeze().cpu().numpy()

# ============================================================
# Save Plots
# ============================================================

u_np = u.squeeze().cpu().numpy()
v_np = v.squeeze().cpu().numpy()

plt.figure(figsize=(15,6))
plt.imshow(-u_np, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("u component velocity (m/s)")
plt.tight_layout()
plt.savefig("u_velocity_new.png", dpi=300)
plt.close()

plt.figure(figsize=(15,6))
plt.imshow(-v_np, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("v component velocity (m/s)")
plt.tight_layout()
plt.savefig("v_velocity_new.png", dpi=300)
plt.close()

print("Plots saved successfully.")
