import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from AI4PDEs_utils import get_weights_linear_2D, create_solid_body_2D

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# PARAMETERS
# ============================================================

dt = 0.05
dx = 1.0
nu = 0.05
ub = -1.0

nx, ny = 256, 64
ntime = 2000
N = nx * ny

os.makedirs("hybrid_results", exist_ok=True)

[w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(dx)

# ============================================================
# PETSc MATRIX (USE TRUE LAPLACIAN WITH CORRECT SIGN)
# ============================================================

A = PETSc.Mat().createAIJ([N, N], nnz=9)
A.setUp()

# IMPORTANT: Flip sign because w1 = -∇²
kernel = -w1[0,0].cpu().numpy()

for j in range(ny):
    for i in range(nx):

        row = j * nx + i

        for dj in [-1,0,1]:
            for di in [-1,0,1]:

                ni = i + di
                nj = j + dj

                weight = kernel[dj+1, di+1]

                if 0 <= ni < nx and 0 <= nj < ny:
                    col = nj * nx + ni
                    A[row, col] = weight

A.assemble()

nullspace = PETSc.NullSpace().create(constant=True)
A.setNullSpace(nullspace)

# ============================================================
# KSP
# ============================================================

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("gmres")
ksp.getPC().setType("gamg")
ksp.setTolerances(rtol=1e-6)
ksp.setFromOptions()

b_vec = PETSc.Vec().createSeq(N)
x_vec = PETSc.Vec().createSeq(N)

# ============================================================
# CNN OPERATORS
# ============================================================

class Operators(nn.Module):
    def __init__(self):
        super().__init__()
        self.ddx = nn.Conv2d(1,1,3,padding=1,bias=False)
        self.ddy = nn.Conv2d(1,1,3,padding=1,bias=False)
        self.lap = nn.Conv2d(1,1,3,padding=1,bias=False)

        self.ddx.weight.data = w2
        self.ddy.weight.data = w3
        self.lap.weight.data = w1

    def solid_body(self,u,v,sigma,dt):
        return u/(1+dt*sigma), v/(1+dt*sigma)

model = Operators().to(device)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

u = torch.zeros((1,1,ny,nx), device=device)
v = torch.zeros((1,1,ny,nx), device=device)
u[:,:,:,0] = ub

sigma = create_solid_body_2D(
    nx, ny,
    int(nx/4), int(ny/2)+5,
    int(ny/4), int(ny/4)
).to(device)

indices = np.arange(N, dtype=np.int32)

# ============================================================
# SIMULATION
# ============================================================

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():

    for t in range(1, ntime+1):

        ADx_u = model.ddx(u)
        ADy_u = model.ddy(u)
        ADx_v = model.ddx(v)
        ADy_v = model.ddy(v)
        AD2_u = model.lap(u)
        AD2_v = model.lap(v)

        u_star = u + dt*(nu*AD2_u - u*ADx_u - v*ADy_u)
        v_star = v + dt*(nu*AD2_v - u*ADx_v - v*ADy_v)

        u_star, v_star = model.solid_body(u_star,v_star,sigma,dt)

        # Projection RHS
        div = -(model.ddx(u_star) + model.ddy(v_star)) / dt
        div_np = div[0,0].cpu().numpy().flatten()

        # Compatibility condition
        div_np -= np.mean(div_np)

        b_vec.setValues(indices, div_np)
        b_vec.assemble()

        ksp.solve(b_vec, x_vec)

        p_np = x_vec.getArray(readonly=True)
        p = torch.from_numpy(
            p_np.reshape(ny,nx).copy()
        ).float().to(device).unsqueeze(0).unsqueeze(0)

        p -= torch.mean(p)

        # Velocity correction
        u = u_star - dt*model.ddx(p)
        v = v_star - dt*model.ddy(p)

        u, v = model.solid_body(u,v,sigma,dt)

        # Boundary conditions
        u[:,:,:,0] = ub
        v[:,:,:,0] = 0.0
        u[:,:,:,-1] = u[:,:,:,-2]
        v[:,:,:,-1] = v[:,:,:,-2]
        u[:,:,0,:] = 0.0
        u[:,:,-1,:] = 0.0
        v[:,:,0,:] = 0.0
        v[:,:,-1,:] = 0.0

        if t % 200 == 0:
            print(f"Step {t} | Residual {ksp.getResidualNorm():.3e}")

torch.cuda.synchronize()
end = time.time()

print("Runtime:", end-start)

# ============================================================
# PLOTTING
# ============================================================

u_plot = u.detach().cpu()[0,0]
v_plot = v.detach().cpu()[0,0]
sigma_plot = sigma.detach().cpu()[0,0]

# U velocity
plt.figure(figsize=(14,4))
plt.imshow(u_plot, origin='lower', cmap='jet')
plt.colorbar()
plt.contourf(sigma_plot.cpu(), levels=[0.5,1], colors='gray', alpha=0.7)
plt.title("u velocity")
plt.savefig("hybrid_results/u-petsc.png")
plt.close()

# V velocity
plt.figure(figsize=(14,4))
plt.imshow(v_plot, origin='lower', cmap='jet')
plt.colorbar()
plt.contourf(sigma_plot.cpu(), levels=[0.5,1], colors='gray', alpha=0.7)
plt.title("v velocity")
plt.savefig("hybrid_results/v-petsc.png")
plt.close()

print("Plots saved successfully.")
