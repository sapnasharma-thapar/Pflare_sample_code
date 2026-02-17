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
import pflare

from AI4PDEs_utils import get_weights_linear_2D, create_solid_body_2D

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)

# ============================================================
# PARAMETERS
# ============================================================

dt = 0.005
dx = dy = 1.0
nu = 0.1
ub = -1.0

nx, ny = 256, 64
ntime = 2000
N = nx * ny

filepath = "hybrid_results"
os.makedirs(filepath, exist_ok=True)

[w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(dx)

# ============================================================
# PETSc MATRIX (Backend decided from terminal)
# ============================================================

A = PETSc.Mat().create()
A.setSizes([N, N])
A.setPreallocationNNZ(5)
A.setFromOptions()
A.setUp()

for j in range(ny):
    for i in range(nx):
        r = j * nx + i
        A[r, r] = 4.0
        if i > 0:      A[r, r-1]   = -1.0
        if i < nx-1:   A[r, r+1]   = -1.0
        if j > 0:      A[r, r-nx]  = -1.0
        if j < ny-1:   A[r, r+nx]  = -1.0

A.assemble()

nullspace = PETSc.NullSpace().create(constant=True)
A.setNullSpace(nullspace)

# ============================================================
# KSP + AIR
# ============================================================

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("gmres")

pc = ksp.getPC()
pc.setType("air")

ksp.setTolerances(rtol=1e-6)
ksp.setFromOptions()

# ============================================================
# VECTORS 
# ============================================================

b_vec = PETSc.Vec().create()
b_vec.setSizes(N)
b_vec.setFromOptions()
b_vec.setUp()

x_vec = PETSc.Vec().create()
x_vec.setSizes(N)
x_vec.setFromOptions()
x_vec.setUp()

print("PETSc configured: GMRES + AIR")

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

# ============================================================
# SIMULATION
# ============================================================

residuals = []
gmres_iterations = []

indices = np.arange(N, dtype=np.int32)

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():

    for t in range(1, ntime+1):

        # ----- Advection + Diffusion -----

        ADx_u = model.ddx(u)
        ADy_u = model.ddy(u)
        ADx_v = model.ddx(v)
        ADy_v = model.ddy(v)
        AD2_u = model.lap(u)
        AD2_v = model.lap(v)

        u_star = u + dt*(nu*AD2_u - u*ADx_u - v*ADy_u)
        v_star = v + dt*(nu*AD2_v - u*ADx_v - v*ADy_v)

        u_star, v_star = model.solid_body(u_star,v_star,sigma,dt)

        # ----- Pressure RHS -----

        div = -(model.ddx(u_star)+model.ddy(v_star))/dt

        if torch.isnan(div).any():
            print("NaN detected at step", t)
            break

        div_np = div[0,0].detach().cpu().numpy().flatten()

        # 🔥 GPU SAFE VECTOR FILL
        b_vec.set(0.0)
        b_vec.setValues(indices, div_np)
        b_vec.assemble()

        ksp.solve(b_vec, x_vec)

        gmres_iterations.append(ksp.getIterationNumber())
        residuals.append(ksp.getResidualNorm())

        # Copy pressure back
        p_np = x_vec.getArray(readonly=True)

        p_tensor = torch.from_numpy(
            p_np.reshape(ny,nx).copy()   # avoid warning
        ).float().to(device).unsqueeze(0).unsqueeze(0)

        p_tensor -= torch.mean(p_tensor)

        # ----- Velocity correction -----

        u = u_star - dt*model.ddx(p_tensor)
        v = v_star - dt*model.ddy(p_tensor)

        u,v = model.solid_body(u,v,sigma,dt)

        # ----- Boundary Conditions -----

        u[:,:,:,0] = ub
        v[:,:,:,0] = 0.0

        u[:,:,:,-1] = u[:,:,:,-2]
        v[:,:,:,-1] = v[:,:,:,-2]

        u[:,:,0,:] = 0.0
        u[:,:,-1,:] = 0.0
        v[:,:,0,:] = 0.0
        v[:,:,-1,:] = 0.0

        if t % 200 == 0:
            print(f"Step {t} | Residual {ksp.getResidualNorm():.6e} | GMRES iters {ksp.getIterationNumber()}")

torch.cuda.synchronize()
end = time.time()

print("Total runtime:", end-start)
print("Average GMRES iterations:", np.mean(gmres_iterations))
print("Simulation complete.")
