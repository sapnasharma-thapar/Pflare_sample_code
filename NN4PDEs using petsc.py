import os
import numpy as np
import pandas as pd
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from petsc4py import PETSc

# --- 1. Setup ---
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

# Import utilities
from AI4PDEs_utils import get_weights_linear_2D

dt = 0.05
dx = 1.0 ; 
dy = 1.0
# Re = 0.0
Re = 0.5
ub = 1.0
nx = 256 ;
ny = 256
ntime = 100

# --- 2. Stencils ---
[w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(dx)
w1 = w1.to(device)
w2 = w2.to(device)
w3 = w3.to(device)
Y, X = torch.meshgrid(torch.arange(ny, device=device)*dy, torch.arange(nx, device=device)*dx, indexing="ij")

# --- 3. Physics Model ---
class AI4CFD(nn.Module):
    def __init__(self):
        super(AI4CFD, self).__init__()
        self.xadv = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.yadv = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.diff = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1

    def forward(self, u):
        u_pad = F.pad(u, (1, 1, 1, 1), mode='constant', value=0)
        Lu = Re * self.diff(u_pad) - ub * self.xadv(u_pad) - ub * self.yadv(u_pad)
        return Lu

    def apply_A(self, u):
        u_pad = F.pad(u, (1, 1, 1, 1), mode='constant', value=0)
        Lu = Re * self.diff(u_pad) - ub * self.xadv(u_pad) - ub * self.yadv(u_pad)
        return u - dt * Lu


def UNIT_TEST(c):
    mass = torch.sum(c)*dx*dy
    if mass < 1e-5: return 0.0, 0.0, 0.0, 0.0
    x_com = torch.sum(c*X)*dx*dy/mass
    y_com = torch.sum(c*Y)*dx*dy/mass
    var = torch.sum(c*((X-x_com)**2+(Y-y_com)**2))*dx*dy/mass
    return mass.item(), x_com.item(), y_com.item(), var.item()

model = AI4CFD().to(device)

# --- 4. PETSc Operator ---
class PETScOperator:
    def __init__(self, model, dt, shape):
        self.model = model
        self.dt = dt
        self.ny, self.nx = shape

    def mult(self, mat, x, y):
        x_np = x.getArray(readonly=True)
        # .copy() prevents PyTorch read-only error
        x_torch = torch.from_numpy(x_np.copy()).float().view(1, 1, self.ny, self.nx).to(device)
        
        with torch.no_grad():
            Lu = self.model(x_torch)
            res = x_torch - self.dt * Lu
            
        y.array = res.cpu().numpy().flatten()

# --- 5. Main Simulation ---
def run_simulation():
    # Initialization
    x0, y0, a = nx*dx/2, ny*dy/2, 25.0
    values_u = torch.zeros((1, 1, ny, nx), device=device)
    values_u[0, 0, (torch.abs(X-x0) <= a) & (torch.abs(Y-y0) <= a)] = 1.0

    # Explicit deep copy to numpy for PETSc
    initial_array = values_u.cpu().numpy().flatten().copy()
    
    # Create vectors
    sol_vec = PETSc.Vec().createWithArray(initial_array)
    rhs_vec = sol_vec.duplicate()
#     rhs_vec = A.createVecRight()
#     sol_vec = A.createVecLeft()
    
    ctx = PETScOperator(model, dt, (ny, nx))
    A = PETSc.Mat().createPython([nx*ny, nx*ny], context=ctx)
    A.setUp()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    # Use 'bcgs' (BiCGStab) to allow solving without A_transpose
    ksp.setType('gmres') 
    ksp.setTolerances(rtol=1e-6)

    print(f"{'Step':<5} | {'Residual':<10} | {'Step Time':<10} | {'Mass':<10} | {'Max_Val':<10}")
    print("-" * 60)

    mass, _, _, _ = UNIT_TEST(values_u)
    print(f"{0:<5} | {'---':<10} | {'---':<10} | {mass:<10.2f} | {values_u.max():<10.2f}")

    # Initialize current_u with initial state to prevent UnboundLocalError
    current_u = values_u 

    # --- START TOTAL TIMER ---
    total_start = time.time()
    frames = []
    for t in range(1, ntime + 1):
        # Load previous solution (sol_vec) into RHS (rhs_vec)
        sol_vec.copy(rhs_vec)
        
        step_start = time.time()
        ksp.solve(rhs_vec, sol_vec) 
        step_elapsed = time.time() - step_start
        
        # Metrics
        current_u = torch.from_numpy(sol_vec.array).view(1, 1, ny, nx).to(device)
        frames.append(current_u.cpu().numpy()[0,0])
        mass, _, _, _ = UNIT_TEST(current_u)
        
        print(f"{t:<5} | {ksp.getResidualNorm():<10.2e} | {step_elapsed:<10.4f} | {mass:<10.2f} | {current_u.max():<10.2f}")

    # --- END TOTAL TIMER ---
    total_end = time.time()
    total_elapsed = total_end - total_start

    print("-" * 60)
    print(f"Total Elapsed Time: {total_elapsed:.4f} seconds")
    print("-" * 60)
    
    # Animation

    fig, ax = plt.subplots(figsize=(6,5))
    img = ax.imshow(frames[0], cmap='jet', origin='lower')
    ax.set_title("Implicit Advection-Diffusion")

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=50,
        blit=True
    )

    ani.save("solution_animation.gif", writer="pillow", fps=20)
    plt.close()
    
    
    # Visualization
    plt.figure(figsize=(6, 5))
    plt.imshow(current_u.cpu().numpy()[0,0], cmap='jet', origin='lower')
    plt.colorbar(label='Concentration')
    plt.title(f"Implicit Solution (t={ntime*dt:.2f})")
    if Re>0.0:
        plt.savefig('solution_Re.png')
    else:
        plt.savefig('solution_pure.png')
    # plt.show()


# ---- BUILD EXPLICIT A MATRIX ----
nx_small = 3
ny_small = 3
N = nx_small * ny_small

A = torch.zeros((N, N), device=device)

def flatten(i, j):
    return i * nx_small + j

for i in range(ny_small):
    for j in range(nx_small):

        u = torch.zeros((1,1,ny_small,nx_small), device=device)
        u[0,0,i,j] = 1.0

        Au = model.apply_A(u)

        col = flatten(i,j)
        A[:, col] = Au.view(-1)

import numpy as np

A_np = A.detach().cpu().numpy()

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=150
)

print("\nExplicit A matrix:\n")
print(A_np)


if __name__ == "__main__":
    run_simulation()
