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

# Import utilities (Assumed available as per your initial code)
from AI4PDEs_utils import get_weights_linear_2D

dt = 0.05
dx = 1.0 ; 
dy = 1.0
nu = 0.5 # Viscosity (Re in your previous code)
nx = 256 ;
ny = 256
ntime = 100

# --- 2. Stencils ---
[w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(dx)
w1 = w1.to(device) # Diffusion
w2 = w2.to(device) # X-Advection
w3 = w3.to(device) # Y-Advection
Y, X = torch.meshgrid(torch.arange(ny, device=device)*dy, torch.arange(nx, device=device)*dx, indexing="ij")

# --- 3. Physics Model (Burgers) ---
class AI4Burgers(nn.Module):
    def __init__(self):
        super(AI4Burgers, self).__init__()
        self.xadv = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.yadv = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.diff = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1

    def forward(self, u, u_vel):
        # Using replicate padding to handle shock behavior at boundaries
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        # Burgers Term: nu*Laplacian(u) - (u*du/dx + u*du/dy)
        # Linearized by using u_vel (u from previous step)
        Lu = nu * self.diff(u_pad) - u_vel * self.xadv(u_pad) - u_vel * self.yadv(u_pad)
        return Lu

def UNIT_TEST(c):
    mass = torch.sum(c)*dx*dy
    if mass < 1e-5: return 0.0, 0.0, 0.0, 0.0
    x_com = torch.sum(c*X)*dx*dy/mass
    y_com = torch.sum(c*Y)*dx*dy/mass
    var = torch.sum(c*((X-x0)**2+(Y-y0)**2))*dx*dy/mass # Using fixed x0, y0 for variance
    return mass.item(), x_com.item(), y_com.item(), var.item()

model = AI4Burgers().to(device)

# --- 4. PETSc Operator ---
class PETScOperator:
    def __init__(self, model, dt, shape):
        self.model = model
        self.dt = dt
        self.ny, self.nx = shape
        self.u_prev = None # Linearization velocity field

    def mult(self, mat, x, y):
        x_np = x.getArray(readonly=True)
        x_torch = torch.from_numpy(x_np.copy()).float().view(1, 1, self.ny, self.nx).to(device)
        
        with torch.no_grad():
            # Apply linearized Burgers operator
            Lu = self.model(x_torch, self.u_prev)
            res = x_torch - self.dt * Lu
            
        y.array = res.cpu().numpy().flatten()

# --- 5. Main Simulation ---
x0, y0 = nx*dx/2, ny*dy/2 # Defined globally for UNIT_TEST

def run_simulation():
    # Initialization (Back to SQUARE block)
    a = 25.0
    values_u = torch.zeros((1, 1, ny, nx), device=device)
    values_u[0, 0, (torch.abs(X-x0) <= a) & (torch.abs(Y-y0) <= a)] = 1.0

    initial_array = values_u.cpu().numpy().flatten().copy()
    
    sol_vec = PETSc.Vec().createWithArray(initial_array)
    rhs_vec = sol_vec.duplicate()
    
    ctx = PETScOperator(model, dt, (ny, nx))
    A = PETSc.Mat().createPython([nx*ny, nx*ny], context=ctx)
    A.setUp()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('gmres') 
    ksp.setTolerances(rtol=1e-6)

    print(f"{'Step':<5} | {'Residual':<10} | {'Step Time':<10} | {'Mass':<10} | {'Max_Val':<10}")
    print("-" * 60)

    mass, _, _, _ = UNIT_TEST(values_u)
    print(f"{0:<5} | {'---':<10} | {'---':<10} | {mass:<10.2f} | {values_u.max():<10.2f}")

    current_u = values_u 
    total_start = time.time()
    frames = []

    for t in range(1, ntime + 1):
        # 1. Update linearization velocity with current solution
        ctx.u_prev = current_u
        
        # 2. Setup RHS and solve
        sol_vec.copy(rhs_vec)
        step_start = time.time()
        ksp.solve(rhs_vec, sol_vec) 
        step_elapsed = time.time() - step_start
        
        # 3. Metrics and Frame capture
        current_u = torch.from_numpy(sol_vec.array).view(1, 1, ny, nx).to(device)
        frames.append(current_u.cpu().numpy()[0,0])
        mass, _, _, _ = UNIT_TEST(current_u)
        
        if t % 5 == 0 or t == 1:
            print(f"{t:<5} | {ksp.getResidualNorm():<10.2e} | {step_elapsed:<10.4f} | {mass:<10.2f} | {current_u.max():<10.2f}")

    total_elapsed = time.time() - total_start
    print("-" * 60)
    print(f"Total Elapsed Time: {total_elapsed:.4f} seconds")
    
    # Visualization and Animation
    fig, ax = plt.subplots(figsize=(6,5))
    img = ax.imshow(frames[0], cmap='jet', origin='lower')
    ax.set_title("Implicit Burgers' Equation (Square Init)")
    plt.colorbar(img)

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=frames[::2], interval=50, blit=True)
    ani.save("burgers_solution.gif", writer="pillow", fps=20)
    plt.show()

if __name__ == "__main__":
    run_simulation()

