import os
from pathlib import Path
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to generate target states for chemotaxis PDE-constrained optimization
# -----------------------------------------------------------------=----------

"""
Solves the adjoint equations to the chemotaxis FCT problem
 -dp/dt + ∇⋅(-Dm*∇p) - X*(1 - η*u)*exp(-ηu)*∇p⋅∇v =  û - u + c*q        in Ω x [0,T]
      -dq/dt + ∇⋅(-Df*∇q + X*u*exp(-ηu)*∇p) + δ*q = v̂ - v        in Ω x [0,T]
                                      ∇p⋅n = ∇q⋅n = 0          on ∂Ω x [0,T]
                                              p(T) = 0  in Ω
                                              q(T) = 0  in Ω
with given û, v̂ and
    - u = 0.8*û
    - v = 0.8*v̂
    - c = 100                                        
(X = chi)
Parameters considered:
  Diffusion parameters: Dm = Df = 0.05
  Chemotaxis parameters: X = 0.25, η = 0.5
  Decay rate of f: δ = 100
  Growth rate of m: γ = 100
"""

# ---------------------------- General Parameters ----------------------------

a1, a2 = 0, 1
dx = 0.1 # 0.005 # Element size
intervals = round((a2 - a1) / dx)
print('here')
dt = 0.001
T = round(100*dt,2)
num_steps = round(T / dt)

show_plots = True # Toggle for visualization
optim = "alltime"

# ---------------------------- PDE Coefficients ------------------------------

delta, Dm, Df, chi, c_gamma, eta = hp.get_chtxs_sys_params()
constant_gamma = df.Constant(c_gamma)

# ---------------------------- Output File Path ------------------------------

output_dir = Path("Chtxs_adjoints")
output_dir.mkdir(parents=True, exist_ok=True)
output_filename_p = output_dir / f"chtxs_p_t{T}.csv"
output_filename_q = output_dir / f"chtxs_q_t{T}.csv"

# ----------------------- Initialize Finite Element Mesh ---------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals,
                        diagonal="right")
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
w = df.TestFunction(V)

# Create connectivities between vertices to find neighboring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

# ------------------------------ Solve PDEs ----------------------------------

vec_length = (num_steps + 1) * nodes
## choose target states as true solutions
target_data_path = f"Chtxs_data_T100_dx{dx}_dt{dt}"
target_file_u = os.path.join(target_data_path, "chtxs_m_t0.5.csv")
target_file_v = os.path.join(target_data_path, "chtxs_f_t0.5.csv")
# _, uhat = hp.import_data_final(target_file_u, nodes, vertex_to_dof,
#                                           num_steps=num_steps, time_dep=True)
# _, vhat = hp.import_data_final(target_file_v, nodes, vertex_to_dof,
#                                           num_steps=num_steps, time_dep=True)

# ## interpolated target states
# _, uhat_T = hp.import_data_final("chtxs_m_t0.5.csv", nodes, vertex_to_dof)
# _, vhat_T = hp.import_data_final("chtxs_f_t0.5.csv", nodes, vertex_to_dof)
# sqnodes = round(np.sqrt(nodes))
# uhat = np.zeros((num_steps+1)*nodes)
# vhat = np.zeros((num_steps+1)*nodes)
# for i in range(num_steps+1): # includes states at time zero
#     start = i*nodes
#     end = (i+1)*nodes
#     uhat[start:end] = i*dt /T * uhat_T
#     vhat[start:end] = i*dt /T * vhat_T

uhat = np.ones(vec_length)
vhat = 2*np.ones(vec_length)

uk = 0.8*uhat
vk = 0.8*vhat
control= 100*np.ones(vec_length)
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
print('here3')

pk, qk = hp.solve_adjoint_chtxs_system(uk, vk, uhat, vhat, pk, qk, control, T, V, 
                                nodes, num_steps, dt, dof_neighbors, optim,
                                show_plots=show_plots, vertex_to_dof=vertex_to_dof,
                                out_folder=output_dir, mesh=mesh,  deltax=dx)

pk.tofile(output_filename_p, sep=",")
qk.tofile(output_filename_q, sep=",")