from pathlib import Path
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to generate target states for chemotaxis PDE-constrained optimization
# -----------------------------------------------------------------=----------

"""
Solves the PDECO problem below with projected gradient descent method and FCT
Cost functional:
J(m,f,c) = 1/2*||m(T) - m̂_T||² + 1/2*||v(T) - fˆ_T||² + β/2*||c||²
(misfit: L²-norms over Ω,  regularization: L²-norm over Ω × [0,T])

min_{m,f,c} J(m,f,c)
subject to:  
  dm/dt + ∇⋅(-Dm*∇m + X*m*exp(-ηm)*∇f) = 0      in Ω × [0,T]
              df/dt + ∇⋅(-Df*∇f) + δ*f = c*m    in Ω × [0,T]
          (-Df*∇m + X*m*exp(-ηm)*∇f)⋅n = 0      on ∂Ω × [0,T]
                                 ∇f⋅n = 0       on ∂Ω × [0,T]
                                 m(0) = m0(x)    in Ω
                                 f(0) = f0(x)    in Ω
                                 c in [ca,cb]
(X = chi)
Parameters considered:
  Diffusion parameters: Dm = Df = 0.05
  Chemotaxis parameters: X = 0.25, η = 0.5
  Decay rate of f: δ = 100
  Growth rate of m: γ = 100
"""

# ---------------------------- General Parameters ----------------------------


a1, a2 = 0, 1
dx = 0.025 #0.025 # 0.005 # Element size
intervals = round((a2 - a1) / dx)

dt = 0.0005 
T = round(1000*dt,2)
num_steps = round(T / dt)

show_plots = True # Toggle for visualization

# ---------------------------- PDE Coefficients ------------------------------

delta, Dm, Df, chi, c_gamma, eta = hp.get_chtxs_sys_params()
constant_gamma = df.Constant(c_gamma)

# ---------------------------- Output File Path ------------------------------

output_dir = Path(f"Chtxs_data_T100_dx{dx}_dt{dt}")
output_dir.mkdir(parents=True, exist_ok=True)
output_filename_m = output_dir / f"chtxs_m_t{T}.csv"
output_filename_f = output_dir / f"chtxs_f_t{T}.csv"

# ----------------------- Initialize Finite Element Mesh ---------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
w = df.TestFunction(V)

# Create connectivities between vertices to find neighboring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

# ----------------------------- Initial Condition ----------------------------

m0, f0 = hp.chtxs_sys_IC(a1, a2, dx, nodes, vertex_to_dof)
z = np.zeros(nodes)

# ------------------------------ Solve PDEs ----------------------------------

# mT, fT = hp.solve_chtxs_system(
#     z, m0, f0, V, nodes, num_steps, dt, dof_neighbors,
#     control_fun=constant_gamma, show_plots=show_plots, vertex_to_dof=vertex_to_dof,
#     generation_mode=True, output_dir=output_dir)

vec_length = (num_steps + 1) * nodes
mvec = np.zeros(vec_length)
fvec = np.zeros(vec_length)
mvec[:nodes] = m0
fvec[:nodes] = f0

mvec, fvec = hp.solve_chtxs_system(
    z, mvec, fvec, V, nodes, num_steps, dt, dof_neighbors,
    control_fun=constant_gamma, show_plots=show_plots, vertex_to_dof=vertex_to_dof,
    generation_mode=False, output_dir=output_dir, rescaling=1)

mvec.tofile(output_filename_m, sep=",")
fvec.tofile(output_filename_f, sep=",")

# ------------------  L^2-norm of true control over Ω × [0,T] ----------------

# M = hp.assemble_sparse(u * w * df.dx)
# control_as_td_vector = c_gamma * np.ones(vec_length)
# control_norm = hp.L2_norm_sq_Q(control_as_td_vector, num_steps, dt, M)
# print(f"L^2-norm of the control over Ω × [0,{T}]:", control_norm)

# beta = 1e-1
# print(f"β/2 *||c_true|| in L^2-norm^2 over Ω × [0,{T}]:", beta/2*control_norm)

# eval_sim = 1/T * 1/((a2-a1)**2) * control_norm
# print(f"Average true control in L^2(Q)^2 over Ω × [0,{T}] :", eval_sim)




# ### version used before:

# ###############################################################################
# ################### Define the stationary matrices ###########################
# ###############################################################################

# # Mass matrix
# M = assemble_sparse_lil(u * v * dx)

# # Row-lumped mass matrix
# M_Lump = row_lump(M, nodes)

# # Stiffness matrix
# Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# # System matrix: equation for f
# Mat_f = M + dt * (Df * Ad + delta * M)

# ###############################################################################
# ######################## Initial conditions for m,f ###########################
# ###############################################################################

# m0_orig = mimura_data_helpers.m_initial_condition(a1, a2, deltax).reshape(nodes)
# f0_orig = m0_orig
# m0 = reorder_vector_to_dof_time(m0_orig, 1, nodes, vertextodof)
# f0 = reorder_vector_to_dof_time(f0_orig, 1, nodes, vertextodof)

# ###############################################################################
# ########################### Initial guesses for GD ############################
# ###############################################################################

# # Change to have smaller vectors of length nodes
# m_prev = m0
# f_prev = f0

# m_new = np.zeros(nodes)
# f_new = np.zeros(nodes)

# t = 0
# for i in range(1, num_steps + 1):    # solve for fk(t_{n+1}), mk(t_{n+1})
#     start = i * nodes
#     end = (i + 1) * nodes
#     t += dt
#     if i % 50 == 0:
#         print('t = ', round(t, 4))
        
#     m_n = m_prev # mk(t_n) 
#     m_n_fun = vec_to_function(m_n, V)
#     f_n_fun = vec_to_function(f_prev, V)
    
#     f_rhs = np.asarray(assemble(f_n_fun * v * dx  + dt * gamma * m_n_fun * v * dx))

#     f_new = spsolve(Mat_f, f_rhs)
    
#     f_np1_fun = vec_to_function(f_new, V)

#     A_m = mimura_data_helpers.mat_chtx_m(f_np1_fun, m_n_fun, Dm, chi, u, v)
#     m_rhs = np.zeros(nodes)
    
#     m_new = FCT_alg(A_m, m_rhs, m_prev, dt, nodes, M, M_Lump, dof_neighbors)    
#     # m_new = spsolve(M - dt*A_m, M@m_n + dt*m_rhs)
    
#     m_re = reorder_vector_from_dof_time(m_new, 1, nodes, vertextodof)
#     f_re = reorder_vector_from_dof_time(f_new, 1, nodes, vertextodof)

#     if show_plots is True and i % 20 == 0:
#         fig2 = plt.figure(figsize = (10, 5))
#         fig2.tight_layout(pad = 3.0)
#         ax2 = plt.subplot(1,2,1)
#         im1 = plt.imshow(m_re.reshape((sqnodes, sqnodes)))#, vmin = min_m, vmax = max_m)
#         fig2.colorbar(im1)
#         plt.title(f'Computed state $m$ at t = {round(t,5)}')
#         ax2 = plt.subplot(1,2,2)
#         im2 = plt.imshow(f_re.reshape((sqnodes, sqnodes)))#, vmin = min_f, vmax = max_f)
#         fig2.colorbar(im2)
#         plt.title(f'Computed state $f$ at t = {round(t,5)}')
#         plt.show()
        
#     m_new.tofile(out_folder_name + f'/chtx_m_t{round(t,4)}.csv', sep = ',')
#     f_new.tofile(out_folder_name + f'/chtx_f_t{round(t,4)}.csv', sep = ',')

#     m_prev = m_new
#     f_prev = f_new

# print(f'{T=}, {dt=}, {deltax=}, {chi=}, {Dm=}, {Df=}')
