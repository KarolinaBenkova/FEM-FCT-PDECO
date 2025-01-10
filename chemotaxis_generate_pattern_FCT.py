from pathlib import Path

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from scipy.sparse.linalg import spsolve, gmres, minres, LinearOperator
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *
import mimura_data_helpers
# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the chemotaxis system
### with Flux-corrected transport method, norms over L^2
# min_{m,f,c} ( ||m(T)-\hat{m}||^2 + ||f(T)-\hat{f}||^2 + beta*||c||^2) / 2
# subject to:
#        dm/dt - Dm*grad^2(m) + div(chi*m*grad(f)) = m(4-m)       in Ωx[0,T]
#                  df/dt - Df*grad^2(f)) + delta*f = c*m          in Ωx[0,T]
#                                 zero  Neumann BC for m,f        on ∂Ωx[0,T]
#                            given initial conditions for m,f     in Ω

# Dm, Df, chi are parameters
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 10
deltax = 0.05*2*2
intervals_line = round((a2 - a1) / deltax)

delta = 1
Dm = 0.05
Df = 0.05
chi = 5 #0.17
gamma = 0.5#2

t0 = 0
dt = 0.1
T = 50 #5*dt #50
num_steps = round((T-t0)/dt)

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
v = TestFunction(V)

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

show_plots = True
out_folder_name = f"chtx_chi{chi}_simplfeathers_dx{deltax}_cos"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M, nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# System matrix: equation for f
Mat_f = M + dt * (Df * Ad + delta * M)

zeros = np.zeros(nodes)
###############################################################################
######################## Initial conditions for m,f ###########################
###############################################################################

m0_orig = mimura_data_helpers.m_initial_condition(a1, a2, deltax).reshape(nodes)
f0_orig = m0_orig #/ delta #m0_orig #1/32 * np.ones(nodes)
m0 = reorder_vector_to_dof_time(m0_orig, 1, nodes, vertextodof)
f0 = reorder_vector_to_dof_time(f0_orig, 1, nodes, vertextodof)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1) * nodes # include zero and final time
zeros_nt = np.zeros(vec_length)

mk = np.zeros(vec_length)
fk = np.zeros(vec_length)

mk[:nodes] = m0
fk[:nodes] = f0

t = 0
for i in range(1, num_steps + 1):    # solve for fk(t_{n+1}), mk(t_{n+1})
    start = i * nodes
    end = (i + 1) * nodes
    t += dt
    if i % 50 == 0:
        print('t = ', round(t, 4))
        
    m_n = mk[start - nodes : start]    # mk(t_n) 
    m_n_fun = vec_to_function(m_n,V)
    f_n_fun = vec_to_function(fk[start - nodes : start],V)
    
    f_rhs = np.asarray(assemble(f_n_fun * v * dx  + dt * gamma * m_n_fun * v * dx))

    fk[start : end] = spsolve(Mat_f, f_rhs)
    
    f_np1_fun = vec_to_function(fk[start : end], V)

    A_m = mimura_data_helpers.mat_chtx_m(f_np1_fun, m_n_fun, Dm, chi, u, v)
    m_rhs = np.zeros(nodes)
    
    mk[start : end] = FCT_alg(A_m, m_rhs, m_n, dt, nodes, M, M_Lump, dof_neighbors)    
    # mk[start : end] =  spsolve(M - dt*A_m, M@m_n + dt*m_rhs)
    
    m_re = reorder_vector_from_dof_time(mk[start : end], 1, nodes, vertextodof)
    f_re = reorder_vector_from_dof_time(fk[start : end], 1, nodes, vertextodof)

    if show_plots is True and i % 5 == 0:
        fig2 = plt.figure(figsize = (10, 5))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(1,2,1)
        im1 = plt.imshow(m_re.reshape((sqnodes, sqnodes)))#, vmin = min_m, vmax = max_m)
        fig2.colorbar(im1)
        plt.title(f'Computed state $m$ at t = {round(t,5)}')
        ax2 = plt.subplot(1,2,2)
        im2 = plt.imshow(f_re.reshape((sqnodes, sqnodes)))#, vmin = min_f, vmax = max_f)
        fig2.colorbar(im2)
        plt.title(f'Computed state $f$ at t = {round(t,5)}')
        plt.show()
        
    mk.tofile(out_folder_name + f'/chtx_m.csv', sep = ',')
    fk.tofile(out_folder_name + f'/chtx_f.csv', sep = ',')


print(f'{T=}, {dt=}, {deltax=}, {chi=}, {Dm=}, {Df=}')
