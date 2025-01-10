from pathlib import Path

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

# ---------------------------------------------------------------------------
### Flux-corrected transport method for the Schnakenberg model to generate patterns
#   -Du grad^2 u + om1 w \cdot grad(u) + gamma(u-u^2v-a) = 0       in Ω
#   -Dv grad^2 v + om2 w \cdot grad(v) + gamma(u^2v-b)   = 0       in Ω
#                        zero flux BCs       on ∂Ω
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                               div (w) = 0           in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.01*2
intervals_line = round((a2-a1)/deltax)
e1 = 0.2
e2 = 0.3
k1 = 1
k2 = 1

t0 = 0
dt = 0.001
T = 1
num_steps = round((T-t0)/dt)

# ### Basic pattern
# Du = 1
# Dv = 10
# c_a = 0.126779
# c_b = 0.792366
# gamma = 1000
# omega = 0

# # Setup used in Garzon-Alvarado et al (2011)
# eps = 1 #1/10
Du = 1/100
Dv = 8.6676
c_a = 0.1
c_b = 0.9
gamma = 230.82
# omega = 0.6

omega1 = 100 #0.6
omega2 = 0.6

C_a = Constant(c_a)
C_b = Constant(c_b)

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
w = TestFunction(V)

# Define the surface measure ds for the boundary
ds = Measure('ds', domain=mesh)

X = np.arange(a1, a2 + deltax, deltax)
Y = np.arange(a1, a2 + deltax, deltax)
X, Y = np.meshgrid(X,Y)

show_plots = True
out_folder_name = f"Schnak_adv_Du{Du}_timedep_vel_coarse_v2"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)
    
def init_conditions(X,Y):
    '''
    Function for the initial conditions.
    Input = mesh grid (X,Y) = square 2D arrays with the same dimensions).
    '''
    # con = 0.0016
    con = 0.1
    u_init = c_a + c_b + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))

    v_init = c_b / pow(c_a + c_b, 2) + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))                  
    
    # con = 0.15
    # u_init = c_a + c_b + con*(np.random.rand(*X.shape)- 0.5)
    # v_init = c_b / pow(c_a + c_b, 2) + con*(np.random.rand(*X.shape)- 0.5)
    return u_init, v_init

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

t=0
# wind = Expression(('-(x[1]-0.5)','(x[0]-0.5)'), degree=4)
wind = Expression(('-(x[1]-0.5)*sin(2*pi*t)','(x[0]-0.5)*sin(2*pi*t)'), degree=4, pi = np.pi, t = t)

# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * w * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(w)) * dx)

# Advection matrix
A = assemble_sparse(dot(wind, grad(u)) * w * dx)

###############################################################################
############################## Initial condition ##############################
###############################################################################
u0_orig, v0_orig = init_conditions(X, Y)

u0 = reorder_vector_to_dof_time(u0_orig.reshape(nodes), 1, nodes, vertextodof)
v0 = reorder_vector_to_dof_time(v0_orig.reshape(nodes), 1, nodes, vertextodof)

vec_length = (num_steps + 1)*nodes # include zero and final time
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)

uk[:nodes] = u0
vk[:nodes] = v0

print(f'dx={deltax}, {dt=}, {T=}')
t=0
for i in range(1, num_steps + 1):    # solve for uk(t_{n+1})
    start = i * nodes
    end = (i + 1) * nodes
    t += dt
    wind.t = t

    print('t = ', round(t, 4))
    
    u_n = uk[start - nodes : start]
    v_n = vk[start - nodes : start]

    # Define previous time-step solution as a function
    u_n_fun = vec_to_function(u_n, V)
    v_n_fun = vec_to_function(v_n, V)
    
    ###########################################################################
    ############################ DECOUPLED, IMEX ##############################
    ###########################################################################
    
    
    ########## Replicate paper results, Fig. 3 Garzon-Alvarado'11 #############
    
    # # solve for u (smaller diffusion coef. --> make advecton-dominated)
    # A = assemble_sparse(dot(wind, grad(u)) * w * dx)
    # mat_u = -(Du*K + omega1*A)
    # rhs_u = np.asarray(assemble((gamma*(c_a + u_n_fun**2 * v_n_fun))* w * dx))
    # # uk[start : end] = spsolve(M - dt*(mat_u - gamma*M), M@u_n + dt*rhs_u)    
    # uk[start : end] = FCT_alg(mat_u, rhs_u, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

    # u_np1_fun = vec_to_function(uk[start : end], V)
    # M_u2 = assemble_sparse(u_np1_fun * u_np1_fun * u * w *dx)
    
    # # solve for v
    # mat_v = -(Dv*K + omega2*A)
    # rhs_v = np.asarray(assemble((gamma*c_b)* w * dx))
    # vk[start : end] = spsolve(M - dt*(mat_v-gamma*M_u2), M@v_n + dt*rhs_v)    
    # # vk[start : end] = FCT_alg(mat_v, rhs_v, v_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M_u2)
    
    # v2: different weak formulation for advection matrix (corresponding signs also change)
    # Solve for u using FCT (advection-dominated equation)
    A = assemble_sparse(dot(wind, grad(w)) * u * dx)
    mat_u = -(Du*Ad - omega1*A)
    rhs_u = np.asarray(assemble((gamma*(c_a + u_n_fun**2 * v_n_fun))* w * dx))
    uk[start : end] = FCT_alg(mat_u, rhs_u, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

    u_np1_fun = vec_to_function(uk[start : end], V)
    M_u2 = assemble_sparse(u_np1_fun * u_np1_fun * u * w *dx)
    
    # Solve for v using a direct solver
    rhs_v = np.asarray(assemble((gamma*c_b)* w * dx))
    vk[start : end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@v_n + dt*rhs_v) 

    u_re = reorder_vector_from_dof_time(uk[start : end], 1, nodes, vertextodof).reshape((sqnodes,sqnodes))
    v_re = reorder_vector_from_dof_time(vk[start : end], 1, nodes, vertextodof).reshape((sqnodes,sqnodes))
    
    if i%1 ==0:
        fig2 = plt.figure(figsize=(12,6), dpi=100)
        ax2 = plt.subplot(2,3,1)
        im1 = plt.imshow(u_re, cmap ="gray")
        fig2.colorbar(im1)
        plt.title(f'$u$ at t={round(i*dt,4)}')
        ax2 = plt.subplot(2,3,2)
        im2 = plt.imshow(v_re, cmap="gray")
        fig2.colorbar(im2)
        plt.title(f'$v$ at t={round(i*dt,4)}')
        plt.show()

print(f'{T=}, {dt=}\n{Du=}, {Dv=}\n{c_a=}, {c_b=}\n{gamma=}\n{omega1=}, {omega2=} ')
uk.tofile(out_folder_name + f'/Schnak_adv_u.csv', sep = ',')
vk.tofile(out_folder_name + f'/Schnak_adv_v.csv', sep = ',')