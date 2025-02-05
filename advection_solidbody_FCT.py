from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *

# ---------------------------------------------------------------------------
### Flux-corrected transport method for the advection(-diffusion) equation
#  du/dt - eps*grad^2(u) + w \dot grad(u)) = c + g       in Ωx[0,T]
#                           dot(grad u, n) = 0           on ∂Ωx[0,T]
#                                    du/dn = 0           on ∂Ωx[0,T]
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]
# w = omega*(-y, x) + 2*(1,1) for rotation and drift with constant velocity 2
# used to generate target state for advection solid body PDECO, c=2
# ---------------------------------------------------------------------------

## Define the parameters
a1 = -1
a2 = 1
deltax = 0.1/2/2
intervals_line = round((a2-a1)/deltax)
# box constraints for c, exact solution is in [0,1]
e1 = 0.2
e2 = 0.3
k1 = 1
k2 = 1
# slit_width = 0.05 
slit_width = 0.1

# diffusion coefficient
eps = 0 #0.001
om = np.pi/40 
# if om = np.pi/10 & dt=0.1, at T=2 the body rotates into starting position

t0 = 0
dt = 0.001 #deltax**2 #
T = 0.5 #2
num_steps = round((T-t0)/dt)

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
v = TestFunction(V)

X = np.arange(a1, a2 + deltax, deltax)
Y = np.arange(a1, a2 + deltax, deltax)
X, Y = np.meshgrid(X,Y)

show_plots = True

def u_init(X,Y):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = np.zeros(X.shape)
    R = np.sqrt(X**2 + (Y-1/3)**2)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if R[i,j] < 1/3 and (abs(X[i,j]) > slit_width or Y[i,j] > 0.5):
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out

def velocity(X,Y):
    wind = Expression(('-x[1]','x[0]'), degree=4)
    move = Expression(('2','2'), degree=4)
    return 1/om*wind + move

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

wind = velocity(X,Y)

# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# Advection matrix
A = assemble_sparse(dot(wind, grad(v))*u * dx)

## System matrix for the state equation
A_u = A - eps * Ad

zeros = np.zeros(nodes)

M_diag = M.diagonal()
M_Lump_diag = M_Lump.diagonal()

###############################################################################
############################## Initial condition ##############################
###############################################################################

u0_orig = u_init(X, Y).reshape(nodes)
u0 = reorder_vector_to_dof_time(u0_orig, 1, nodes, vertextodof)

vec_length = (num_steps + 1)*nodes # include zero and final time
uk = np.zeros(vec_length)
uk[:nodes] = u0

print(f'dx={deltax}, {dt=}, {T=}')
t=0
for i in range(1, num_steps + 1):    # solve for uk(t_{n+1})
    start = i * nodes
    end = (i + 1) * nodes
    t += dt
    print('t = ', round(t, 4))
    
    uk_n = uk[start - nodes : start] # uk(t_n), i.e. previous time step at k-th GD iteration

    u_rhs = np.zeros(nodes)
    
    # no FCT solution
    # uk[start:end] = spsolve(M - dt*A_u, M @ uk_n)
    
    # low-order solution
    # D = artificial_diffusion_mat(A_u)
    # Mat_u_Low = M_Lump - dt * (A_u + D)
    # Rhs_u_Low = M_Lump @ uk_n
    # uk[start:end] = spsolve(Mat_u_Low, Rhs_u_Low)
    
    uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
    filename_start = 'solid_body_rotation_drift_wideslit/solidbody_t'

    uk[start : end].tofile(filename_start + '{t:.3f}_u.csv', sep = ',')
    
    uk_re = reorder_vector_from_dof_time(uk[start:end],1, nodes, vertextodof)

    plt.imshow(uk_re.reshape((sqnodes,sqnodes)))
    plt.colorbar()
    plt.title(f'Computed state $u$ at t = {round(t,5)}')
    plt.show()
        
###############################################################################    
# Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)

# min_u = min(np.amin(uk), np.amin(u_init(X, Y)))
# max_u = max(np.amax(uk), np.amax(u_init(X,Y)))

# for i in range(num_steps):
#     startU = (i+1) * nodes
#     endU = (i+2) * nodes
#     tU = (i+1) * dt
    
#     u_re = uk_re[startU : endU].reshape((sqnodes,sqnodes))
#     # filename_start = 'solid_body_rotation_drift/solidbody_t'
#     # filename_start = 'solid_body_rotation_drift_wideslit/solidbody_t'
#     # uk[startU : endU].tofile(filename_start + str(tU) + '_u.csv', sep = ',')

#     if show_plots is True and i%20 == 0:
#         fig2 = plt.figure(figsize = (10,5))
#         fig2.tight_layout(pad = 3.0)
#         ax2 = plt.subplot(1,2,1)
#         im1 = plt.imshow(u0_orig.reshape((sqnodes,sqnodes)), extent =[a1,a2,a1,a2]) #, vmin = min_u, vmax = max_u,
#         fig2.colorbar(im1)
#         plt.title(f'Exact solution $u$ at t = {round(tU,5)}')
#         ax2 = plt.subplot(1,2,2)
#         im2 = plt.imshow(u_re, extent =[a1,a2,a1,a2]) #, vmin = min_u, vmax = max_u,
#         fig2.colorbar(im2)
#         plt.title(f'Computed state $u$ at t = {round(tU,5)}')
#         plt.show()
        
#         # filename = f'solid_body_rotation_drift/plot_{i:03}.png'  # e.g., plot_001.png, plot_002.png, etc.
#         # plt.savefig(filename)
#         # plt.close()
        
#     print('------------------------------------------------------')

u_re_T = uk_re[num_steps * nodes :]

E_u = np.linalg.norm(u0 - u_re_T)
RE_u = E_u / np.linalg.norm(u0)
WE_u = deltax * E_u

print(f'{dt=}, {deltax=}, {T=}, {om=}')
# print('Relative errors')
# print('u:', RE_u)
# print('Weighted errors')
# print('u:', WE_u)
# print(RE_u ,  ',', WE_u)

