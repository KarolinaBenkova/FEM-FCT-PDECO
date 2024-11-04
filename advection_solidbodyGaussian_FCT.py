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

filename_start = 'Gaussian_drift_025_c10/gaussian_t'

def u_init(X,Y):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    c = 20/2
    d = 5
    out = np.exp(-c *( X**2 + d*(Y-1/3)**2))
    return out

def velocity(X,Y):
    wind = Expression(('-x[1]','x[0]'), degree=4)
    move = Expression(('2','2'), degree=4)
    # return 1/om*wind + move
    
    ## drift only, no rotation
    return move

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
    
    uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)

    uk[start : end].tofile(filename_start + f'{t:.3f}_u.csv', sep = ',')
    
    uk_re = reorder_vector_from_dof_time(uk[start:end],1, nodes, vertextodof)

    plt.imshow(uk_re.reshape((sqnodes,sqnodes)))
    plt.colorbar()
    plt.title(f'Computed state $u$ at t = {round(t,5)}')
    plt.show()
        
###############################################################################    

print(f'{dt=}, {deltax=}, {T=}, {om=}')

