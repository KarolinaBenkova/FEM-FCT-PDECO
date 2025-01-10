from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

from pathlib import Path

# ---------------------------------------------------------------------------
### Flux-corrected transport method for a nonlinear advection-reaction-diffusion equation
#  du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = s       in Ωx[0,T]
#                     dot(-eps*grad(u) + w*u, n) = 0       on ∂Ωx[0,T]
#                                           u(0) = u0(x)   in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]
### Note: thanks to this, we get the BC du/dn = 0 on ∂Ωx[0,T]
# used to generate target state for advection PDECO
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1/2/2
intervals_line = round((a2-a1)/deltax)

# diffusion coefficient
eps = 0.0001
# speed of wind
speed = 1

t0 = 0
dt = 0.001
T = 1
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
filename_start = 'nonlinear_stripes_source_control_coarse/advection_t'
# if not Path(filename_start).exists():
    # Path(filename_start).mkdir(parents=True)

def u_init(X,Y):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    kk = 4
    out = 5*Y*(Y-1)*X*(X-1)*np.sin(kk*X*np.pi)
    return out

k1 = 2
k2 = 2
source_fun_expr = Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

wind = Expression(('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
          'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4, speed = speed)
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
mat_u = A - eps * Ad

zeros = np.zeros(nodes)

###############################################################################
############################## Initial condition ##############################
###############################################################################

u0_orig = u_init(X, Y).reshape(nodes)
u0 = reorder_vector_to_dof_time(u0_orig, 1, nodes, vertextodof)
plt.imshow(u_init(X, Y))
plt.colorbar()
plt.show()

vec_length = (num_steps + 1)*nodes # include zero and final time
uk = np.zeros(vec_length)
uk[:nodes] = u0

uk_dir = np.zeros(vec_length)
uk_dir[:nodes] = u0


print(f'dx={deltax}, {dt=}, {T=}')
t=0
for i in range(1,num_steps + 1):    # solve for uk(t_{n+1})
    start = i * nodes
    end = (i + 1) * nodes
    t += dt
    print('t = ', round(t, 4))
    
    u_n = uk[start - nodes : start]
    u_n_fun = vec_to_function(u_n, V) 
    M_u2 = assemble_sparse(u_n_fun * u_n_fun * u * v *dx)

    u_rhs = np.asarray(assemble(source_fun_expr*v*dx))
    
    uk[start:end] = FCT_alg(mat_u, u_rhs, u_n, dt, nodes, M, M_Lump, 
                    dof_neighbors, source_mat = -M + 1/3*M_u2)

    uk_re = reorder_vector_from_dof_time(uk[start:end],1, nodes, vertextodof)
    uk[start : end].tofile(filename_start + f'{t:.3f}_u.csv', sep = ',')

    if i%10 ==0:
        plt.imshow(uk_re.reshape((sqnodes,sqnodes)))
        plt.colorbar()
        plt.title(f'Computed state $u$ at t = {round(t,5)}')
        plt.show()

uk.tofile(filename_start + f'_u.csv', sep = ',')

        