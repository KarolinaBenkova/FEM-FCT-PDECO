from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *

from pathlib import Path

# ---------------------------------------------------------------------------
### Flux-corrected transport method for the advection(-diffusion) equation
#  du/dt - eps*grad^2(u) + div( w grad(u)) = s           in Ωx[0,T]
#                        ?   dot(grad u, n) = 0           on ∂Ωx[0,T]
#                                    du/dn = 0           on ∂Ωx[0,T]
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                               ?  div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]
# used to generate target state for advection PDECO
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1/2/2/2
intervals_line = round((a2-a1)/deltax)

# diffusion coefficient
eps = 0.0001
# speed of wind
speed = 1

t0 = 0
dt = 0.001
T = 0.5
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
filename_start = 'advection_stripes_source_control_wind2/advection_t'
if not Path(filename_start).exists():
    Path(filename_start).mkdir(parents=True)

def u_init(X,Y):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    # out = np.zeros(X.shape)
    kk = 4
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         # if X[i,j] > 0.25 and Y[i,j] > 0.25 and X[i,j] < 0.75 and Y[i,j] < 0.75:
    #         if 0.25 <= X[i,j] <= 0.75 and 0.25 <= Y[i,j] <= 0.75:
    #             out[i,j] = np.sin(kk*X[i,j]*np.pi)*np.sin(kk*Y[i,j]*np.pi)          

    # out = np.sin(kk*X*np.pi)*np.sin(kk*Y*np.pi)              
    out = 5*Y*(Y-1)*X*(X-1)*np.sin(kk*X*np.pi) #*np.sin(kk*Y*np.pi)
    # out = X+Y
    return out

k1 = 2
k2 = 2
source_fun_expr = Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)
# source_fun_expr = Expression('x[0] + x[1]', degree=4)

def velocity(X,Y):
    # wind = Expression(('-speed*(x[1]-0.5)','speed*(x[0]-0.5)'), degree=4, speed = speed)
    # drift = Constant(('1','1'))
    
    # wind 2:
    wind = Expression(('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
              'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4, speed = speed)
    return wind

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
    
    uk_n = uk[start - nodes : start] # uk(t_n), i.e. previous time step at k-th GD iteration

    u_rhs = np.asarray(assemble(source_fun_expr*v*dx))
    # u_rhs = np.zeros(nodes)
    
    # uk[start:end], mat_FCT, dif_FCT = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
    uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)

    # direct solver 
    uk_dir_n = uk_dir[start - nodes : start]
    mat_u = M - dt*A_u
    rhs_u = M @ uk_dir_n + dt * u_rhs
    uk_dir[start:end] = spsolve(mat_u, rhs_u)
    
    uk_re = reorder_vector_from_dof_time(uk[start:end],1, nodes, vertextodof)
    uk_dir_re = reorder_vector_from_dof_time(uk_dir[start:end],1, nodes, vertextodof)
    uk[start : end].tofile(filename_start + f'{t:.3f}_u.csv', sep = ',')

    if i%10 ==0:
        # plt.imshow(uk_dir_re.reshape((sqnodes,sqnodes)))
        plt.imshow(uk_re.reshape((sqnodes,sqnodes)))
        plt.colorbar()
        plt.title(f'Computed state $u$ at t = {round(t,5)}')
        plt.show()
        
        uk_FE = Function(V)
        uk_FE.vector()[:] =  uk[start:end]
        plot(uk_FE)
        plt.imshow(uk_dir_re.reshape((sqnodes,sqnodes)))
        plt.colorbar()
        plt.title(f'Computed state $u$ at t = {round(t,5)}')
        plt.show()

# mat_u = M - dt*A_u
# mat_u = mat_u.todense()
# print(np.all(mat_FCT == mat_u))

# mat_dif = mat_FCT - mat_u
# plt.imshow(mat_dif)
# plt.colorbar()
# plt.title('Diffence between mat_FCT and mat_u')
# plt.show()

# plt.imshow(mat_FCT)
# plt.colorbar()
# plt.title('mat_FCT')
# plt.show()
# plt.imshow(mat_u) 
# plt.colorbar()
# plt.title('mat_u')
# plt.show()
# plt.imshow(dif_FCT.todense())
# plt.colorbar()
# plt.title('artif. dif. matrix')
# plt.show()

# from scipy.sparse import spdiags
# Ad_offdiag = Ad - spdiags(Ad.diagonal(), diags = 0, m = nodes, n = nodes)
# Ad_offdiag = Ad_offdiag.todense()
# plt.imshow(Ad_offdiag) 
# plt.title('off-diag entries of Ad')
# plt.colorbar()
# plt.show()
# print(np.amin(Ad_offdiag), np.amax(Ad_offdiag))

# compare sols

# dif_norm = np.linalg.norm(u_dir - u_FCT)