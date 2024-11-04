from pathlib import Path
import os
import sys

import dolfin as df
from dolfin import dx
# from dolfin import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import psutil
import sys
from helpers import *

matplotlib.use('Agg')
# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the advection-diffusion equation
### with Flux-corrected transport method
# min_{u,v,a,b} 1/2*||u(T)-\hat{u}_T||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt - eps*grad^2(u) + div( u (omega*w + c*m) ) = 0   in Ωx[0,T]
#                           dot(grad u, n) = 0            on ∂Ωx[0,T]
#                                    du/dn = 0            on ∂Ωx[0,T]
#                                     u(0) = u0(x)        in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]

# b = drift vector, e.g. (1,1)
# c = control variable, velocity of the drift

# Optimality conditions:
#  du/dt - eps*grad^2(u) + w \dot grad(u)) = 0                     in Ωx[0,T]
#    -dp/dt - eps*grad^2 p - w \dot grad(p)= 0                     in Ωx[0,T]
#                            dp/dn = du/dn = 0                     on ∂Ωx[0,T]
#                                     u(0) = u0(x)                 in Ω
#                                     p(T) = hat{u}_T - u(T)       in Ω
# gradient equation:      beta*c - u*dot(m, grad(p)) = 0           in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = -1
a2 = 1
deltax = 0.1/2/2
intervals_line = round((a2-a1)/deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = 5
c_lower = 0
e1 = 0.2
e2 = 0.3
k1 = 1
k2 = 1
# slit_width = 0.05
# slit_width = 0.1

# diffusion coefficient
eps = 0
om = np.pi/40

t0 = 0
dt = 0.001
T = 0.1
num_steps = round((T-t0)/dt)
tol = 10**-2 # !!!
# example_name = 'solidbody'
example_name = 'gaussian'
# folder_name = 'solid_body_rotation_drift'
# folder_name = 'solid_body_rotation_drift_wideslit'
# folder_name = 'Gaussian_rotation_drift_025'
# folder_name = 'Gaussian_drift_025' # zero rotation
folder_name = 'Gaussian_drift_025_c10' # zero rotation, lower exponent in absolute terms
out_folder_name = f"advection_Gaussian_drift_c10_T{T}_beta{beta}_tol{tol}"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)
    
# Initialize a square mesh
mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals_line, intervals_line)
V = df.FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()
sqnodes = round(np.sqrt(nodes))

u = df.TrialFunction(V)
v = df.TestFunction(V)

X = np.arange(a1, a2 + deltax, deltax)
Y = np.arange(a1, a2 + deltax, deltax)
X, Y = np.meshgrid(X,Y)

show_plots = True

def u_init(x,y):
    '''
    Initialisation of the position of the solid body/Gaussian object at time zero.
    Input = mesh grid (x,y = square 2D arrays with the same dimensions), time
    '''
    # out = np.zeros(x.shape)
    # R = np.sqrt(x**2 + (y-1/3)**2)
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         if R[i,j] < 1/3 and (abs(x[i,j]) > slit_width or y[i,j] > 0.5):
    #             out[i,j] = 1
    #         else:
    #             out[i,j] = 0    
    c = 20/2
    d = 5
    out = np.exp(-c *( x**2 + d*(y-1/3)**2))
    print(f'Init. condition uses {c=}, {d=}')
    return out

def velocity():
    wind = df.Expression(('-x[1]','x[0]'), degree=4)
    print(f'Velocity field is 1/om*(-y,x) with {om=}')
    return 1/om*wind

drift = df.Constant(('1','1'))
rot = velocity()

vertextodof = df.vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)
M_diag = M.diagonal()

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# Advection matrix for rotation
Arot = 0*assemble_sparse(dot(rot, grad(v))*u * dx)

###############################################################################
########################### Initial guesses for GD ############################
#### ck: use np.ones to multiply drift by one
#### uk: use target states on all time steps
#### pk, dk: np.zeros
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time

u0_orig = u_init(X, Y).reshape(nodes)

# importing data in dof ordering
pk = np.zeros(vec_length)
dot_drift_grad_pk = np.zeros(vec_length)
ck = 1*np.ones(vec_length)
dk = np.zeros(vec_length)

u0 = reorder_vector_to_dof_time(u0_orig, 1, nodes, vertextodof)

uhat_all = np.zeros(vec_length)
uhat_all[:nodes] = u0
# Iterate through files '001.csv' to '099.csv'
for i in range(1, num_steps+1):
    start = i * nodes
    end = (i + 1) * nodes
    t = i*dt
    uhat_all[start : end] = np.genfromtxt(folder_name + '/' + example_name + 
                                            '_t' + f'{t:.3f}_u.csv', delimiter=',')
    
uhat_all_re = reorder_vector_from_dof_time(uhat_all, num_steps + 1, nodes, vertextodof)
uk = np.copy(uhat_all)

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional(uk, uhat_all, ck, num_steps, dt, M, beta, optim='alltime')

cost_fun_vals = []
cost_fidelity_vals = []
cost_control_vals = []

stop_crit_costfun = 5 
print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
while  (stop_crit_costfun >= tol) and it < 1000:

    it += 1
    print(f'\n{it=}')
        
    # In k-th iteration we solve for u^k, p^k using c^k (S1 & S2)
    # and calculate c^{k+1} (S5)
    
    ###########################################################################
    ############### 1. solve the state equation using FCT #####################
    ###########################################################################
    print('Solving state equation...')
    t=0
    uk[nodes:] = np.zeros(num_steps * nodes) # initialise uk, keep IC
    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        uk_n = uk[start - nodes : start] # uk(t_n), i.e. previous time step at k-th GD iteration
        ck_np1_fun = vec_to_function(ck[start : end], V)

        u_rhs = np.zeros(nodes)
        
        Adrift1 = assemble_sparse(dot(drift, grad(ck_np1_fun))*u * v * dx) # pseudo-mass matrix
        Adrift2 = assemble_sparse(dot(drift, grad(v)) * ck_np1_fun * u * dx) # pseudo-stiffness matrix
        
        ## System matrix for the state equation
        A_u = - eps * Ad + Arot + Adrift1 + Adrift2
        
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
    
    print(f'{L2_norm_sq_Q(uk - uhat_all, num_steps, dt, M)=}')
    ###########################################################################
    ############### 2. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) 
    dot_drift_grad_pk = np.zeros(vec_length)
    t = T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V) # uk(t_n)
        ck_n_fun = vec_to_function(ck[start : end], V)
        
        uhat_n_fun = vec_to_function(uhat_all[start:end], V) 

        Adrift1 = assemble_sparse(dot(drift, grad(ck_n_fun))*u * v * dx) # pseudo-mass matrix
        Adrift2 = assemble_sparse(dot(drift, grad(v)) * ck_n_fun * u * dx) # pseudo-stiffness matrix

        A_p = - eps * Ad - Arot - Adrift1 - Adrift2
        
        p_rhs = np.asarray(assemble((uhat_n_fun - uk_n_fun) * v * dx))
        
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors)
        
    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    for i in range(num_steps + 1): # calculate dk, across all time steps (incl. 0 and T)
        start = i * nodes
        end = (i + 1) * nodes
        
        uk_fun = vec_to_function(uk[start:end], V)
        pk_fun = vec_to_function(pk[start:end], V)
        
        rhs_dk = -(beta*M*ck[start:end] \
                    + np.asarray(assemble(pk_fun * dot(drift, grad(uk_fun))*v*dx)))
        
        dk[start:end] = ChebSI(rhs_dk, M, M_diag, 20, 0.5, 2)
        
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    
    print('Starting Armijo line search...')
    sk, u_inc = armijo_line_search_sbr_drift(uk, pk, ck, dk, uhat_all, eps, drift, 
                  num_steps, dt, nodes, M, M_Lump, Ad, Arot, c_lower, c_upper, 
                  beta, V, dof_neighbors, optim='alltime')
        
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    
    cost_fun_kp1 = cost_functional(u_inc, uhat_all, ckp1, num_steps, dt, M, beta,
                                   optim='alltime')
    
    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals.append(L2_norm_sq_Q(u_inc - uhat_all, num_steps, dt, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    stop_crit_costfun = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit_costfun=}')
    
    uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
    ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
    pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
    
    for i in range(num_steps):
        startP = i * nodes
        endP = (i+1) * nodes
        tP = i * dt
        
        startU = (i+1) * nodes
        endU = (i+2) * nodes
        tU = (i+1) * dt
        
        u_re = uk_re[startU : endU]
        c_re = ck_re[startP : endP]
        p_re = pk_re[startP : endP]
        uhat_all_re_t = uhat_all_re[startU : endU]
            
        u_re = u_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))
        uhat_all_re_t = uhat_all_re_t.reshape((sqnodes,sqnodes))

        if show_plots is True and (i%10 == 0 or i==num_steps-1):
            fig = plt.figure(figsize=(20, 5))

            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(uhat_all_re_t)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $u$ at t = {round(tU, 5)}')
        
            ax = fig.add_subplot(1, 4, 2)
            im2 = ax.imshow(u_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f'Computed state $u$ at t = {round(tU, 5)}')
        
            ax = fig.add_subplot(1, 4, 3)
            im3 = ax.imshow(p_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f'Computed adjoint $p$ at t = {round(tP, 5)}')
        
            ax = fig.add_subplot(1, 4, 4)
            im4 = ax.imshow(c_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f'Computed control $c$ at t = {round(tP, 5)}')
        
            fig.tight_layout(pad=3.0)
            plt.savefig(out_folder_name + f'/it_{it}_plot_{i:03}.png')
        
            # Clear and remove objects explicitly
            ax.clear()      # Clear axes
            cb1.remove()     # Remove colorbars
            cb2.remove()
            cb3.remove()
            cb4.remove()
            del im1, im2, im3, im4, cb1, cb2, cb3, cb4
            fig.clf()
            plt.close(fig)  
    
    if it > 1:
        fig2 = plt.figure(figsize=(15, 5))
    
        ax2 = fig2.add_subplot(1, 3, 1)
        im1 = plt.plot(np.arange(1, it + 1), cost_fun_vals)
        plt.title(f'{it=} Cost functional')
        
        ax2 = fig2.add_subplot(1, 3, 2)
        im2 = plt.plot(np.arange(1, it + 1), cost_fidelity_vals)
        plt.title('Data fidelity norm in L2(Omega)^2')
        
        ax2 = fig2.add_subplot(1, 3, 3)
        im3 = plt.plot(np.arange(1, it + 1), cost_control_vals)
        plt.title('Regularisation norm in L2(Q)^2')
        
        fig2.tight_layout(pad=3.0)
        plt.savefig(out_folder_name + f'/progress_it_{it}_plot_{i:03}.png')
        
        # Clear and remove objects explicitly
        ax2.clear()      # Clear axes
        del im1, im2, im3
        fig2.clf()
        plt.close(fig2)
    
    max_ck = []
    for i in range(num_steps):
        start = i * nodes
        end = (i+1) * nodes
        max_ck.append(np.amax(ck[start:end]))
    np.mean(max_ck)

    print('Mean of the control in L_2(Omega) squared:', np.mean(max_ck))
    print('Square root of the mean: ', np.sqrt(np.mean(max_ck)))
    
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the memory usage in MB
    mem_usage = process.memory_info().rss / (1024 * 1024)

    print(f"Memory usage: {mem_usage:.2f} MB")


###############################################################################    

uk.tofile(out_folder_name + f'/gaussian_T{T}_beta{beta}_u.csv', sep = ',')
ck.tofile(out_folder_name + f'/gaussian_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder_name + f'/gaussian_T{T}_beta{beta}_p.csv', sep = ',')

print(f'Exit:\n Stop. crit.: {stop_crit_costfun}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print(f'Output saved to:', out_folder_name)

