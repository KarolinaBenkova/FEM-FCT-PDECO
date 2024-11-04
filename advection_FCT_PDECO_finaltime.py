from pathlib import Path

import dolfin as df
from dolfin import dx
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the advection-diffusion equation
### with Flux-corrected transport method
## uses desired states only at final time
# min_{u,v,a,b} 1/2*||u(T)-\hat{u}_T||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt - eps*grad^2(u) + div(w*u) = c      in Ωx[0,T]
#                    dot(grad u, n) = 0      on ∂Ωx[0,T]
#                              u(0) = u0(x)  in Ω

# w = velocity/wind vector with the following properties:
#                               w \dot n = 0           on ∂Ωx[0,T]

# Optimality conditions:
#         du/dt - eps*grad^2(u) + grad(w*u) = c           in Ωx[0,T]
#    -dp/dt - eps*grad^2 p - w \dot grad(p) = 0           in Ωx[0,T]
#                             dp/dn = du/dn = 0                on ∂Ωx[0,T]
#                                      u(0) = u0(x)            in Ω
#                                      p(T) = uhat_T - u(T)    in Ω
# gradient equation:                     c = 1 \ beta * p
#                                        c = proj_[ca,cb] (1/beta*p) in Ωx[0,T]

# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1/2/2/2
intervals_line = round((a2-a1)/deltax)
beta = 0.001

# box constraints for c, exact solution is in [0,1]
c_upper = 1
c_lower = -1
# diffusion coefficient
eps = 0.0001
# speed of wind
speed = 1

t0 = 0
dt = 0.001
T = 0.2
final_time = '0.200'
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!

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

example_name = 'advection_stripes_source_control_wind2/advection'
out_folder_name = f"advection_stripes_source_control_wind2_T{T}_beta{beta}_tol{tol}"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)
    
def u_init(x,y):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    kk = 4
    out = 5*y*(y-1)*x*(x-1)*np.sin(kk*x*np.pi)
    return out

def velocity(x,y):
    if a1==0 and a2==1: ## [0,1]^2
        # wx = - speed*(Y-0.5)
        # wy = speed*(X-0.5)
        
        # wind = Expression(('-speed*(x[1]-0.5)','speed*(x[0]-0.5)'), degree=4, speed = speed)
        
        
        ## wind 2:
        wx =  speed*2*(y-0.5)*x*(1-x)
        wy = -speed*2*(x-0.5)*y*(1-y)
        
        wind = df.Expression(('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
                  'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4, speed = speed)

    else:
        raise ValueError("No velocity field defined for the domain specified.")

    return wx, wy, wind

vertextodof = df.vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

_,_,wind = velocity(X,Y)
W = df.VectorFunctionSpace(mesh, "CG", 1)
wind_fun = df.Function(W)
wind_fun = df.project(wind, W)
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

# # Advection matrices
# Aa1 = assemble_sparse(div(wind_fun*u*v)*dx) ## int_\Omega div(w*v*u) *dx
# Aa2 = assemble_sparse(dot(wind, grad(v))*u * dx) ## int_\Omega w*grad(v) *u*dx
# ## System matrix for the state equation
# A_u = Aa2 - eps * Ad
# ## System matrix for the adjoint equation (opposite sign of transport matrix)
# A_p = Aa1 - Aa2 - eps * Ad

A_SE = assemble_sparse(dot(wind, grad(v))*u * dx)
A_AE = assemble_sparse(dot(wind, grad(u))*v * dx)
A_u = - eps * Ad + A_SE
A_p = - eps * Ad + A_AE


zeros = np.zeros(nodes)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time

uk = np.zeros(vec_length)
pk = np.zeros(vec_length)
ck = np.zeros(vec_length)#np.ones(vec_length) #np.zeros(vec_length)
dk = np.zeros(vec_length)
wk = np.zeros(vec_length)

u0_orig = u_init(X, Y).reshape(nodes)
u0 = reorder_vector_to_dof_time(u0_orig, 1, nodes, vertextodof)
uhat_T = np.genfromtxt(example_name + '_t' + final_time + '_u.csv', delimiter=',')
uhat_T_re = reorder_vector_from_dof_time(uhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))

uk[:nodes] = u0
wk[:nodes] = u0

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, optim='finaltime')
cost_fun_vals = []
cost_fidelity_vals = []
cost_control_vals = []

stop_crit = 5

print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
# while  (stop_crit >= tol and cost_fun_k >= 0.02) and it < 100:
while stop_crit >= tol and it < 1000:

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
        
        u_rhs = np.asarray(assemble(ck_np1_fun*v*dx))        ## !!
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)

    ###########################################################################
    ############### 2. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) # includes the final-time condition
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    
    # plt.imshow(reorder_vector_from_dof_time(pk[num_steps * nodes :], 1, nodes, vertextodof).reshape((sqnodes,sqnodes)))
    # plt.colorbar()
    # plt.title(f'p(T) at {it=}')
    # plt.show()
    
    t=T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        
        p_rhs =  np.zeros(nodes)
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors)

    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta*ck - pk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    
    print('Solving equation for move in u...')
    t=0
    wk[nodes:] = np.zeros(num_steps * nodes)
    for i in range(1, num_steps + 1):
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        wk_n = wk[start - nodes : start]
        dk_np1_fun = vec_to_function(dk[start : end], V)
        wk_n_fun = vec_to_function(wk_n, V)
        
        w_rhs = np.asarray(assemble(dk_np1_fun * v * dx))
        wk[start:end] = FCT_alg(A_u, w_rhs, wk_n, dt, nodes, M, M_Lump, dof_neighbors)

    print(f'{cost_fun_k=}')
    print('\nStarting Armijo line search...')
    # sk, u_inc = armijo_line_search(uk, pk, wk, ck, dk, uhat_T, num_steps, dt, 
    #                                M, c_lower, c_upper, beta, cost_fun_k, 
    #                                optim = 'finaltime')
    sk, u_inc = armijo_line_search(uk, ck, dk, uhat_T, num_steps, dt, M, 
                                   c_lower, c_upper, beta, cost_fun_k, nodes, 
                                   w = wk, optim = 'finaltime')
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    cost_fun_kp1 = cost_functional(u_inc, uhat_T, ckp1, num_steps, dt, M, beta,
                                   optim='finaltime')
    print(f'{cost_fun_kp1=}')

    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals.append(L2_norm_sq_Omega(u_inc[num_steps*nodes:] - uhat_T, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    stop_crit = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit=}')
    
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
            
        u_re = u_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))
        
        if show_plots is True and i%5 == 0:
            fig = plt.figure(figsize=(20, 5))

            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(uhat_T_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $u$ at t = {T}')
        
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
        plt.savefig(out_folder_name + f'/progress_plot.png')
        
        # Clear and remove objects explicitly
        ax2.clear()      # Clear axes
        del im1, im2, im3
        fig2.clf()
        plt.close(fig2)
            
###############################################################################    
uk.tofile(out_folder_name + f'/advection_T{T}_beta{beta}_u.csv', sep = ',')
ck.tofile(out_folder_name + f'/advection_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder_name + f'/advection_T{T}_beta{beta}_p.csv', sep = ',')


print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('L2_\Omega^2 (u(T) - uhat_T)=', L2_norm_sq_Omega(uk[num_steps*nodes]-uhat_T,M))

print(np.amax(ck[num_steps*nodes:]))
print(np.amin(ck[num_steps*nodes:]))