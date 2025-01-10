from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from helpers import *

import csv
from datetime import datetime
import time

# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for a nonlinear advection-reaction-diffusion equation
### with Flux-corrected transport method
### Source control
# min_{u,v,a,b} 1/2*||u-\hat{u}||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c       in Ωx[0,T]
#                     dot(-eps*grad(u) + w*u, n) = 0       on ∂Ωx[0,T]
#                                           u(0) = u0(x)   in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]
### Note: thanks to this, we get a BC for u: du/dn = 0 on ∂Ωx[0,T]

# Optimality conditions:
#  du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c               in Ωx[0,T]
#    -dp/dt + div(-eps*grad(p) + w*p) +u^2*p - p = \hat{u} - u     in Ωx[0,T]
#                            dp/dn = du/dn = 0                     on ∂Ωx[0,T]
#                                     u(0) = u0(x)                 in Ω
#                                     p(T) = 0                     in Ω
# gradient equation:                     c = 1 \ beta * p
#                                        c = proj_[ca,cb] (1/beta*p) in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1/2/2
intervals_line = round((a2-a1)/deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = -1
c_lower = 1
# diffusion coefficient
eps = 0.0001
# speed of wind
speed = 1

t0 = 0
dt = 0.001
T = 0.5
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!

k1 = 2
k2 = 2

show_plots = True
example_name = f"nonlinear_stripes_source_control_coarse/advection_t"
out_folder_name = f"NL_AT_T{T}_beta{beta}_Ca{c_upper}_Cb{c_lower}_tol{tol}_pert"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)

wind = Expression(('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
          'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4, speed = speed)
source_fun_expr = Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)

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

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)


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

## System matrix for the adjoint equation (opposite sign of transport matrix)
A_p = - A - eps * Ad

zeros = np.zeros(nodes)

###############################################################################
################ Target states & initial conditions for m,f ###################
###############################################################################

def u_init(X,Y):
    '''
    Function for the initial solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    kk = 4
    out = 5*Y*(Y-1)*X*(X-1)*np.sin(kk*X*np.pi)
    return out

u0_orig = u_init(X, Y)
u0 = reorder_vector_to_dof_time(u0_orig.reshape(nodes), 1, nodes, vertextodof)

# uhat = np.zeros((num_steps + 1)*nodes)
uhat = np.genfromtxt(example_name + f'_u.csv', delimiter=',')[:(num_steps+1)*nodes]
uhat_re = reorder_vector_from_dof_time(uhat, num_steps + 1, nodes, vertextodof)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time
uk = np.zeros(vec_length)
pk = np.zeros(vec_length)
ck = np.zeros(vec_length)
dk = np.zeros(vec_length)
perturbation = np.random.uniform(-1e-2, 1e-2, vec_length)
ck += perturbation

uk[:nodes] = u0

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional(uk, uhat, ck, num_steps, dt, M, beta, optim='alltime')
cost_fun_vals = []
cost_fidelity_vals_u = []
cost_control_vals = []
mean_control_vals = []
stop_crit = 5

print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')

# Record the start time of the simulation
start_time = time.time()

while (stop_crit >= tol ) and it<1000:
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
        uk_n_fun = vec_to_function(uk[start : end], V)

        ck_np1_fun = vec_to_function(ck[start : end], V)
        
        M_u2 = assemble_sparse(uk_n_fun * uk_n_fun * u * v *dx)
        u_rhs = np.asarray(assemble((ck_np1_fun) * v * dx))
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors,
                                source_mat = -M + 1/3*M_u2)
        
    ###########################################################################
    ############### 2. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) # includes the final-time condition
    t=T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V) # uk(t_n)
        uhat_n_fun = vec_to_function(uhat[start : end], V) # uhat(t_n)
        
        p_rhs =  np.asarray(assemble((uhat_n_fun - uk_n_fun) * v * dx))
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors,
                                source_mat = M_u2 - M)
        
    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta*ck - pk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    
    print('Starting Armijo line search...')
    sk, u_inc = armijo_line_search(uk, ck, dk, uhat, num_steps, dt, M, 
                       c_lower, c_upper, beta, cost_fun_k, nodes,  V = V,
                       optim = 'alltime', dof_neighbors= dof_neighbors,
                       example = 'nonlinear')
     
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    cost_fun_kp1 = cost_functional(u_inc, uhat, ckp1, num_steps, dt, M, beta,
                           optim='alltime')
    print(f'{cost_fun_kp1=}')
    
    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals_u.append(L2_norm_sq_Q(u_inc - uhat, num_steps, dt, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    max_ck = []
    for i in range(num_steps):
        start = i * nodes
        end = (i+1) * nodes
        max_ck.append(np.amax(ck[start:end]))
    np.mean(max_ck)
    print('Mean of the control maxima over time steps:', np.mean(max_ck))
    mean_control_vals.append(np.mean(max_ck))
        
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
        uhat_re_t = uhat_re[startU : endU]
            
        u_re = u_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))
        uhat_re_t = uhat_re_t.reshape((sqnodes,sqnodes))
        
        if show_plots is True and (i%5 == 0 or i == num_steps-1):
            
            fig = plt.figure(figsize=(20, 5))

            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(uhat_re_t)
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
        im2_u = plt.plot(np.arange(1, it + 1), cost_fidelity_vals_u)
        plt.title('Data fidelity norms in L2(Omega)^2')
        plt.legend()
        
        ax2 = fig2.add_subplot(1, 3, 3)
        im3 = plt.plot(np.arange(1, it + 1), cost_control_vals)
        plt.title('Regularisation norm in L2(Q)^2')
        
        fig2.tight_layout(pad=3.0)
        plt.savefig(out_folder_name + f'/progress_plot.png')
        
        # Clear and remove objects explicitly
        ax2.clear()      # Clear axes
        del im1, im2_u, im3
        fig2.clf()
        plt.close(fig2)
        
        fig3 = plt.figure()
        im4 = plt.plot(np.arange(1, it + 1), mean_control_vals)
        plt.title('Mean of the max across time of the control at each iteration')
        plt.savefig(out_folder_name + f'/control_means_plot.png')
        
        # Clear and remove objects explicitly
        del im4
        fig3.clf()
        plt.close(fig3)
        
###############################################################################    
# Record the end time of the simulation
end_time = time.time()

# # Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)

uk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_u.csv', sep = ',')
ck.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_p.csv', sep = ',')

print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('Solutions saved to the folder:', out_folder_name)
print('L2_Q^2 (u - uhat)=', L2_norm_sq_Q(uk-uhat, num_steps, dt, M))

# print('Max. of the control at final time step:', np.amax(ck[num_steps*nodes:]))
# print('Min. of the control at final time step:', np.amin(ck[num_steps*nodes:]))
# print('Mean of the control at final time step:', np.mean(ck[num_steps*nodes:]))

eval_sim = 1/T * 1/((a2-a1)**2) * L2_norm_sq_Q(ck, num_steps, dt, M)
print(f'{eval_sim=}')

simulation_duration = end_time - start_time

# Prepare the data to be written to the CSV
data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "T": T,
    "beta": beta,
    "tol": tol,
    "c_lower": c_lower,
    "c_upper": c_upper,
    "eval_sim": eval_sim,
    "simulation_duration": simulation_duration,
    "out_folder_name": out_folder_name
}

csv_file_path = "NL_AT_simulation_results.csv"
file_exists = os.path.isfile(csv_file_path)

# Write the data to the CSV file
with open(csv_file_path, mode='a', newline='') as csv_file:
    fieldnames = ["timestamp", "T", "beta", "tol", "c_lower", "c_upper", "eval_sim", "simulation_duration", "out_folder_name"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header only if the file does not exist
    if not file_exists:
        writer.writeheader()
    
    # Write the data
    writer.writerow(data)
    