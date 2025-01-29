from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from helpers import *
from helpers import solve_nonlinear_equation

import csv
from datetime import datetime
import time

# General parameters
a1, a2 = (0, 1)
deltax = 0.1/2/2
intervals = round((a2-a1)/deltax)
beta = 0.1
dt = 0.001
T = 0.5
T_data = 2
num_steps = round(T/dt)
show_plots = True

# Box constraints for c
c_lower = -1
c_upper = 1

# Gradient descent parameters
tol = 10**-4
max_iter_armijo = 5
max_iter_GD = 50

# File paths
target_data_path = f"nonlinear_stripes_source_control_coarse/advection_t"
out_folder = f"ref_NL_FT_T{T}_Tdata{T_data}beta{beta}_Ca{c_lower}_Cb{c_upper}_tol{tol}"
if not Path(out_folder).exists():
    Path(out_folder).mkdir(parents=True)

###############################################################################
############################# Initialization  #################################
###############################################################################

mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals, intervals)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))
vertextodof = vertex_to_dof_map(V)
u = TrialFunction(V)
v = TestFunction(V)

# Mass matrix
M = assemble_sparse(u * v * dx)

# Explicitly create connectivities between vertices to find neighbouring nodes
mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

# Initial condition and target state at t=T
u0 = nonlinear_equation_IC(a1, a2, deltax, nodes, vertextodof)
uhat_T_re, uhat_T = import_data_final(target_data_path, f'_uT{T_data:03}', nodes, vertextodof)

vec_length = (num_steps + 1)*nodes # include zero and final time

# Initial guess for the control
ck = np.zeros(vec_length)

# Solve the state equation for the corresponding state
uk = np.zeros(vec_length)
uk[:nodes] = u0

uk, _ = solve_nonlinear_equation(ck, uk, None, V, nodes, num_steps, dt, dof_neighbors)

# Solve the adjoint equation
pk = np.zeros(vec_length)
pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
pk = solve_adjoint_nonlinear_equation(uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors)

cost_fun_k = cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, optim='finaltime')
cost_fun_kp1 = (2 + tol)*cost_fun_k
stop_crit = rel_err(cost_fun_kp1, cost_fun_k)

dk = np.zeros(vec_length)
eps, speed, k1, k2, _ = get_nonlinear_eqns_params()

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
fail_count = 0
fail_restart_count = 0
fail_pass = False
cost_fun_vals, cost_fidel_vals, cost_c_vals, mean_c_vals = ([] for _ in range(4))

print('Starting projected gradient descent method with the parameters:')
print(f'dx={deltax}, {dt=}, {T=}, {beta=}, {c_lower=}, {c_upper=}')
print(f'{eps=}, {speed=}, {k1=}, {k2=}')
print(f'{tol=}, {max_iter_GD=}, {max_iter_armijo=}, {k2=}')

start_time = time.time()

while (stop_crit >= tol or fail_pass) and it < max_iter_GD:
    print(f'\n{it=}')

    ## 1. choose the descent direction 
    dk = -(beta*ck - pk)
    
    ## 2. Find optimal stepsize with Armijo line search and calculate uk, ck
    print('Starting Armijo line search...')
    uk, ck, iters = armijo_line_search_ref(uk, ck, dk, uhat_T, num_steps, dt, 
                       c_lower, c_upper, beta, cost_fun_k, nodes, 'finaltime', 
                       V, dof_neighbors=dof_neighbors, 
                       nonlinear_solver=solve_nonlinear_equation, max_iter=max_iter_armijo)
    
    ## 3. Solve the adjoint equation using ukp1, ckp1
    pk = solve_adjoint_nonlinear_equation(uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors)
    
    ## 4. Calculate metrics
    cost_fun_kp1 = cost_functional(u_inc, uhat_T, ckp1, num_steps, dt, M, beta,
                           optim='finaltime')
    stop_crit = rel_err(cost_fun_kp1, cost_fun_k)
    eval_sim = 1/T * 1/((a2-a1)**2) * L2_norm_sq_Q(ck, num_steps, dt, M)

    print(f'{cost_fun_kp1=}')
    print(f'{eval_sim=}')

    cost_fun_vals.append(cost_fun_kp1)
    cost_fidel_vals.append(L2_norm_sq_Omega(u_inc[num_steps*nodes:] - uhat_T, M))
    cost_c_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
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
        
        u_re = uk_re[startU : endU].reshape((sqnodes,sqnodes))
        c_re = ck_re[startP : endP].reshape((sqnodes,sqnodes))
        p_re = pk_re[startP : endP].reshape((sqnodes,sqnodes))
            
        if show_plots is True and (i%20 == 0 or i == num_steps-1):
            
            fig = plt.figure(figsize=(20, 5))

            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(uhat_T_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $u$ taken at t={T_data}')
        
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
            plt.savefig(out_folder + f'/it_{it}_plot_{i:03}.png')
        
            # Clear and remove objects explicitly
            ax.clear()      # Clear axes
            cb1.remove()    # Remove colorbars
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
        im2_u = plt.plot(np.arange(1, it + 1), cost_fidel_vals)
        plt.title('Data fidelity norms in L2(Omega)^2')
        
        ax2 = fig2.add_subplot(1, 3, 3)
        im3 = plt.plot(np.arange(1, it + 1), cost_c_vals)
        plt.title('Regularisation norm in L2(Q)^2')
        
        fig2.tight_layout(pad=3.0)
        plt.savefig(out_folder + f'/progress_plot.png')
        
        # Clear and remove objects explicitly
        ax2.clear()      # Clear axes
        del im1, im2_u, im3
        fig2.clf()
        plt.close(fig2)
        
        fig3 = plt.figure()
        im4 = plt.plot(np.arange(1, it + 1), mean_c_vals)
        plt.title('Mean of the max across time of the control at each iteration')
        plt.savefig(out_folder + f'/control_means_plot.png')
        
        # Clear and remove objects explicitly
        del im4
        fig3.clf()
        plt.close(fig3)
    
        
    ## Make updates
    it += 1
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    
    print(f'{stop_crit=}')

###############################################################################    
# Record the end time of the simulation
end_time = time.time()

if fail_count == 3  or fail_restart_count == 5 or (it == max_iter_GD and fail_count > 0):
    print(f'Restoring the solutions from iteration {it_backup}')
    uk = u_backup
    pk = p_backup
    ck = c_backup
    
# Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)

uk.tofile(out_folder + f'/Schnak_adv_T{T}_beta{beta}_u.csv', sep = ',')
ck.tofile(out_folder + f'/Schnak_adv_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder + f'/Schnak_adv_T{T}_beta{beta}_p.csv', sep = ',')

print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('Solutions saved to the folder:', out_folder)
print('L2_\Omega^2 (u(T) - uhat_T)=', L2_norm_sq_Omega(uk[num_steps*nodes:]-uhat_T,M))

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
    "out_folder_name": out_folder
}

csv_file_path = "NL_FT_simulation_results.csv"
file_exists = os.path.isfile(csv_file_path)

# Write the data to the CSV file
with open(csv_file_path, mode='a', newline='') as csv_file:
    fieldnames = ["timestamp", "T", "beta", "tol", "c_lower", "c_upper", "eval_sim", "simulation_duration", "out_folder_name"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header only if the file does not exist
    if not file_exists:
        writer.writeheader()
    
    writer.writerow(data)