from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from helpers import *

import csv
from datetime import datetime
import time
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
# linearise equation for m
    
# Optimality conditions:
#        dm/dt - Dm*grad^2(m) + div(chi*m*grad(f)) = m(4-m)       in Ωx[0,T]
#                  df/dt - Df*grad^2(f)) + delta*f = c*m          in Ωx[0,T]
#    -dp/dt - Dm*grad^2 p - chi*grad(p)*grad(f)) \
#                                 - chi*(4-2mk)*p - cq = 0         in Ωx[0,T]
#    -dq/dt - Df*grad^2(q)) + div(chi*m*grad(p)) + delta*q = 0     in Ωx[0,T]
#            dm/dn = df/dn = dp/dn = dq/dn = 0                     on ∂Ωx[0,T]
#                                     u(0) = u0(x)                 in Ω
#                                     p(T) = 0                     in Ω
# gradient equation:           c = proj_[ca,cb] (1 / beta*q * m)   in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.005
intervals_line = round((a2 - a1) / deltax)
beta = 0.1
# box constraints for c, exact solution is in ?
c_upper = 0 
c_lower = 100

delta = 100
Dm = 0.05
Df = 0.05
chi = 0.25
gamma = 100

t0 = 0
dt = 0.1
T = 5
T_data = T
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
v = TestFunction(V)

show_plots = True
example_name = f"chtx_chi{chi}_simplfeathers_dx{deltax}_Jan/chtx"
out_folder_name = f"chtx_FT_T{T}_beta{beta}_Ca{c_upper}_Cb{c_lower}_tol{tol}_chi{chi}"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M, nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# System matrix: equation for f and q
Mat_fq = M + dt * (Df * Ad + delta * M)

###############################################################################
################ Target states & initial conditions for m,f ###################
###############################################################################

m0_orig = mimura_data_helpers.m_initial_condition(a1, a2, deltax).reshape(nodes)
f0_orig = m0_orig
m0 = reorder_vector_to_dof_time(m0_orig, 1, nodes, vertextodof)
f0 = reorder_vector_to_dof_time(f0_orig, 1, nodes, vertextodof)

mhat_T = np.genfromtxt(example_name + f'_m_t{T_data:.1f}.csv', delimiter=',')
fhat_T = np.genfromtxt(example_name + f'_f_t{T_data:.1f}.csv', delimiter=',')
mhat_T_re = reorder_vector_from_dof_time(mhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))
fhat_T_re = reorder_vector_from_dof_time(fhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1) * nodes # include zero and final time

mk = np.zeros(vec_length)
fk = np.zeros(vec_length)
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
ck = np.zeros(vec_length)
dk = np.zeros(vec_length)

mk[:nodes] = m0
fk[:nodes] = f0

w_mk = np.copy(mk)
w_fk = np.copy(fk)

sk = 0
###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional(mk, mhat_T, ck, num_steps, dt, M, beta, 
                                var2 = fk, var2_target = fhat_T,
                                optim='finaltime')
cost_fun_vals = []
cost_fidelity_vals_m = []
cost_fidelity_vals_f = []
cost_control_vals = []
mean_control_vals = []
stop_crit = 5
max_iter_armijo = 10
print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')

# Record the start time of the simulation
start_time = time.time()

while stop_crit >= tol and it<100:
    it += 1
    print(f'\n{it=}')
        
    # In k-th iteration we solve for f^k, m^k, q^k, p^k using c^k (S1 & S2)
    # and calculate c^{k+1} (S5)
    
    ###########################################################################
    ############## Solve the state equations using FCT for m ##################
    ###########################################################################
    
    print('Solving state equations...')
    t = 0
    # initialise m,f and keep ICs
    fk[nodes :] = np.zeros(num_steps * nodes)
    mk[nodes :] = np.zeros(num_steps * nodes)
    for i in range(1, num_steps + 1):    # solve for fk(t_{n+1}), mk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        m_n = mk[start - nodes : start]    # mk(t_n) 
        m_n_fun = vec_to_function(m_n,V)
        c_np1_fun = vec_to_function(ck[start : end],V)
        f_n_fun = vec_to_function(fk[start - nodes : start],V)
        
        # f_rhs = rhs_chtx_f(f_n_fun, m_n_fun, c_np1_fun, dt, v)
        f_rhs = np.asarray(assemble(f_n_fun * v * dx  + dt * c_np1_fun * m_n_fun * v * dx))

        fk[start : end] = spsolve(Mat_fq, f_rhs)
        
        f_np1_fun = vec_to_function(fk[start : end], V)

        A_m = mat_chtx_m(f_np1_fun, m_n_fun, Dm, chi, u, v)
        # m_rhs = rhs_chtx_m(m_n_fun, v)
        m_rhs = np.zeros(nodes)

        mk[start : end] =  FCT_alg(A_m, m_rhs, m_n, dt, nodes, M, M_Lump, dof_neighbors)

    
    ###########################################################################
    ############## Solve the adjoint equations using FCT for p ################
    ###########################################################################
    
    qk = np.zeros(vec_length) 
    pk = np.zeros(vec_length)
    # insert final-time condition
    qk[num_steps * nodes :] = fhat_T - fk[num_steps * nodes :]
    pk[num_steps * nodes :] = mhat_T - mk[num_steps * nodes :]
    t = T
    print('Solving adjoint equations...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        q_np1 = qk[end : end + nodes] # qk(t_{n+1})
        p_np1 = pk[end : end + nodes] # pk(t_{n+1})

        p_np1_fun = vec_to_function(p_np1, V) 
        q_np1_fun = vec_to_function(q_np1, V)
        m_n_fun = vec_to_function(mk[start : end], V)      # mk(t_n)
        f_n_fun = vec_to_function(fk[start : end], V)      # fk(t_n)
        c_n_fun = vec_to_function(ck[start : end], V)      # ck(t_n)

        q_rhs = rhs_chtx_q(q_np1_fun, m_n_fun, p_np1_fun, chi, dt, v)
        
        qk[start:end] = spsolve(Mat_fq, q_rhs)
        
        q_n_fun = vec_to_function(qk[start : end], V)      # qk(t_n)

        A_p = mat_chtx_p(f_n_fun, m_n_fun, Dm, chi, u, v)
        ### !! check the reaction matrices for m and p
        p_rhs = rhs_chtx_p(c_n_fun, q_n_fun, v)
        
        pk[start:end] = FCT_alg(A_p, p_rhs, p_np1, dt, nodes, M, M_Lump, dof_neighbors)
        
    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta * ck - qk * mk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    # initialise w_m, w_f and keep ICs
    w_fk[nodes :] = np.zeros(num_steps * nodes)
    w_mk[nodes :] = np.zeros(num_steps * nodes)
    print('Solving equations for move in m, f...')
    for i in range(1, num_steps + 1):  # solve for w_fk(t_{n+1}), w_mk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        w_m_n = w_mk[start - nodes : start]    # w_mk(t_n) 
        w_m_n_fun = vec_to_function(w_m_n,V)
        dk_np1_fun = vec_to_function(dk[start : end], V)
        w_f_n_fun = vec_to_function(w_fk[start - nodes : start],V)
        
        w_f_rhs = np.asarray(assemble(w_f_n_fun * v * dx  + dt * dk_np1_fun * w_m_n_fun * v * dx))

        w_fk[start : end] = spsolve(Mat_fq, w_f_rhs)
        
        w_f_np1_fun = vec_to_function(w_fk[start : end], V)

        A_m = mat_chtx_m(w_f_np1_fun, w_m_n_fun, Dm, chi, u, v)
        w_m_rhs = np.zeros(nodes)

        w_mk[start : end] =  FCT_alg(A_m, w_m_rhs, w_m_n, dt, nodes, M, M_Lump, dof_neighbors)
        
    print('Starting Armijo line search...')
    sk, m_inc, f_inc = armijo_line_search(mk, ck, dk, mhat_T, num_steps, dt, M, 
                          c_lower, c_upper, beta, cost_fun_k, nodes,  V = V,
                          optim = 'finaltime', dof_neighbors= dof_neighbors,
                          example = 'chtxs', var2 = fk, var2_target = fhat_T,
                          w1 = w_mk, w2 = w_fk, max_iter = max_iter_armijo)
    
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk * dk, c_lower, c_upper)
    cost_fun_kp1 = cost_functional(m_inc, mhat_T, ckp1, num_steps, dt, M, beta,
                           optim='finaltime', var2 = f_inc, var2_target=fhat_T)
    print(f'{cost_fun_kp1=}')

    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals_m.append(L2_norm_sq_Omega(m_inc[num_steps*nodes:] - mhat_T, M))
    cost_fidelity_vals_f.append(L2_norm_sq_Omega(f_inc[num_steps*nodes:] - fhat_T, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    max_ck = []
    for i in range(num_steps):
        start = i * nodes
        end = (i+1) * nodes
        max_ck.append(np.amax(ck[start:end]))
    print('Mean of the control maxima over time steps:', np.mean(max_ck))
    mean_control_vals.append(np.mean(max_ck))
        
    stop_crit = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit=}')
    
    mk_re = reorder_vector_from_dof_time(mk, num_steps + 1, nodes, vertextodof)
    fk_re = reorder_vector_from_dof_time(fk, num_steps + 1, nodes, vertextodof)
    ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
    pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
    qk_re = reorder_vector_from_dof_time(qk, num_steps + 1, nodes, vertextodof)

    for i in range(num_steps):
        startP = i * nodes
        endP = (i+1) * nodes
        tP = i * dt
        
        startU = (i+1) * nodes
        endU = (i+2) * nodes
        tU = (i+1) * dt
        
        m_re = mk_re[startU : endU]
        f_re = fk_re[startU : endU]
        c_re = ck_re[startP : endP]
        p_re = pk_re[startP : endP]
        q_re = qk_re[startP : endP]
            
        m_re = m_re.reshape((sqnodes,sqnodes))
        f_re = f_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))
        q_re = q_re.reshape((sqnodes,sqnodes))
        
        if show_plots is True and (i%5 == 0 or i == num_steps-1):
            
            fig = plt.figure(figsize=(20, 10))

            ax = fig.add_subplot(2, 4, 1)
            im1 = ax.imshow(mhat_T_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $m$ at t = {T}')
        
            ax = fig.add_subplot(2, 4, 2)
            im2 = ax.imshow(m_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f'Computed state $m$ at t = {round(tU, 5)}')
        
            ax = fig.add_subplot(2, 4, 3)
            im3 = ax.imshow(p_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f'Computed adjoint $p$ at t = {round(tP, 5)}')
        
            ax = fig.add_subplot(2, 4, 4)
            im4 = ax.imshow(c_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f'Computed control $c$ at t = {round(tP, 5)}')
            
            ax = fig.add_subplot(2, 4, 5)
            im5 = ax.imshow(fhat_T_re)
            cb5 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $f$ at t = {T}')
        
            ax = fig.add_subplot(2, 4, 6)
            im6 = ax.imshow(f_re)
            cb6 = fig.colorbar(im2, ax=ax)
            ax.set_title(f'Computed state $f$ at t = {round(tU, 5)}')
        
            ax = fig.add_subplot(2, 4, 7)
            im7 = ax.imshow(q_re)
            cb7 = fig.colorbar(im3, ax=ax)
            ax.set_title(f'Computed adjoint $q$ at t = {round(tP, 5)}')
        
            fig.tight_layout(pad=3.0)
            plt.savefig(out_folder_name + f'/it_{it}_plot_{i:03}.png')
        
            # Clear and remove objects explicitly
            ax.clear()      # Clear axes
            cb1.remove()     # Remove colorbars
            cb2.remove()
            cb3.remove()
            cb4.remove()
            cb5.remove()
            cb6.remove()
            cb7.remove()
            del im1, im2, im3, im4, im5, im6, im7, cb1, cb2, cb3, cb4, cb5, cb6, cb7
            fig.clf()
            plt.close(fig) 
            
    if it > 1:
        fig2 = plt.figure(figsize=(15, 5))

        ax2 = fig2.add_subplot(1, 3, 1)
        im1 = plt.plot(np.arange(1, it + 1), cost_fun_vals)
        plt.title(f'{it=} Cost functional')
        
        ax2 = fig2.add_subplot(1, 3, 2)
        im2_u = plt.plot(np.arange(1, it + 1), cost_fidelity_vals_m, label='for m')
        im2_v = plt.plot(np.arange(1, it + 1), cost_fidelity_vals_f, label='for f')
        plt.title('Data fidelity norms in L2(Omega)^2')
        plt.legend()
        
        ax2 = fig2.add_subplot(1, 3, 3)
        im3 = plt.plot(np.arange(1, it + 1), cost_control_vals)
        plt.title('Regularisation norm in L2(Q)^2')
        
        fig2.tight_layout(pad=3.0)
        plt.savefig(out_folder_name + f'/progress_plot.png')
        
        # Clear and remove objects explicitly
        ax2.clear()      # Clear axes
        del im1, im2_u, im2_v, im3
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

# Mapping to order the solution vectors based on vertex indices
mk_re = reorder_vector_from_dof_time(mk, num_steps + 1, nodes, vertextodof)
fk_re = reorder_vector_from_dof_time(fk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
qk_re = reorder_vector_from_dof_time(qk, num_steps + 1, nodes, vertextodof)

mk.tofile(out_folder_name + f'/chtxs_T{T}_beta{beta}_m.csv', sep = ',')
fk.tofile(out_folder_name + f'/chtxs_T{T}_beta{beta}_f.csv', sep = ',')
ck.tofile(out_folder_name + f'/chtxs_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder_name + f'/chtxs_T{T}_beta{beta}_p.csv', sep = ',')
qk.tofile(out_folder_name + f'/chtxs_T{T}_beta{beta}_q.csv', sep = ',')

print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('Solutions saved to the folder:', out_folder_name)
print('L2_\Omega^2 (m(T) - mhat_T)=', L2_norm_sq_Omega(mk[num_steps*nodes]-mhat_T,M))
print('L2_\Omega^2 (f(T) - fhat_T)=', L2_norm_sq_Omega(fk[num_steps*nodes]-fhat_T,M))

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

csv_file_path = "chtx_FT_simulation_results.csv"
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