from pathlib import Path
from datetime import datetime
import os
import csv
import time
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to solve the PDECO problem with a chemotaxis system (final-time optim.)
# (uses the variable names u,v instead of m,f, respectively)
# ----------------------------------------------------------------------------

"""
Solves the PDECO problem below with projected gradient descent method and FCT
Cost functional:
J(u,v,c) = 1/2*||u(T) - û_T||² + 1/2*||v(T) - v̂_T||² + β/2*||c||²
(misfit: L²-norms over Ω,  regularization: L²-norm over Ω × [0,T])

min_{u,v,c} J(u,v,c)
subject to:  
  du/dt + ∇⋅(-Dm*∇u + X*u*exp(-ηu)*∇v) = 0      in Ω × [0,T]
              dv/dt + ∇⋅(-Dv*∇v) + δ*v = c*(u/r)    in Ω × [0,T]
          (-Df*∇u + X*u*exp(-ηu)*∇v)⋅n = 0      on ∂Ω × [0,T]
                                 ∇v⋅n = 0       on ∂Ω × [0,T]
                                 u(0) = u0(x)    in Ω
                                 v(0) = v0(x)    in Ω
                                 c in [ca,cb]
(X = chi, control to be recovered: c_orig = 100)
r is rescaling coefficient. If r=1/10, then c = 1/10*c_orig = 10, if r=1, then c=c_orig.
Additional optimality conditions:
- Adjoint equations, BCs and final-time conditions
  -dp/dt + ∇⋅(-Dm*∇p) - X*(1 - η*u)*exp(-ηu)*∇p⋅∇v = c*(q/r)       in Ω x [0,T]
       -dq/dt + ∇⋅(-Df*∇q + X*u*exp(-ηu)*∇p) + δ*q = 0          in Ω x [0,T]
                                       ∇p⋅n = ∇q⋅n = 0          on ∂Ω x [0,T]
                                               p(T) = û_T - u(T)  in Ω
                                               q(T) = v̂_T - v(T)  in Ω
- Gradient equation:  β*c - q*u = 0   in Ωx[0,T]
"""

# ---------------------------- General Parameters ----------------------------
a1, a2 = 0, 1
dx = 0.025 # Element size
intervals = round((a2-a1)/dx)

dt = 0.001/2
T = 200*dt
T_data = T
num_steps = round(T / dt)

produce_plots = True # Toggle for visualization
optim = "finaltime"

# ---------------------------- PDECO parameters ------------------------------

beta = 1e-4 # Regularization parameter

# Box constraints for c
c_lower = 0
c_upper = 20

delta, Dm, Df, chi, true_control, eta = hp.get_chtxs_sys_params()
rescaling = 1/10 # also need to adjust in the solvers for state and adjoint equations
true_control = true_control*rescaling

# --------------------- Gradient descent parameters --------------------------

tol = 1e-4
max_iter_armijo = 10
max_iter_GD = 50
armijo_gamma = 1e-5
armijo_s0 = 2

# ----------------------- Input & Output file paths --------------------------

# target_data_path = "chtx_chi0.25_simplfeathers_dx0.005_Jan"
target_data_path = f"Chtxs_data_T100_dx{dx}_dt{dt}"#"Chtxs_data_T100_coarse"
target_data_file_name_u = f"chtxs_m"
target_data_file_name_v = f"chtxs_f"
# target_file_u = os.path.join(target_data_path, f"{target_data_file_name_u}_t{T_data}.csv")
# target_file_v = os.path.join(target_data_path, f"{target_data_file_name_v}_t{T_data}.csv")
target_file_u = os.path.join(target_data_path, f"{target_data_file_name_u}_t0.5.csv")
target_file_v = os.path.join(target_data_path, f"{target_data_file_name_v}_t0.5.csv")

out_folder = f"Chtx_FT_T{T}_Tdata{T_data}_dt{dt}_beta{beta}_Ca{c_lower}_Cb{c_upper}_tol{tol}"
if not Path(out_folder).exists():
    Path(out_folder).mkdir(parents=True)
   
print(f"dx={dx}, {dt=}, {T=}, {T_data=}, {beta=}, {c_lower=}, {c_upper=}")
print(f"{Dm=}, {Df=}, {delta=}, {chi=}, {eta=}")
print(f"{tol=}, {max_iter_GD=}, {max_iter_armijo=}, {armijo_gamma=}, {armijo_s0=}")
print(f"{rescaling=}")

# ------------------------------ Initialization ------------------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
W = df.VectorFunctionSpace(mesh, "CG", 1)
nodes = V.dim()
sqnodes = round(np.sqrt(nodes))
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
w = df.TestFunction(V)

M = hp.assemble_sparse(u * w * df.dx) # Mass matrix

# Explicitly create connectivities between vertices to find neighbouring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

u0, v0 = hp.chtxs_sys_IC(a1, a2, dx, nodes, vertex_to_dof)

## choose target states as true solutions
uhat_T_re, uhat_T = hp.import_data_final(target_file_u, nodes, vertex_to_dof,
                                         num_steps=num_steps)
vhat_T_re, vhat_T = hp.import_data_final(target_file_v, nodes, vertex_to_dof,
                                         num_steps=num_steps)

# ## Target states interpolation for initialization of uk, vk:
# sqnodes = round(np.sqrt(nodes))
# uhat_interpol = np.zeros((num_steps+1)*nodes)
# vhat_interpol = np.zeros((num_steps+1)*nodes)
# for i in range(num_steps+1): # includes states at time zero
#     start = i*nodes
#     end = (i+1)*nodes
#     uhat_interpol[start:end] = i*dt /T * uhat_T
#     vhat_interpol[start:end] = i*dt /T * vhat_T

# ----------------- Initialize gradient descent variables --------------------

vec_length = (num_steps + 1) * nodes # Include zero and final time

# Initial guess for the control
ck = np.zeros(vec_length)

# Solve the state equation for the corresponding state
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)
uk[:nodes] = u0
vk[:nodes] = v0
uk, vk = hp.solve_chtxs_system(
    ck, uk, vk, V, nodes, num_steps, dt, dof_neighbors)

# uk = 0.8*np.copy(uhat_interpol)
# vk = 0.8*np.copy(vhat_interpol)
# uk[:nodes] = u0
# vk[:nodes] = v0

# Solve the adjoint equation
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
pk, qk = hp.solve_adjoint_chtxs_system(uk, vk, uhat_T, vhat_T, pk, qk, ck, T, 
                                       V, nodes, num_steps, dt, dof_neighbors, optim,
                                       mesh=mesh, deltax=dx, vertex_to_dof=vertex_to_dof, rescaling=rescaling)


# Calculate initial cost functional
cost_fun_old = hp.cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, 
                               optim, var2=vk, var2_target=vhat_T)
cost_fun_new = (2 + tol) * cost_fun_old
stop_crit = hp.rel_err(cost_fun_new, cost_fun_old)

dk = np.zeros(vec_length)

it = 0
fail_count = 0
fail_restart_count = 0
fail_count_max = 5
fail_restart_count_max = 5
fail_pass = False
cost_fun_vals, cost_fidel_vals_u, cost_fidel_vals_v, cost_c_vals, armijo_its = ([] for _ in range(5))
cost_fun_vals.append(cost_fun_old)

start_time = time.time()

##############################################################################
# ------------------------ PROJECTED GRADIENT DESCENT ------------------------
##############################################################################

while (stop_crit >= tol or fail_pass or it < 2) and it < max_iter_GD:
    print(f"\n{it=}")
    
    ## 1. choose the descent direction 
    dk = -(beta*ck - qk*uk/rescaling)
    
    ### Preconditioner approach
    # #1. Preconditioner M = diag(max |uk * qk / rescaling|)
    # Prec_dk = diags(np.max(np.abs(uk * qk/rescaling)) * np.ones_like(qk), offsets=0, format="csc")
    # dk = spsolve(Prec_dk, -(beta*ck - qk*uk/rescaling))
    # print("Using preconditioned dk")

    ## 2. Find optimal stepsize with Armijo line search and calculate uk, ck
    print("Starting Armijo line search...")
    uk, vk, ck, iters = hp.armijo_line_search_ref(uk, ck, dk, uhat_T, num_steps, dt, 
                       c_lower, c_upper, beta, cost_fun_old, nodes, optim, 
                       V, dof_neighbors=dof_neighbors, var2=vk, var2_target=vhat_T,
                        nonlinear_solver=hp.solve_chtxs_system, max_iter=max_iter_armijo,
                        gam=armijo_gamma, s0=armijo_s0)

    ## 3. Solve the adjoint equation using new uk and vk
    pk, qk = hp.solve_adjoint_chtxs_system(uk, vk, uhat_T, vhat_T, pk, qk, ck, T, 
                                           V, nodes, num_steps, dt, dof_neighbors, optim,
                                           mesh=mesh, deltax=dx, vertex_to_dof=vertex_to_dof, rescaling=rescaling)
    
    if iters == max_iter_armijo:
        fail_count += 1
        fail_pass = True
        if it == 0:
            # Save the current solution as the last best solution
            u_backup = uk
            v_backup = vk
            p_backup = pk
            q_backup = qk
            c_backup = ck
            it_backup = it
            
        if fail_count == fail_count_max:
            # end while loop, assume we have found the most optimal solution
            print("Maximum number of failed Armijo line search iterations reached. Exiting...")
            break
    
    elif iters < max_iter_armijo:
        if fail_count > 0:
            # If armijo converged after a fail, reset the counter
            fail_count = 0
            fail_restart_count += 1
            fail_pass = False
        
        if fail_restart_count < fail_restart_count_max:
            # Save the current solution as the last best solution
            u_backup = uk
            v_backup = vk
            p_backup = pk
            q_backup = qk
            c_backup = ck
            it_backup = it
        elif fail_restart_count == fail_restart_count_max:
            # End while loop, assume we have found the most optimal solution
            print("Maximum number of restarts reached. Exiting...")
            break
    
    ## 4. Calculate metrics
    cost_fun_new = hp.cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, 
                                   optim, var2=vk, var2_target=vhat_T)
    stop_crit = hp.rel_err(cost_fun_new, cost_fun_old)
    eval_sim = 1/T * 1/((a2-a1)**2) * hp.L2_norm_sq_Q(ck, num_steps, dt, M)
   
    print(f"{cost_fun_new=}")
    print(f"{eval_sim=}")
   
    cost_fun_vals.append(cost_fun_new)
    cost_fidel_vals_u.append(hp.L2_norm_sq_Omega(uk[num_steps*nodes:] - uhat_T, M))
    cost_fidel_vals_v.append(hp.L2_norm_sq_Omega(vk[num_steps*nodes:] - vhat_T, M))
    cost_c_vals.append(hp.L2_norm_sq_Q(ck, num_steps, dt, M))
    armijo_its.append(iters)

    if produce_plots is True:
        hp.plot_two_var_solution(
            uk, vk, pk, qk, ck, uhat_T_re, vhat_T_re, T_data, it, nodes, 
            num_steps, dt, out_folder, vertex_to_dof, optim, step_freq=2)

    hp.plot_progress(
        cost_fun_vals, cost_fidel_vals_u, cost_c_vals, it, out_folder, 
        cost_fidel_vals2=cost_fidel_vals_v, v1_name="u", v2_name="v")

    ## Make updates
    it += 1
    cost_fun_old = cost_fun_new

    print(f"Stopping criterion: {stop_crit}")
    
# --------------------------- Save results -----------------------------------
      
# Record the end time of the simulation
end_time = time.time()
simulation_duration = end_time - start_time

if fail_count == fail_count_max or fail_restart_count == fail_restart_count_max or (it == max_iter_GD and fail_count > 0):
    print(f"Restoring the solutions from iteration {it_backup}")
    uk = u_backup
    vk = v_backup
    pk = p_backup
    qk = q_backup
    ck = c_backup

eval_sim = 1/T * 1/((a2-a1)**2) * hp.L2_norm_sq_Q(ck, num_steps, dt, M)
misfit_norm_u = cost_fidel_vals_u[-1]
misfit_norm_v = cost_fidel_vals_v[-1]
control_as_td_vector = true_control * np.ones(vec_length)
true_control_norm = hp.L2_norm_sq_Q(control_as_td_vector, num_steps, dt, M)

uk.tofile(out_folder + "/Chtxs_u.csv", sep = ",")
vk.tofile(out_folder + "/Chtxs_v.csv", sep = ",")
ck.tofile(out_folder + "/Chtxs_c.csv", sep = ",")
pk.tofile(out_folder + "/Chtxs_p.csv", sep = ",")
qk.tofile(out_folder + "/Chtxs_q.csv", sep = ",")

# Prepare the data to be written to the CSV
data = {"timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
    "Sim. duration" : round(simulation_duration, 2), "T" : T, "T_data" : T_data, 
    "beta" : beta, "tol" : tol, "GD its" : it, "Armijo its" : armijo_its, 
    "C_ad" : f"[{c_lower}, {c_upper}]", "Mean c. in L^2(Q)^2" : eval_sim, 
    "Misfit norm u" : misfit_norm_u, "Misfit norm v" : misfit_norm_v,
    "J(c_true)" : beta/2*true_control_norm, "out_folder_name" : out_folder}

csv_file_path = "Chtx_FT_simulation_results.csv"
file_exists = os.path.isfile(csv_file_path)

# Write the data to the CSV file
with open(csv_file_path, mode="a", newline="") as csv_file:
    fieldnames = ["timestamp", "Sim. duration", "T", "T_data", "beta", "tol", 
                  "GD its",  "Armijo its", "C_ad", "Mean c. in L^2(Q)^2",  
                  "Misfit norm u", "Misfit norm v", "J(c_true)", "out_folder_name"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header only if the file does not exist
    if not file_exists:
        writer.writeheader()

    writer.writerow(data)

print(f"\nExit:\nFinal stopping criterion: {stop_crit} \nIterations: {it}")
print("Armijo iterations:", armijo_its)
print("Solutions saved to:", out_folder)
print("||u(T) - û_T|| in L^2(Ω)^2 :", misfit_norm_u)
print("||v(T) - v̂_T|| in L^2(Ω)^2 :", misfit_norm_v)
print("Average control in L^2(Q)^2:", eval_sim)
print(f"Final cost functional value for Ω × [0,{T}]:", cost_fun_new)
print(f"β/2 *||c_true|| in L^2-norm^2 over Ω × [0,{T_data}]:", beta/2*true_control_norm)
print("Control mean at t=T:", np.mean(ck[num_steps*nodes:]))
