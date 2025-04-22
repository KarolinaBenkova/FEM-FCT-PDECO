from pathlib import Path
from datetime import datetime
import os
import csv
import time
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to solve the PDECO problem with advective Schnakenberg equations
# -----------------------------------------------------------------=----------

"""
Solves the PDECO problem below with projected gradient descent method and FCT.
Cost functional:
J(u,v,c) = 1/2*||u(T) - û_T||² + 1/2*||v(T) - v̂_T||² + β/2*||c||²
(misfit: L²-norms over Ω,  regularization: L²-norm over Ω × [0,T])

min_{u,v,c} J(u,v,c)
subject to:
 du/dt + ∇⋅(-Du*∇u + ω1*w*u) + γ(u-u²v) = γ*(c/r)     in Ω × [0,T]
 dv/dt + ∇⋅(-Dv*∇v + ω2*w*v) + γ(u²v-b) = 0       in Ω × [0,T]
                      (-Du*∇u + ω1*w*u)⋅n = 0       on ∂Ω × [0,T]
                      (-Dv*∇v + ω2*w*v)⋅n = 0       on ∂Ω × [0,T]
                                     u(0) = u0(x)   in Ω
                                     v(0) = v0(x)   in Ω
                                     c in [ca,cb]
where w is a velocity/wind vector satisfying:
  ∇⋅w = 0  in Ω × [0,T]
r is rescaling coefficient. If r=100, then c = 100*a = 10, if r=1, then c=a.

Additional optimality conditions:
- Adjoint equations, BCs and final-time conditions
  -dp/dt + ∇⋅(-Du*∇p - ω1*w*p) + γ*p + 2*γ*u*v*(q-p) = 0      in Ω x [0,T]
          -dq/dt + ∇⋅(-Dv*∇q - ω2*w*q) + γ*u²*(q-p) = 0      in Ω x [0,T]
                                         ∇p⋅n = ∇q⋅n = 0     on ∂Ω x [0,T]
                                                 p(T) = û_T - u(T)  in Ω
                                                 q(T) = v̂_T - v(T)  in Ω
- Gradient equation:  β*c - γ*p = 0   in Ω x [0,T]
"""

# ---------------------------- General Parameters ----------------------------
a1, a2 = 0, 1
dx = 0.025 # Element size
intervals = round((a2-a1)/dx)

dt = 0.001
T = 0.5
T_data = 1
num_steps = round(T / dt)

produce_plots = True # Toggle for visualization
optim = "finaltime"

# ---------------------------- PDECO parameters ------------------------------

beta = 1e-1 # Regularization parameter

# Box constraints for c
c_lower = 0
c_upper = 10

Du, Dv, true_control, c_b, gamma, omega1, omega2, wind = hp.get_schnak_sys_params()
rescaling = 1 #100
true_control = true_control*rescaling

# --------------------- Gradient descent parameters --------------------------

tol = 1e-3
max_iter_armijo = 10
max_iter_GD = 50

# ----------------------- Input & Output file paths --------------------------

target_data_path = "AdvSchnak_data_T2_stationarywind1_corr"
target_data_file_name_u = "schnak_u"
target_data_file_name_v = "schnak_v"
target_file_u = os.path.join(target_data_path, f"{target_data_file_name_u}_T{T_data}.csv")
target_file_v = os.path.join(target_data_path, f"{target_data_file_name_v}_T{T_data}.csv")

out_folder = f"ref_Sch_FT_T{T}_Tdata{T_data}_beta{beta}_Ca{c_lower}_Cb{c_upper}_tol{tol}_rescaled"
if not Path(out_folder).exists():
    Path(out_folder).mkdir(parents=True)
   
print(f"dx={dx}, {dt=}, {T=}, {T_data=}, {beta=}, {c_lower=}, {c_upper=}")
print(f"{Du=}, {Dv=}, {c_b=}, {gamma=}, {omega1=}, {omega2=}, {true_control=}")
print(f"{tol=}, {max_iter_GD=}, {max_iter_armijo=}")

# ------------------------------ Initialization ------------------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
# W = df.VectorFunctionSpace(mesh, "CG", 1)
nodes = V.dim()
sqnodes = round(np.sqrt(nodes))
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
w = df.TestFunction(V)

M = hp.assemble_sparse(u * w * df.dx) # Mass matrix

# Explicitly create connectivities between vertices to find neighbouring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

u0, v0 = hp.schnak_sys_IC(a1, a2, dx, nodes, vertex_to_dof)

if not os.path.exists(target_file_u):
    hp.extract_data(
        target_data_path, target_data_file_name_u, T_data, dt, nodes, vertex_to_dof)
if not os.path.exists(target_file_v):
    hp.extract_data(
        target_data_path, target_data_file_name_v, T_data, dt, nodes, vertex_to_dof)

uhat_T_re, uhat_T = hp.import_data_final(target_file_u, nodes, vertex_to_dof)
vhat_T_re, vhat_T = hp.import_data_final(target_file_v, nodes, vertex_to_dof)

# ----------------- Initialize gradient descent variables --------------------

vec_length = (num_steps + 1) * nodes # Include zero and final time

# Initial guess for the control
ck = np.zeros(vec_length)

# Solve the state equation for the corresponding state
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)
uk[:nodes] = u0
vk[:nodes] = v0
uk, vk = hp.solve_schnak_system(
    ck, uk, vk, V, nodes, num_steps, dt, dof_neighbors)

# Solve the adjoint equation
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
pk, qk = hp.solve_adjoint_schnak_system(uk, vk, uhat_T, vhat_T, pk, qk, T, V, 
                                        nodes, num_steps, dt, dof_neighbors)

# Calculate initial cost functional
cost_fun_old = hp.cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, 
                                optim, var2=vk, var2_target=vhat_T)
cost_fun_new = (2 + tol) * cost_fun_old
stop_crit = hp.rel_err(cost_fun_new, cost_fun_old)

dk = np.zeros(vec_length)

it = 0
fail_count = 0
fail_restart_count = 0
fail_pass = False
cost_fun_vals, cost_fidel_vals_u, cost_fidel_vals_v, cost_c_vals, armijo_its = ([] for _ in range(5))
cost_fun_vals.append(cost_fun_old)

start_time = time.time()

##############################################################################
# ------------------------ PROJECTED GRADIENT DESCENT ------------------------
##############################################################################

while (stop_crit >= tol or fail_pass) and it < max_iter_GD:
    print(f"\n{it=}")
    
    ## 1. choose the descent direction 
    # dk = -(beta*ck - gamma*pk)
    dk = -(beta*ck - gamma/rescaling*pk)
    
    ## 2. Find optimal stepsize with Armijo line search and calculate uk, ck
    print("Starting Armijo line search...")
    uk, vk, ck, iters = hp.armijo_line_search_ref(uk, ck, dk, uhat_T, num_steps, dt, 
                        c_lower, c_upper, beta, cost_fun_old, nodes, optim, 
                        V, dof_neighbors=dof_neighbors, var2=vk, var2_target=vhat_T,
                        nonlinear_solver=hp.solve_schnak_system, max_iter=max_iter_armijo)
      
    ## 3. Solve the adjoint equation using new uk and vk
    pk, qk = hp.solve_adjoint_schnak_system(uk, vk, uhat_T, vhat_T, pk, qk, T, 
                                  V, nodes, num_steps, dt, dof_neighbors)        
    
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
            
        if fail_count == 3:
            # end while loop, assume we have found the most optimal solution
            print("Maximum number of failed Armijo line search iterations reached. Exiting...")
            break
    
    elif iters < max_iter_armijo:
        if fail_count > 0:
            # If armijo converged after a fail, reset the counter
            fail_count = 0
            fail_restart_count += 1
            fail_pass = False

        if fail_restart_count < 5:
            # Save the current solution as the last best solution
            u_backup = uk
            v_backup = vk
            p_backup = pk
            q_backup = qk
            c_backup = ck
            it_backup = it
        elif fail_restart_count == 5:
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
            num_steps, dt, out_folder, vertex_to_dof)

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

if fail_count == 3 or fail_restart_count == 5 or (it == max_iter_GD and fail_count > 0):
    print(f"Restoring the solutions from iteration {it_backup}")
    uk = u_backup
    vk = v_backup
    pk = p_backup
    qk = q_backup
    ck = c_backup

eval_sim = 1/T * 1/((a2-a1)**2) * hp.L2_norm_sq_Q(ck, num_steps, dt, M)
misfit_norm_u = hp.L2_norm_sq_Omega(uk[num_steps * nodes:] - uhat_T, M)
misfit_norm_v = hp.L2_norm_sq_Omega(vk[num_steps * nodes:] - vhat_T, M)
control_as_td_vector = true_control * np.ones(vec_length)
true_control_norm = hp.L2_norm_sq_Q(control_as_td_vector, num_steps, dt, M)

u_ctrue, _ = hp.solve_nonlinear_equation(
    control_as_td_vector, uk, None, V, nodes, num_steps, dt, dof_neighbors,
    show_plots=False, vertex_to_dof=vertex_to_dof)

uk.tofile(out_folder + "/AdvSchnak_u.csv", sep = ",")
vk.tofile(out_folder + "/AdvSchnak_v.csv", sep = ",")
ck.tofile(out_folder + "/AdvSchnak_c.csv", sep = ",")
pk.tofile(out_folder + "/AdvSchnak_p.csv", sep = ",")
qk.tofile(out_folder + "/AdvSchnak_q.csv", sep = ",")

# Prepare the data to be written to the CSV
data = {"timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
    "Sim. duration" : round(simulation_duration, 2), "T" : T, "T_data" : T_data, 
    "beta" : beta, "tol" : tol, "GD its" : it, "Armijo its" : armijo_its, 
    "C_ad" : f"[{c_lower}, {c_upper}]", "Mean c. in L^2(Q)^2" : eval_sim, 
    "Misfit norm u" : misfit_norm_u, "Misfit norm v" : misfit_norm_v,
    "J(c_true)" : beta/2*true_control_norm, "out_folder_name" : out_folder}

csv_file_path = "AdvSchnak_FT_simulation_results.csv"
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

import matplotlib.pyplot as plt
mean_ck = []
for i in range(num_steps):
    start = i * nodes
    end = (i+1) * nodes
    mean_ck.append(np.mean(ck[start:end]))
plt.plot(mean_ck)
plt.title("Mean of the control over domain at each time step")
plt.show()

# # ------------ Cost functional over [0,T_PDECO] using c_true, ----------------
# # ---- corresponding u calculated over [0,T_PDECO], and uhat at T_data  ------
# # --------------------- (T_PDECO is T in this script) ------------------------

# control_as_td_vector = true_control * np.ones(vec_length)
# true_control_norm = hp.L2_norm_sq_Q(control_as_td_vector, num_steps, dt, M)

# # Calculate u that corresponds to c_true over [0,T]
# u_ctrue, _ = hp.solve_nonlinear_equation(
#     control_as_td_vector, uk, None, V, nodes, num_steps, dt, dof_neighbors,
#     show_plots=False, vertex_to_dof=vertex_to_dof)

# for bet in [1e-1, 1e-2, 1e-3]:
#     cost_fun_ctrue = hp.cost_functional(u_ctrue, uhat_T, control_as_td_vector, 
#                                         num_steps, dt, M, bet, optim)
#     print(f"Cost functional with c_true over [0,{T}], {T_data=}, {bet=}:", cost_fun_ctrue)


