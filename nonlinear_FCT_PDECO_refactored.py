from pathlib import Path
from datetime import datetime
import os
import csv
import time
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to solve the PDECO problem with nonlinear advection equation
# -----------------------------------------------------------------=----------

"""
Solves the PDECO problem below with projected gradient descent method and FCT
Cost functional:
J(u,c) = 1/2*||u(T) - û_T||^2 + beta/2*||c||^2
(L^2-norms over Ω and Ω × [0,T])

min_{u,c} J(u,c)
subject to:
  du/dt + div(-eps * grad(u) + w * u) - u + (1/3) * u^3 = c    in Ω × [0,T]
                    (-eps * grad(u) + w * u) ⋅ n = 0           on ∂Ω × [0,T]
                                            u(0) = u0(x)       in Ω
                                            c in [ca,cb]
where w is a velocity/wind vector satisfying:
     div(w) = 0  in Ω × [0,T]
      w ⋅ n = 0  on ∂Ω × [0,T]

Additional optimality conditions:
- Adjoint equation, BC and final-time condition
  -dp/dt + div(-eps * grad(p) + w * p) +u^2 * p - p = 0      in Ωx[0,T]
                                   dp/dn = 0                 on ∂Ωx[0,T]
                                    p(T) = û_T - u(T)  in Ω
- Gradient equation:           β * c - p = 0
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

# ---------------------------- PDECO parameters ------------------------------

beta = 1e-1 # Regularization parameter

# Box constraints for c
c_lower = -1
c_upper = 1

eps, speed, _ = hp.get_nonlinear_eqns_params()

# --------------------- Gradient descent parameters --------------------------

tol = 1e-4
max_iter_armijo = 5
max_iter_GD = 50

# ----------------------- Input & Output file paths --------------------------

target_data_path = "NL_data_eps0.0001_sp1_T2"
target_data_file_name = "advection"
target_file = os.path.join(target_data_path, f"{target_data_file_name}_T{T_data}.csv")

out_folder = f"ref_NL_FT_T{T}_Tdata{T_data}beta{beta}_Ca{c_lower}_Cb{c_upper}_tol{tol}"
if not Path(out_folder).exists():
    Path(out_folder).mkdir(parents=True)
   
print(f"dx={dx}, {dt=}, {T=}, {beta=}, {c_lower=}, {c_upper=}")
print(f"{eps=}, {speed=}")
print(f"{tol=}, {max_iter_GD=}, {max_iter_armijo=}")

# ------------------------------ Initialization ------------------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
sqnodes = round(np.sqrt(nodes))
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
v = df.TestFunction(V)

M = hp.assemble_sparse(u * v * df.dx) # Mass matrix

# Explicitly create connectivities between vertices to find neighbouring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

u0 = hp.nonlinear_equation_IC(a1, a2, dx, nodes, vertex_to_dof)

if not os.path.exists(target_file):
    hp.extract_data(
        target_data_path, target_data_file_name, T_data, dt, nodes, vertex_to_dof)
uhat_T_re, uhat_T = hp.import_data_final(target_file, nodes, vertex_to_dof)

# ----------------- Initialize gradient descent variables --------------------

vec_length = (num_steps + 1) * nodes # Include zero and final time

# Initial guess for the control
ck = np.zeros(vec_length)

# Solve the state equation for the corresponding state
uk = np.zeros(vec_length)
uk[:nodes] = u0
uk, _ = hp.solve_nonlinear_equation(
    ck, uk, None, V, nodes, num_steps, dt, dof_neighbors)

# Solve the adjoint equation
pk = np.zeros(vec_length)
pk = hp.solve_adjoint_nonlinear_equation(
    uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors)

# Calculate initial cost functional
cost_fun_old = hp.cost_functional(
    uk, uhat_T, ck, num_steps, dt, M, beta, optim="finaltime")
cost_fun_new = (2 + tol) * cost_fun_old
stop_crit = hp.rel_err(cost_fun_new, cost_fun_old)

dk = np.zeros(vec_length)

it = 0
fail_count = 0
fail_restart_count = 0
fail_pass = False
cost_fun_vals, cost_fidel_vals, cost_c_vals, armijo_its = ([] for _ in range(4))
cost_fun_vals.append(cost_fun_old)

start_time = time.time()

##############################################################################
# ------------------------ PROJECTED GRADIENT DESCENT ------------------------
##############################################################################

while (stop_crit >= tol or fail_pass) and it < max_iter_GD:
    print(f"\nIteration: {it}")

    ## 1. choose the descent direction
    dk = -(beta * ck - pk)

    ## 2. Find optimal stepsize with Armijo line search and calculate uk, ck
    print("Starting Armijo line search...")
    uk, ck, iters = hp.armijo_line_search_ref(
        uk, ck, dk, uhat_T, num_steps, dt, c_lower, c_upper, beta, cost_fun_old,
        nodes, "finaltime", V, dof_neighbors=dof_neighbors,
        nonlinear_solver=hp.solve_nonlinear_equation, max_iter=max_iter_armijo)
    
    ## 3. Solve the adjoint equation using new uk
    pk = hp.solve_adjoint_nonlinear_equation(
        uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors)

    if iters == max_iter_armijo:
        fail_count += 1
        fail_pass = True
        if fail_count == 3:
            # end while loop, assume we have found the most optimal solution
            print("Maximum number of failed Armijo line search iterations reached. Exiting...")
            break
    
    elif iters < max_iter_armijo:
        if fail_count > 0:
            # If armijo converged after a fail, reset the counter
            fail_count = 0
            fail_restart_count += 1
        # Save the current solution as the last best solution
        u_backup = uk
        p_backup = pk
        c_backup = ck
        it_backup = it

        if fail_restart_count == 5:
            # End while loop, assume we have found the most optimal solution
            print("Maximum number of restarts reached. Exiting...")
            break

    ## 4. Calculate metrics
    cost_fun_new = hp.cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta,
                           optim="finaltime")
    stop_crit = hp.rel_err(cost_fun_new, cost_fun_old)
    eval_sim = 1/T * 1/((a2-a1)**2) * hp.L2_norm_sq_Q(ck, num_steps, dt, M)

    print(f"{cost_fun_new=}")
    print(f"{eval_sim=}")

    cost_fun_vals.append(cost_fun_new)
    cost_fidel_vals.append(hp.L2_norm_sq_Omega(uk[num_steps*nodes:] - uhat_T, M))
    cost_c_vals.append(hp.L2_norm_sq_Q(ck, num_steps, dt, M))
    armijo_its.append(iters)

    if produce_plots is True:
        hp.plot_nonlinear_solution(uk, pk, ck, uhat_T_re, T_data, it,
                                nodes, num_steps, dt, out_folder, vertex_to_dof)

    hp.plot_progress(cost_fun_vals, cost_fidel_vals, cost_c_vals, it, out_folder)

    ## Make updates
    it += 1
    cost_fun_old = cost_fun_new

    print(f"Stopping criterion: {stop_crit}")

# --------------------------- Save results -----------------------------------

# Record the end time of the simulation
end_time = time.time()
simulation_duration = end_time - start_time

if fail_count == 3  or fail_restart_count == 5 or (it == max_iter_GD and fail_count > 0):
    print(f'Restoring the solutions from iteration {it_backup}')
    uk = u_backup
    pk = p_backup
    ck = c_backup

eval_sim = 1/T * 1/((a2-a1)**2) * hp.L2_norm_sq_Q(ck, num_steps, dt, M)

misfit_norm = hp.L2_norm_sq_Omega(uk[num_steps * nodes:] - uhat_T, M)
true_control_norm = hp.norm_true_control("nonlinear", T_data, dt, M, V)

uk.tofile(out_folder + "/NL_u.csv", sep = ",")
ck.tofile(out_folder + "/NL_c.csv", sep = ",")
pk.tofile(out_folder + "/NL_p.csv", sep = ",")

# Prepare the data to be written to the CSV
data = {"timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
    "Sim. duration" : round(simulation_duration, 2), "T" : T, "T_data" : T_data, 
    "beta" : beta, "tol" : tol, "GD its" : it, "Armijo its" : armijo_its, 
    "C_ad" : f"[{c_lower}, {c_upper}]", "Mean c. in L^2(Q)^2" : eval_sim, 
    "Misfit norm" : misfit_norm, "J(c_true)" : true_control_norm/beta,
    "out_folder_name" : out_folder}

csv_file_path = "NL_FT_simulation_results.csv"
file_exists = os.path.isfile(csv_file_path)

# Write the data to the CSV file
with open(csv_file_path, mode="a", newline="") as csv_file:
    fieldnames = ["timestamp", "Sim. duration", "T", "T_data", "beta", "tol", 
                  "GD its",  "Armijo its", "C_ad", "Mean c. in L^2(Q)^2",  
                  "Misfit norm", "J(c_true)", "out_folder_name"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header only if the file does not exist
    if not file_exists:
        writer.writeheader()

    writer.writerow(data)

print(f"\nExit:\nFinal stopping criterion: {stop_crit} \nIterations: {it}")
print("Armijo iterations:", armijo_its)
print("Solutions saved to:", out_folder)
print("||u(T) - û_T|| in L^2(Ω)^2 :", misfit_norm)
print("Average control in L^2(Q)^2:", eval_sim)
print(f"Final cost functional value for Ω × [0,{T}]:", cost_fun_new)
print(f"1/β *||c_true|| in L^2-norm^2 over Ω × [0,{T_data}]:", true_control_norm/beta)
