from pathlib import Path
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to generate target states for advective Schnakenberg PDE-constrained 
#                                   optimization
# -----------------------------------------------------------------=----------

"""
Solves the advective Schnakenberg system:
 du/dt + div(-Du * grad(u) + ω1 * w * u) + γ(u-u^2v) = γ*a       in Ω × [0,T]
 dv/dt + div(-Dv * grad(v) + ω2 * w * v) + γ(u^2v-b) = 0         in Ω × [0,T]
                      (-Du * grad(u) + ω1 * w * u) ⋅ n = 0       on ∂Ω × [0,T]
                      (-Dv * grad(v) + ω2 * w * v) ⋅ n = 0       on ∂Ω × [0,T]
                                                   u(0) = u0(x)  in Ω
                                                   v(0) = v0(x)  in Ω
                                      
and calculates the L^2(Q)-norm of the parameter "a" (Q = Ω × [0,T]).                      

where w is a velocity/wind vector satisfying:
     div(w) = 0  in Ω × [0,T]
     
Parameters used in Garzon-Alvarado et al (2011):
 Du = 1/100,  Dv = 8.6676, a = 0.1, b = 0.9, γ = 230.82, ω1 = 100, ω2 = 0.6
 w = [-(y-0.5)*sin(2πt),
      (x-0.5)*sin(2πt)]
"""

# ---------------------------- General Parameters ----------------------------

a1, a2 = 0, 1
dx = 0.025 # Element size   ### check: originally used dx=0.02
intervals = round((a2 - a1) / dx)

dt = 0.001
T = 2
num_steps = round(T / dt)

show_plots = True # Toggle for visualization

# ---------------------------- PDE Coefficients ------------------------------

Du, Dv, c_a, c_b, gamma, omega1, omega2, wind = hp.get_schnak_sys_params()
constant_a = df.Constant(c_a)
print("Control parameter is the constant a =", c_a)

# ---------------------------- Output File Path ------------------------------

output_dir = Path(f"AdvSchnak_data_T{T}")
output_dir.mkdir(parents=True, exist_ok=True)
output_filename_u = output_dir / "schnak_u.csv"
output_filename_v = output_dir / "schnak_v.csv"

# ----------------------- Initialize Finite Element Mesh ---------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
w = df.TestFunction(V)

# Create connectivities between vertices to find neighboring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

# ----------------------------- Initial Condition ----------------------------

u0, v0 = hp.schnak_sys_IC(a1, a2, dx, nodes, vertex_to_dof)

vec_length = (num_steps + 1) * nodes  # Include time zero
z = np.zeros(vec_length)
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)
uk[:nodes] = u0
vk[:nodes] = v0

# ------------------------------ Solve PDEs ----------------------------------

uk, vk = hp.solve_schnak_system(
    z, uk, vk, V, nodes, num_steps, dt, dof_neighbors,
    control_fun=constant_a, show_plots=show_plots, vertex_to_dof=vertex_to_dof)

uk.tofile(output_filename_u, sep=",")
vk.tofile(output_filename_v, sep=",")

# ------------------  L^2-norm of true control over Ω × [0,T] ----------------

M = hp.assemble_sparse(u * w * df.dx)
control_as_td_vector = c_a * np.ones(vec_length)
control_norm = hp.L2_norm_sq_Q(control_as_td_vector, num_steps, dt, M)
print(f"L^2-norm of the control over Ω × [0,{T}]:", control_norm)
