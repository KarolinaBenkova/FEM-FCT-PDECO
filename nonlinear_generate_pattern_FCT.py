from pathlib import Path
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to generate target states for advection PDE-constrained optimization
# -----------------------------------------------------------------=----------


"""
Solves the nonlinear advection-reaction-diffusion equation:
    du/dt + div(-ε * grad(u) + w * u) - u + (1/3) * u^3 = s    in Ω × [0,T]
                      (-ε * grad(u) + w * u) ⋅ n = 0           on ∂Ω × [0,T]
                                              u(0) = u0(x)       in Ω
and calculates the L^2(Q)-norm of the source function (Q = Ω × [0,T]).                      

where w is a velocity/wind vector satisfying:
     div(w) = 0  in Ω × [0,T]
      w ⋅ n = 0  on ∂Ω × [0,T]
Note: This leads to Neumann boundary condition: du/dn = 0 on ∂Ω × [0,T].
Parameters considered:
 Diffusion parameters: ε = 1e-4
 Wind field involves a scaling coefficient (wind speed)
  w = [2 * speed * (y-0.5) * x * (1 - x),  
       -2 * speed * (x-0.5) * y * (1 - y)]
"""

# ---------------------------- General Parameters ----------------------------

a1, a2 = 0, 1
dx = 0.025 # Element size
intervals = round((a2 - a1) / dx)

dt = 0.001
T = 2
num_steps = round(T / dt)

show_plots = True # Toggle for visualization

# ---------------------------- PDE Coefficients ------------------------------

eps, speed, wind = hp.get_nonlinear_eqns_params()
print(f"Diffusion parameter: {eps=} \n Wind speed: {speed=}")

# ----------------------------- Source Function ------------------------------

k, l = 2, 2
source_fun = df.Expression( "sin(k1 * pi * x[0]) * sin(k2 * pi * x[1])",
    degree=4, pi=np.pi, k1=k, k2=l)

# ---------------------------- Output File Path ------------------------------

output_dir = Path(f"NL_data_eps{eps}_sp{speed}_T{T}")
output_dir.mkdir(parents=True, exist_ok=True)
output_filename = output_dir / "advection.csv"

# ----------------------- Initialize Finite Element Mesh ---------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
v = df.TestFunction(V)

# Create connectivities between vertices to find neighboring nodes
mesh.init(0, 1)
dof_neighbors = hp.find_node_neighbours(mesh, nodes, vertex_to_dof)

# ----------------------------- Initial Condition ----------------------------

u0 = hp.nonlinear_equation_IC(a1, a2, dx, nodes, vertex_to_dof)

vec_length = (num_steps + 1) * nodes  # Include time zero
z = np.zeros(vec_length)
uk = np.zeros(vec_length)
uk[:nodes] = u0

# ------------------------------- Solve PDE ----------------------------------

uk, _ = hp.solve_nonlinear_equation(
    z, uk, None, V, nodes, num_steps, dt, dof_neighbors,
    control_fun=source_fun, show_plots=show_plots, vertex_to_dof=vertex_to_dof)

uk.tofile(output_filename, sep=",")

# ------------------  L^2-norm of true control over Ω × [0,T] ----------------

M = hp.assemble_sparse(u * v * df.dx)
source_vector = df.interpolate(source_fun, V).vector().get_local()
source_vector_td = np.tile(source_vector, num_steps + 1)
control_norm = hp.L2_norm_sq_Q(source_vector_td, num_steps, dt, M)
print(f"L^2-norm of the control over Ω × [0,{T}]:", control_norm)
