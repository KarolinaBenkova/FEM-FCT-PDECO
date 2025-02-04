from pathlib import Path
import dolfin as df
import numpy as np
import helpers as hp

# ----------------------------------------------------------------------------
# Script to generate target states for advection PDE-constrained optimization
# -----------------------------------------------------------------=----------


"""
Solves the nonlinear advection-reaction-diffusion equation:
    du/dt + div(-eps * grad(u) + w * u) - u + (1/3) * u^3 = s    in Ω × [0,T]
                      (-eps * grad(u) + w * u) ⋅ n = 0           on ∂Ω × [0,T]
                                              u(0) = u0(x)       in Ω

where w is a velocity/wind vector satisfying:
     div(w) = 0  in Ω × [0,T]
      w ⋅ n = 0  on ∂Ω × [0,T]
Note: This leads to Neumann boundary condition: du/dn = 0 on ∂Ω × [0,T].
"""

# ---------------------------- General Parameters ----------------------------

a1, a2 = 0, 1
dx = 0.025  # Element size
intervals = round((a2 - a1) / dx)

dt = 0.001
T = 2
num_steps = round(T / dt)

show_plots = True  # Toggle for visualization

# ---------------------------- PDE Coefficients ------------------------------

eps = 1e-4  # Diffusion coefficient
speed = 1   # Wind speed

# ----------------------------- Source Function ------------------------------

k, l = 2, 2
source_fun = df.Expression( "sin(k1 * pi * x[0]) * sin(k2 * pi * x[1])",
    degree=4, pi=np.pi, k1=k, k2=l)

# ----------------------------- Wind Field -----------------------------------

wind = df.Expression(("speed * 2 * (x[1] - 0.5) * x[0] * (1 - x[0])",
                     "-speed * 2 * (x[0] - 0.5) * x[1] * (1 - x[1])"),
                     degree=4, speed=speed)

# ---------------------------- Output File Path ------------------------------

output_dir = Path(f"NL_data_eps{eps}_sp{speed}")
output_dir.mkdir(parents=True, exist_ok=True)
output_filename = output_dir / "advection_t_u.csv"

# ----------------------- Initialize Finite Element Mesh ---------------------

mesh = df.RectangleMesh(df.Point(a1, a1), df.Point(a2, a2), intervals, intervals)
V = df.FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
vertex_to_dof = df.vertex_to_dof_map(V)

u = df.TrialFunction(V)
v = df.TestFunction(V)

# Create connectivity between vertices to find neighboring nodes
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
    control_fun=source_fun, show_plots=show_plots, vertextodof=vertex_to_dof)

uk.tofile(output_filename, sep=",")
