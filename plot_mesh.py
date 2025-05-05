from dolfin import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(5, 5)

fig = plt.figure(figsize=(5, 5))
plot(mesh, linewidth=1.5)

plt.gca().set_aspect('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig("mesh_plot.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()

V = FunctionSpace(mesh, "CG", 1)
nodes = V.dim()
num_elements = mesh.num_cells()
