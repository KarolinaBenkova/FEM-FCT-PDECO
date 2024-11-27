from pathlib import Path

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

# ---------------------------------------------------------------------------
### Flux-corrected transport method for the PDECO problem for the Schnakenberg model
# min_{u,v,a,b} beta1/2*||u-\hat{u}||^2 + beta2/2*||v-\hat{v}||^2 + alpha1/2*||a||^2 + alpha2/2*||b||^2  (norms in L^2)
# subject to:
#   -Du grad^2 u + om1 w \cdot grad(u) + gamma(u-u^2v-a) = 0       in Ω
#   -Dv grad^2 v + om2 w \cdot grad(v) + gamma(u^2v-b)   = 0       in Ω
#                        zero flux BCs       on ∂Ω
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                               div (w) = 0           in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1 #0.01
intervals_line = round((a2 - a1) / deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = 0.5
c_lower = 0

Du = 1/100
Dv = 8.6676
c_a = 0.1
c_b = 0.9
gamma = 230.82
omega1 = 100 #0.6
omega2 = 0.6

t0 = 0
dt = 0.001
T = 0.2
T_data = 0.2
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!

example_name = f"Schnak_adv_Du{Du}_timedep_vel/Schnak_adv"
out_folder_name = f"Schnak_adv_Du{Du}_timedep_vel_T{T}_beta{beta}_tol{tol}_ckinit_ones"
if not Path(out_folder_name).exists():
    Path(out_folder_name).mkdir(parents=True)

wind = Expression(('-(x[1]-0.5)*sin(2*pi*t)','(x[0]-0.5)*sin(2*pi*t)'), degree=4, pi = np.pi, t = 0)
    
show_plots = True

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

X = np.arange(a1, a2 + deltax, deltax)
Y = np.arange(a1, a2 + deltax, deltax)
X, Y = np.meshgrid(X,Y)

u = TrialFunction(V)
w = TestFunction(V)

W = VectorFunctionSpace(mesh, "CG", 1)
wind_fun = Function(W)
wind_fun = project(wind, W)

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

M = assemble_sparse_lil(u * w * dx)
M_Lump = row_lump(M, nodes)
Ad = assemble_sparse(dot(grad(u), grad(w)) * dx)

def init_conditions(X,Y):
    '''
    Function for the initial conditions.
    Input = mesh grid (X,Y) = square 2D arrays with the same dimensions).
    '''
    con = 0.1
    u_init = c_a + c_b + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))

    v_init = c_b / pow(c_a + c_b, 2) + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))                  
    return u_init, v_init

u0_orig, v0_orig = init_conditions(X, Y)

u0 = reorder_vector_to_dof_time(u0_orig.reshape(nodes), 1, nodes, vertextodof)
v0 = reorder_vector_to_dof_time(v0_orig.reshape(nodes), 1, nodes, vertextodof)


uhat_T = np.genfromtxt(example_name + f'_uT{T_data:03}.csv', delimiter=',')
vhat_T = np.genfromtxt(example_name + f'_vT{T_data:03}.csv', delimiter=',')
uhat_T_re = reorder_vector_from_dof_time(uhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))
vhat_T_re = reorder_vector_from_dof_time(vhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))

vec_length = (num_steps + 1)*nodes 
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
ck = np.ones(vec_length) 
dk = np.zeros(vec_length)
uk[:nodes] = u0
vk[:nodes] = v0

sk = 0
it = 0
cost_fun_k = 10*cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, 
                                var2 = vk, var2_target = vhat_T,
                                optim='finaltime')
cost_fun_vals = []
cost_fidelity_vals_u = []
cost_fidelity_vals_v = []
cost_control_vals = []
mean_control_vals = []
stop_crit = 5
while stop_crit >= tol and it<1000:
    it += 1
    
    t = 0
    uk[nodes :] = np.zeros(num_steps * nodes)
    vk[nodes :] = np.zeros(num_steps * nodes)
    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1}), vk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        wind.t = t
        A = assemble_sparse(dot(wind, grad(u)) * w * dx)

        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        u_n = uk[start - nodes : start]
        v_n = vk[start - nodes : start]
        c_np1 = ck[start : end]
        
        u_n_fun = vec_to_function(u_n, V)
        v_n_fun = vec_to_function(v_n, V)
        c_np1_fun = vec_to_function(c_np1, V)
        
        mat_u = -(Du*Ad + omega1*A)
        rhs_u = np.asarray(assemble((gamma*(c_np1_fun + u_n_fun**2 * v_n_fun))* w * dx))
        uk[start : end] = FCT_alg(mat_u, rhs_u, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

        u_np1_fun = vec_to_function(uk[start : end], V)
        M_u2 = assemble_sparse(u_np1_fun * u_np1_fun * u * w *dx)
        
        # Solve for v using a direct solver
        rhs_v = np.asarray(assemble((gamma*c_b)* w * dx))
        vk[start : end] = spsolve(M + dt*(Dv*Ad + omega2*A + gamma*M_u2), M@v_n + dt*rhs_v) 
            
    qk = np.zeros(vec_length) 
    pk = np.zeros(vec_length)
    qk[num_steps * nodes :] = vhat_T - vk[num_steps * nodes :]
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    t = T
    for i in reversed(range(0, num_steps)): # solve for pk(t_n), qk(t_n)
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        q_np1 = qk[end : end + nodes] # qk(t_{n+1})
        p_np1 = pk[end : end + nodes] # pk(t_{n+1})

        p_np1_fun = vec_to_function(p_np1, V) 
        q_np1_fun = vec_to_function(q_np1, V)
        u_n_fun = vec_to_function(uk[start : end], V)      # uk(t_n)
        v_n_fun = vec_to_function(vk[start : end], V)      # vk(t_n)
        c_n_fun = vec_to_function(ck[start : end], V)      # ck(t_n)
        
        wind.t = t
        wind_fun = project(wind, W)
        A = assemble_sparse(div(wind_fun*u) * w * dx)
        
        
        M_u2 = assemble_sparse(u_n_fun * u_n_fun * u * w *dx)
        rhs_q = np.asarray(assemble(gamma * p_np1_fun * u_n_fun**2 * w * dx))
        qk[start:end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@q_np1 + dt*rhs_q) 
        
        q_n_fun = vec_to_function(qk[start:end], V)
        
        mat_p = -Du*Ad + omega1*A
        M_uv = assemble_sparse(u_n_fun * v_n_fun * u * w *dx)
        rhs_p = np.asarray(assemble(- 2 * gamma * u_n_fun * v_n_fun * q_n_fun * w * dx))
        
        pk[start:end] = FCT_alg(mat_p, rhs_p, p_np1, dt, nodes, M, M_Lump, 
                                dof_neighbors, source_mat=gamma*M - 2*gamma*M_uv)

        p_np1_fun = vec_to_function(pk[end : end + nodes], V)
           
    
    dk = -(beta * ck - gamma * qk)
    
    sk, u_inc, v_inc = armijo_line_search(uk, ck, dk, uhat_T, num_steps, dt, M, 
                          c_lower, c_upper, beta, cost_fun_k, nodes,  V = V,
                          optim = 'finaltime', dof_neighbors= dof_neighbors,
                          example = 'Schnak', var2 = vk, var2_target = vhat_T)
    
    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    cost_fun_kp1 = cost_functional(u_inc, uhat_T, ckp1, num_steps, dt, M, beta,
                           optim='finaltime', var2 = v_inc, var2_target=vhat_T)
    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals_u.append(L2_norm_sq_Omega(u_inc[num_steps*nodes:] - uhat_T, M))
    cost_fidelity_vals_v.append(L2_norm_sq_Omega(v_inc[num_steps*nodes:] - vhat_T, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    max_ck = []
    for i in range(num_steps):
        start = i * nodes
        end = (i+1) * nodes
        max_ck.append(np.amax(ck[start:end]))
    np.mean(max_ck)
    mean_control_vals.append(np.mean(max_ck))
        
    stop_crit = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
print(np.amin(ck[num_steps*nodes:]))
