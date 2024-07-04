from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from scipy.sparse.linalg import spsolve, gmres, minres, LinearOperator
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *
import mimura_data_helpers
import data_helpers

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
a2 = 16
deltax = 1/8
intervals_line = round((a2 - a1) / deltax)
beta = 1
# box constraints for c, exact solution is in [0,1]
c_upper = 1.5
c_lower = 0

delta = 32
Dm = 0.0625
Df = 1
chi = 8.5

t0 = 0
dt = 0.1
T = 14
num_steps = round((T-t0)/dt)
tol = 10**-5 # !!!
example_name = 'mimura_tsujikawa'

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
v = TestFunction(V)

show_plots = True

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(assemble(u * v * dx))

# Row-lumped mass matrix
M_Lump = row_lump(M, nodes)

# Stiffness matrix
Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))

# System matrix: equation for f and q
Mat_fq = M + dt * (Df * Ad + delta * M)

###############################################################################
################ Target states & initial conditions for m,f ###################
###############################################################################

m0_orig = mimura_data_helpers.m_initial_condition(a1, a2, deltax).reshape(nodes)
f0_orig = 1/32 * np.ones(nodes)
m0 = reorder_vector_to_dof_time(m0_orig, 1, nodes, vertextodof)
f0 = reorder_vector_to_dof_time(f0_orig, 1, nodes, vertextodof)

mhat_T_orig = data_helpers.get_data_array('m', example_name, T)
fhat_T_orig = data_helpers.get_data_array('f', example_name, T)
mhat_T = reorder_vector_to_dof_time(mhat_T_orig, 1, nodes, vertextodof)
fhat_T = reorder_vector_to_dof_time(fhat_T_orig, 1, nodes, vertextodof)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1) * nodes # include zero and final time
zeros_nt = np.zeros(vec_length)

ck = np.zeros(vec_length)

mk = np.zeros(vec_length)
fk = np.zeros(vec_length)
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
ck = np.zeros(vec_length)
dk = np.zeros(vec_length)

mk[:nodes] = m0
fk[:nodes] = f0

sk = 0
###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional_proj_chtxs(mk, fk, ck, 
   dk, sk, mhat_T, fhat_T, num_steps, dt, nodes, M, c_lower, c_upper, beta)

stop_crit = 5
stop_crit2 = 5 
print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
# while ((stop_crit >= tol ) or (stop_crit2 >= tol)) and it<1000:
while (stop_crit2 >= tol) and it<1000:

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
        
        f_rhs = rhs_chtx_f(f_n_fun, m_n_fun, c_np1_fun, dt, v)

        fk[start : end] = spsolve(Mat_fq, f_rhs)
        
        f_np1_fun = vec_to_function(fk[start : end], V)

        A_m = mimura_data_helpers.mat_chtx_m(f_np1_fun, m_n_fun, Dm, chi, u, v)
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

        A_p = mimura_data_helpers.mat_chtx_p(f_n_fun, m_n_fun, Dm, chi, u, v)
        p_rhs = rhs_chtx_p(c_n_fun, q_n_fun, v)
        
        pk[start:end] = FCT_alg(A_p, p_rhs, p_np1, dt, nodes, M, M_Lump, dof_neighbors)
        
    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta * ck - qk * mk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    
    print('Starting Armijo line search...')
    sk = armijo_line_search_chtxs(mk, fk, qk, ck, dk, mhat_T, fhat_T, Mat_fq, 
                                  chi, Dm, Df, num_steps, dt, nodes, M, M_Lump, 
                                  Ad, c_lower, c_upper, beta, V, dof_neighbors)
    
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk * dk, c_lower, c_upper)
    
    stop_crit = L2_norm_sq_Q(ckp1 - ck, num_steps, dt, M) \
        / L2_norm_sq_Q(ck, num_steps, dt, M)

    # Check the cost functional - stopping criterion
    cost_fun_kp1 = cost_functional_proj_chtxs(mk, fk, ckp1, dk, sk, 
              mhat_T, fhat_T, num_steps, dt, nodes, M, c_lower, c_upper, beta)
    stop_crit2 = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)

    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit=}')
    print(f'{stop_crit2=}')
    
###############################################################################    
# Mapping to order the solution vectors based on vertex indices
mk_re = reorder_vector_from_dof_time(mk, num_steps + 1, nodes, vertextodof)
fk_re = reorder_vector_from_dof_time(fk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
qk_re = reorder_vector_from_dof_time(qk, num_steps + 1, nodes, vertextodof)

min_m = np.amin(mk)
min_f = np.amin(fk)
min_c = np.amin(ck)
min_p = np.amin(pk)
min_q = np.amin(qk)


max_m = np.amax(mk)
max_f = np.amax(fk)
max_c = np.amax(ck)
max_p = np.amax(pk)
max_q = np.amax(qk)


for i in range(num_steps):
    start_adj = i * nodes
    end_adj = (i+1) * nodes
    t_adj = i * dt
    
    start_st = (i+1) * nodes
    end_st = (i+2) * nodes
    t_st = (i+1) * dt
    
    m_re = mk_re[start_st : end_st]
    f_re = fk_re[start_st : end_st]
    c_re = ck_re[start_adj : end_adj]
    p_re = pk_re[start_adj : end_adj]
    q_re = qk_re[start_adj : end_adj]

        
    m_re = m_re.reshape((sqnodes, sqnodes))
    f_re = f_re.reshape((sqnodes, sqnodes))
    c_re = c_re.reshape((sqnodes, sqnodes))
    p_re = p_re.reshape((sqnodes, sqnodes))
    q_re = q_re.reshape((sqnodes, sqnodes))
    
    if show_plots is True and i % 20 == 0:
        fig2 = plt.figure(figsize = (15, 10))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(2,3,1)
        im1 = plt.imshow(m_re, vmin = min_m, vmax = max_m)
        fig2.colorbar(im1)
        plt.title(f'Computed state $m$ at t = {round(t_st,5)}')
        ax2 = plt.subplot(2,3,2)
        im2 = plt.imshow(f_re, vmin = min_f, vmax = max_f)
        fig2.colorbar(im2)
        plt.title(f'Computed state $f$ at t = {round(t_st,5)}')
        ax2 = plt.subplot(2,3,3)
        im1 = plt.imshow(c_re, vmin = min_c, vmax = max_c)
        fig2.colorbar(im1)
        plt.title(f'Computed control $c$ at t = {round(t_adj,5)}')
        ax2 = plt.subplot(2,3,4)
        im2 = plt.imshow(p_re, vmin = min_p, vmax = max_p)
        fig2.colorbar(im2)
        plt.title(f'Computed adjoint $p$ at t = {round(t_adj,5)}')
        ax2 = plt.subplot(2,3,5)
        im1 = plt.imshow(q_re, vmin = min_q, vmax = max_q)
        fig2.colorbar(im1)
        plt.title(f'Computed adjoint $q$ at t = {round(t_adj,5)}')
        plt.show()
        
    print('------------------------------------------------------')

print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx = {deltax}')   
print(f'{dt=}, {T=}, {beta=}')
