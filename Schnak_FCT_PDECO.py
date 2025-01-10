from pathlib import Path

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

# ---------------------------------------------------------------------------
### Flux-corrected transport method for the PDECO problem for the Schnakenberg model
# min_{u,v,a,b} beta1/2*||u-\hat{u}||^2 + beta2/2*||v-\hat{v}||^2 + alpha1/2*||a||^2 + alpha2/2*||b||^2  (norms in L^2)
# subject to:
#   du/dt - Du grad^2 u + om1 w \cdot grad(u) + gamma(u-u^2v-a) = 0       in Ω
#   dv/dt - Dv grad^2 v + om2 w \cdot grad(v) + gamma(u^2v-b)   = 0       in Ω
#                        zero flux BCs       on ∂Ω
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                               div (w) = 0           in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.01*2
intervals_line = round((a2 - a1) / deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = 1
c_lower = -1

Du = 1/100
Dv = 8.6676
c_a = 0.1
c_b = 0.9
gamma = 230.82
omega1 = 100 
omega2 = 0.6

t0 = 0
dt = 0.001*2
T = 3*dt #0.2
T_data = 0.2#T
num_steps = round((T-t0)/dt)
tol = 10**-4 #5*10**-3 # !!!

example_name = f"Schnak_adv_Du{Du}_timedep_vel_coarse_v2/Schnak_adv"
out_folder_name = f"Schnak_adv_FT_Du{Du}_timedep_vel_T{T}_beta{beta}_tol{tol}_ckinit_0.1_cupper{c_upper}pert_coarse_Nt100_v2"
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

np.random.seed(5)

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * w * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M, nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(w)) * dx)

###############################################################################
################ Target states & initial conditions for m,f ###################

###############################################################################
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

# uhat_T = np.zeros(nodes)
# vhat_T = np.zeros(nodes)
uhat_T = np.genfromtxt(example_name + f'_uT{T_data:03}.csv', delimiter=',')
vhat_T = np.genfromtxt(example_name + f'_vT{T_data:03}.csv', delimiter=',')
uhat_T_re = reorder_vector_from_dof_time(uhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))
vhat_T_re = reorder_vector_from_dof_time(vhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time
uk = np.zeros(vec_length)
vk = np.zeros(vec_length)
pk = np.zeros(vec_length)
qk = np.zeros(vec_length)
ck = 0.1*np.ones(vec_length) + 0.01*(np.random.rand(vec_length)-0.5) #np.zeros(vec_length)
dk = np.zeros(vec_length)
uk[:nodes] = u0
vk[:nodes] = v0

sk = 0
###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

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
print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
while stop_crit >= tol and it<1000:
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
    uk[nodes :] = np.zeros(num_steps * nodes)
    vk[nodes :] = np.zeros(num_steps * nodes)
    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1}), vk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        wind.t = t

        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        u_n = uk[start - nodes : start]
        v_n = vk[start - nodes : start]
        c_np1 = ck[start : end]
        
        # Define previous time-step solution as a function
        u_n_fun = vec_to_function(u_n, V)
        v_n_fun = vec_to_function(v_n, V)
        c_np1_fun = vec_to_function(c_np1, V)
        
        # # Solve for u using FCT (advection-dominated equation)
        # A = assemble_sparse(dot(wind, grad(u)) * w * dx)
        # mat_u = -(Du*Ad + omega1*A)
        # rhs_u = np.asarray(assemble((gamma*(c_np1_fun + u_n_fun**2 * v_n_fun))* w * dx))
        # uk[start : end] = FCT_alg(mat_u, rhs_u, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

        # u_np1_fun = vec_to_function(uk[start : end], V)
        # M_u2 = assemble_sparse(u_np1_fun * u_np1_fun * u * w *dx)
        
        # # Solve for v using a direct solver
        # rhs_v = np.asarray(assemble((gamma*c_b)* w * dx))
        # vk[start : end] = spsolve(M + dt*(Dv*Ad + omega2*A + gamma*M_u2), M@v_n + dt*rhs_v) 
        
        # v2: different weak formulation for advection matrix (corresponding signs also change)
        # Solve for u using FCT (advection-dominated equation)
        A = assemble_sparse(dot(wind, grad(w)) * u * dx)
        mat_u = -(Du*Ad - omega1*A)
        rhs_u = np.asarray(assemble((gamma*(c_np1_fun + u_n_fun**2 * v_n_fun))* w * dx))
        uk[start : end] = FCT_alg(mat_u, rhs_u, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

        u_np1_fun = vec_to_function(uk[start : end], V)
        M_u2 = assemble_sparse(u_np1_fun * u_np1_fun * u * w *dx)
        
        # Solve for v using a direct solver
        rhs_v = np.asarray(assemble((gamma*c_b)* w * dx))
        vk[start : end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@v_n + dt*rhs_v) 
            
    ###########################################################################
    ############## Solve the adjoint equations using FCT for p ################
    ###########################################################################
    
    qk = np.zeros(vec_length) 
    pk = np.zeros(vec_length)
    # insert final-time condition
    qk[num_steps * nodes :] = vhat_T - vk[num_steps * nodes :]
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    t = T
    print('Solving adjoint equations...')
    for i in reversed(range(0, num_steps)): # solve for pk(t_n), qk(t_n)
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        q_np1 = qk[end : end + nodes] # qk(t_{n+1})
        p_np1 = pk[end : end + nodes] # pk(t_{n+1})

        p_np1_fun = vec_to_function(p_np1, V) 
        q_np1_fun = vec_to_function(q_np1, V)
        u_n_fun = vec_to_function(uk[start : end], V)      # uk(t_n)
        v_n_fun = vec_to_function(vk[start : end], V)      # vk(t_n)
        c_n_fun = vec_to_function(ck[start : end], V)      # ck(t_n)
        
        wind.t = t
        wind_fun = project(wind, W)
        
        ## Approach 1: first solve for p, then for q
        ### needs to be fixed
        # mat_p = -Du*Ad + omega1*A
        # rhs_p = np.asarray(assemble(2 * gamma * u_n_fun * v_n_fun * (p_np1_fun - q_np1_fun) * w * dx))
        # pk[start:end] = FCT_alg(mat_p, rhs_p, p_np1, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)

        # p_np1_fun = vec_to_function(pk[end : end + nodes], V)
        # rhs_q = np.asarray(assemble(gamma * p_np1_fun * u_n_fun**2 * w * dx))
        
        # qk[start:end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@q_np1 + dt*rhs_q) 
        
        # Approach 2: first solve for q, then for p
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
       
        # ## Approach 2 v2: first solve for q, then for p & different weak formulation for advection matrix (signs don't change)
        # A = assemble_sparse(dot(wind, grad(u)) * w * dx)
        # M_u2 = assemble_sparse(u_n_fun * u_n_fun * u * w *dx)
        # rhs_q = np.asarray(assemble(gamma * p_np1_fun * u_n_fun**2 * w * dx))
        # qk[start:end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@q_np1 + dt*rhs_q) 
        
        # q_n_fun = vec_to_function(qk[start:end], V)
        
        # mat_p = -Du*Ad + omega1*A
        # M_uv = assemble_sparse(u_n_fun * v_n_fun * u * w *dx)
        # rhs_p = np.asarray(assemble(- 2 * gamma * u_n_fun * v_n_fun * q_n_fun * w * dx))
        
        # pk[start:end] = FCT_alg(mat_p, rhs_p, p_np1, dt, nodes, M, M_Lump, 
        #                         dof_neighbors, source_mat=gamma*M - 2*gamma*M_uv)
           
    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta * ck - gamma * pk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    print(f'{cost_fun_k=}')
    
    print('Starting Armijo line search...')
    sk, u_inc, v_inc = armijo_line_search(uk, ck, dk, uhat_T, num_steps, dt, M, 
                          c_lower, c_upper, beta, cost_fun_k, nodes,  V = V,
                          optim = 'finaltime', dof_neighbors= dof_neighbors,
                          example = 'Schnak', var2 = vk, var2_target = vhat_T)
    
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    cost_fun_kp1 = cost_functional(u_inc, uhat_T, ckp1, num_steps, dt, M, beta,
                           optim='finaltime', var2 = v_inc, var2_target=vhat_T)
    print(f'{cost_fun_kp1=}')

    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals_u.append(L2_norm_sq_Omega(u_inc[num_steps*nodes:] - uhat_T, M))
    cost_fidelity_vals_v.append(L2_norm_sq_Omega(v_inc[num_steps*nodes:] - vhat_T, M))
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
    
    uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
    vk_re = reorder_vector_from_dof_time(vk, num_steps + 1, nodes, vertextodof)
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
        
        u_re = uk_re[startU : endU]
        v_re = vk_re[startU : endU]
        c_re = ck_re[startP : endP]
        p_re = pk_re[startP : endP]
        q_re = qk_re[startP : endP]
            
        u_re = u_re.reshape((sqnodes,sqnodes))
        v_re = v_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))
        q_re = q_re.reshape((sqnodes,sqnodes))
        
        if show_plots is True and (i%5 == 0 or i == num_steps-1):
            
            fig = plt.figure(figsize=(20, 10))

            ax = fig.add_subplot(2, 4, 1)
            im1 = ax.imshow(uhat_T_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $u$ at t = {T}')
        
            ax = fig.add_subplot(2, 4, 2)
            im2 = ax.imshow(u_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f'Computed state $u$ at t = {round(tU, 5)}')
        
            ax = fig.add_subplot(2, 4, 3)
            im3 = ax.imshow(p_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f'Computed adjoint $p$ at t = {round(tP, 5)}')
        
            ax = fig.add_subplot(2, 4, 4)
            im4 = ax.imshow(c_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f'Computed control $c$ at t = {round(tP, 5)}')
            
            ax = fig.add_subplot(2, 4, 5)
            im5 = ax.imshow(vhat_T_re)
            cb5 = fig.colorbar(im1, ax=ax)
            ax.set_title(f'{it=}, Desired state for $v$ at t = {T}')
        
            ax = fig.add_subplot(2, 4, 6)
            im6 = ax.imshow(v_re)
            cb6 = fig.colorbar(im2, ax=ax)
            ax.set_title(f'Computed state $v$ at t = {round(tU, 5)}')
        
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
        im2_u = plt.plot(np.arange(1, it + 1), cost_fidelity_vals_u, label='for u')
        im2_v = plt.plot(np.arange(1, it + 1), cost_fidelity_vals_v, label='for v')
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
# # Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
vk_re = reorder_vector_from_dof_time(vk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
qk_re = reorder_vector_from_dof_time(qk, num_steps + 1, nodes, vertextodof)

uk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_u.csv', sep = ',')
vk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_v.csv', sep = ',')
ck.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_c.csv', sep = ',')
pk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_p.csv', sep = ',')
qk.tofile(out_folder_name + f'/Schnak_adv_T{T}_beta{beta}_q.csv', sep = ',')


print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('Solutions saved to the folder:', out_folder_name)
print('L2_\Omega^2 (u(T) - uhat_T)=', L2_norm_sq_Omega(uk[num_steps*nodes]-uhat_T,M))
print('L2_\Omega^2 (v(T) - vhat_T)=', L2_norm_sq_Omega(vk[num_steps*nodes]-vhat_T,M))

print('Max. of the control at final time step:', np.amax(ck[num_steps*nodes:]))
print('Min. of the control at final time step:', np.amin(ck[num_steps*nodes:]))
print('Mean of the control at final time step:', np.mean(ck[num_steps*nodes:]))

means_ck = []
for i in range(num_steps):
    start = i * nodes
    end = (i+1) * nodes
    means_ck.append(np.mean(ck[start:end]))
plt.plot(means_ck)
plt.title('Mean of c at each time step, final iteration')
plt.show()