from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *
import data_helpers

# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the advection-diffusion equation
### with Flux-corrected transport method
# min_{u,v,a,b} 1/2*||u(T)-\hat{u}_T||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt - eps*grad^2(u) + div( u (omega*w + c*m) ) = 0   in Ωx[0,T]
#                           dot(grad u, n) = 0            on ∂Ωx[0,T]
#                                    du/dn = 0            on ∂Ωx[0,T]
#                                     u(0) = u0(x)        in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]

# b = drift vector, e.g. (1,1)
# c = control variable, velocity of the drift

# Optimality conditions:
#  du/dt - eps*grad^2(u) + w \dot grad(u)) = 0                     in Ωx[0,T]
#    -dp/dt - eps*grad^2 p - w \dot grad(p)= 0                     in Ωx[0,T]
#                            dp/dn = du/dn = 0                     on ∂Ωx[0,T]
#                                     u(0) = u0(x)                 in Ω
#                                     p(T) = hat{u}_T - u(T)       in Ω
# gradient equation:      beta*c - u*dot(m, grad(p)) = 0           in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = -1
a2 = 1
deltax = 0.1/2/2
intervals_line = round((a2-a1)/deltax)
beta = 1
# box constraints for c, exact solution is in [0,1]
c_upper = 5
c_lower = 0
e1 = 0.2
e2 = 0.3
k1 = 1
k2 = 1

# diffusion coefficient
eps = 0
om = np.pi/40

t0 = 0
dt = 0.001 #deltax**2 #0.01
T = 0.25
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!
example_name = 'solidbody'

# Initialize a square mesh
mesh = RectangleMesh(Point(a1, a1), Point(a2, a2), intervals_line, intervals_line)
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()    
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
v = TestFunction(V)

X = np.arange(a1, a2 + deltax, deltax)
Y = np.arange(a1, a2 + deltax, deltax)
X, Y = np.meshgrid(X,Y)

show_plots = True

def u_init(X,Y):
    '''
    Initialisation of the position of the solid body at time zero.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = np.zeros(X.shape)
    R = np.sqrt(X**2 + (Y-1/3)**2)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if R[i,j] < 1/3 and (abs(X[i,j]) > 0.05 or Y[i,j] > 0.5):
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out

def rotation(X,Y):
    wind = Expression(('-x[1]','x[0]'), degree=4)
    return 1/om*wind

drift = Constant(('1','1'))
rot = rotation(X,Y)

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)
M_diag = M.diagonal()

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# Advection matrix for rotation
Arot = assemble_sparse(dot(rot, grad(v))*u * dx)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time

u0_orig = u_init(X, Y).reshape(nodes)

# importing data in dof ordering
uhat_T = data_helpers.get_data_array('u', example_name, T)

zeros_nt = np.zeros(vec_length)
uk = np.zeros(vec_length)
pk = np.zeros(vec_length)
dot_drift_grad_pk = np.zeros(vec_length)
ck = np.ones(vec_length)
dk = np.zeros(vec_length)

u0 = reorder_vector_to_dof_time(u0_orig, 1, nodes, vertextodof)
uhat_T_re = reorder_vector_from_dof_time(uhat_T, 1, nodes, vertextodof).reshape((sqnodes,sqnodes))

uk[:nodes] = u0
uk[num_steps*nodes:] = uhat_T
c_prev = ck

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
# cost_fun_k = 10*cost_functional_proj_FT(uk, zeros_nt, ck, zeros_nt, 0, uhat_T, np.zeros(nodes), num_steps, dt, M, c_lower, c_upper, beta) 
cost_fun_k = 10* cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta, optim='finaltime')
cost_fun_vals = []
cost_fidelity_vals = []
cost_control_vals = []

RE_costfun = 5 

print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
while(RE_costfun >= tol) and it < 1000:
    it += 1
    print(f'\n{it=}')
    
    # In k-th iteration we solve for u^k, p^k using c^k (S1 & S2)
    # and calculate c^{k+1} (S5)
    
    ###########################################################################
    ############### 1. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) 
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    dot_drift_grad_pk = np.zeros(vec_length)
    t = T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 10 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V) # uk(t_n)
        ck_n_fun = vec_to_function(c_prev[start : end], V)
        
        Adrift1 = assemble_sparse(dot(drift, grad(ck_n_fun))*u * v * dx) # pseudo-mass matrix
        Adrift2 = assemble_sparse(dot(drift, grad(v)) * ck_n_fun * u * dx) # pseudo-stiffness matrix

        A_p = - eps * Ad - Arot - Adrift1 - Adrift2
        
        p_rhs =  np.zeros(nodes)
        
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors)
    
    ###########################################################################
    ##################### 2. choose the descent direction #####################
    ###########################################################################

    for i in range(num_steps + 1): # calculate dk, across all time steps (incl. 0 and T)
        start = i * nodes
        end = (i + 1) * nodes
        
        uk_fun = vec_to_function(uk[start:end], V)
        pk_fun = vec_to_function(pk[start:end], V)
        
        rhs_dk = -(beta*M*c_prev[start:end] \
                    + np.asarray(assemble(pk_fun * dot(drift, grad(uk_fun))*v*dx)))
        
        dk[start:end] = ChebSI(rhs_dk, M, M_diag, 20, 0.5, 2)
    
    
    ###########################################################################
    #### 3. Initialize stepsize s=1, calculate new control and project C_ad ###
    ###########################################################################
    s0 = 1
    ck = np.clip(c_prev + s0*dk, c_lower, c_upper)

    ###########################################################################
    ############### 4. solve the state equation using FCT #####################
    ###########################################################################
    print('Solving state equation...')
    t=0
    uk[nodes:] = np.zeros(num_steps * nodes) # initialise uk, keep IC
    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 10 == 0:
            print('t = ', round(t, 4))
        
        uk_n = uk[start - nodes : start] # uk(t_n), i.e. previous time step at k-th GD iteration
        ck_np1_fun = vec_to_function(ck[start : end], V)

        u_rhs = np.zeros(nodes)
        
        Adrift1 = assemble_sparse(dot(drift, grad(ck_np1_fun))*u * v * dx) # pseudo-mass matrix
        Adrift2 = assemble_sparse(dot(drift, grad(v)) * ck_np1_fun * u * dx) # pseudo-stiffness matrix
        
        ## System matrix for the state equation
        A_u = - eps * Ad + Arot + Adrift1 + Adrift2
        
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
    
    print(f'{L2_norm_sq_Q(uk - uhat_T, num_steps, dt, M)=}')
   
    cost_fun_k = cost_functional(uk, uhat_T, ck, num_steps, dt, M, beta,
                                 optim = 'finaltime')

    
    ###########################################################################
    ########################## 5. step size control ###########################
    ###########################################################################
    
    print('Starting Armijo line search...')

    k = 0 # counter
    s = s0
    cost_dif = 10**5
    gam = 1e-4
    max_iter = 10
    w = TrialFunction(V)
    v = TestFunction(V)
    
    # Stationarity measure: Norm of the projected gradient (Hinze, p. 107)
    stat_measure = L2_norm_sq_Q(np.clip(ck + s * dk, c_lower, c_upper) - ck,
                                 num_steps, dt, M)
    print(f'{stat_measure=}')
    cost_fun_init = cost_fun_k
    # to this point, we have obtained ck that we calculated using the grad. eqn (with pk) and s0 at this iteration
    while cost_dif > - gam / s * stat_measure and k < max_iter:
        print(f'{k =}')

        s = s0*( 1/2 ** k)
        
        # Calculate the incremented c using the new step size
        c_inc = np.clip(ck + s * dk, c_lower, c_upper)
        
        ############### Solve the state equation using FCT ####################
        print('Solving state equations...')
        t = 0
        # initialise u and keep ICs
        uk[nodes:] = np.zeros(num_steps * nodes)
        for i in range(1,num_steps+1):    # solve for f(t_{n+1}), m(t_{n+1})
            start = i * nodes
            end = (i + 1) * nodes
            t += dt
            if i % 20 == 0:
                print('t =', round(t, 4))
            
            uk_n = uk[start - nodes : start] # uk(t_n)
            ck_inc_fun = vec_to_function(c_inc[start : end], V)

            u_rhs = np.zeros(nodes)
            
            Adrift1 = assemble_sparse(dot(drift, grad(ck_inc_fun)) * w * v * dx) # pseudo-mass matrix
            Adrift2 = assemble_sparse(dot(drift, grad(v)) * ck_inc_fun * w * dx) # pseudo-stiffness matrix
            
            ## System matrix for the state equation
            A_u = - eps * Ad + Arot + Adrift1 + Adrift2
            
            uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
            
        #######################################################################
        cost_fun_armijo = cost_functional(uk, uhat_T, c_inc, num_steps, dt, M, 
                                          beta, optim='finaltime')
        cost_dif = cost_fun_armijo - cost_fun_init
        stat_measure = L2_norm_sq_Q(c_inc - ck, num_steps, dt, M)

        k += 1
        
    print(f'Armijo exit at {k=} with {s=}')    
    sk = s
    
    # Save the updated norms to lists
    cost_fun_vals.append(cost_fun_armijo)
    cost_fidelity_vals.append(L2_norm_sq_Q(uk - uhat_T, num_steps, dt, M))
    cost_control_vals.append(L2_norm_sq_Q(c_inc, num_steps, dt, M))
    
    RE_costfun = np.abs(cost_fun_k - cost_fun_armijo) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_armijo
    c_prev = c_inc
    
    print(f'{RE_costfun=}')
    
    # uk.tofile('solidbody_pdeco/solidbody_it' + str(it) + '_u.csv', sep = ',')
    # ck.tofile('solidbody_pdeco/solidbody_it' + str(it) + '_c.csv', sep = ',')
    # pk.tofile('solidbody_pdeco/solidbody_it' + str(it) + '_p.csv', sep = ',')

    
    uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
    ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
    pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)
    
    for i in range(num_steps):
        startP = i * nodes
        endP = (i+1) * nodes
        tP = i * dt
        
        startU = (i+1) * nodes
        endU = (i+2) * nodes
        tU = (i+1) * dt
        
        u_dof = uk[startU : endU]
        # u_dof.tofile('solidbody_pdeco/solidbody_t' + str(tU) + '.csv', sep = ',')
        
        u_re = uk_re[startU : endU]
        c_re = ck_re[startP : endP]
        p_re = pk_re[startP : endP]
            
        u_re = u_re.reshape((sqnodes,sqnodes))
        c_re = c_re.reshape((sqnodes,sqnodes))
        p_re = p_re.reshape((sqnodes,sqnodes))

        
        if show_plots is True and i%5 == 0:
            fig2 = plt.figure(figsize = (20, 5))
            fig2.tight_layout(pad = 3.0)
            ax2 = plt.subplot(1,4,1)
            im1 = plt.imshow(uhat_T_re)
            fig2.colorbar(im1)
            plt.title(f'Desired state for $u$ at t = {T}')
            fig2.tight_layout(pad = 3.0)
            ax2 = plt.subplot(1,4,2)
            im1 = plt.imshow(u_re)
            fig2.colorbar(im1)
            plt.title(f'Computed state $u$ at t = {round(tU,5)}')
            ax2 = plt.subplot(1,4,3)
            im2 = plt.imshow(p_re)
            fig2.colorbar(im2)
            plt.title(f'Computed adjoint $p$ at t = {round(tP,5)}')
            ax2 = plt.subplot(1,4,4)
            im1 = plt.imshow(c_re)
            fig2.colorbar(im1)
            plt.title(f'Computed control $c$ at t = {round(tP,5)}')
            plt.show()
           
            print('------------------------------------------------------')
    
    fig2 = plt.figure(figsize = (15,5))
    fig2.tight_layout(pad = 3.0)
    ax2 = plt.subplot(1,3,1)
    im1 = plt.plot(cost_fun_vals)#, title = 'total cost')
    
    ax2 = plt.subplot(1,3,2)
    im2 = plt.plot(cost_fidelity_vals)# title = 'data fidelity norm')
    
    ax2 = plt.subplot(1,3,3)
    im3 = plt.plot(cost_control_vals)#, title = 'control norm')
    
    plt.show()
    
###############################################################################    
# Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)

# min_u = min(np.amin(uk), np.amin(uex(T, X, Y, e1, k1)))
# min_c = min(np.amin(ck), np.amin(cex(0, X, Y, e2, k2)))
# min_p = min(np.amin(pk), np.amin(pex(0, X, Y, e2, k2)))

# max_u = max(np.amax(uk), np.amax(uex(0,X,Y,e1,k1)))
# max_c = max(np.amax(ck), np.amax(cex((num_steps - 1) * dt, X, Y, e2, k2)))
# max_p = max(np.amax(pk), np.amax(pex((num_steps - 1) * dt, X, Y, e2, k2)))

for i in range(num_steps):
    startP = i * nodes
    endP = (i+1) * nodes
    tP = i * dt
    
    startU = (i+1) * nodes
    endU = (i+2) * nodes
    tU = (i+1) * dt
    
    u_dof = uk[startU : endU]
    u_dof.tofile('solidbody_pdeco/solidbody_t' + str(tU) + '.csv', sep = ',')
    
    
    u_re = uk_re[startU : endU]
    c_re = ck_re[startP : endP]
    p_re = pk_re[startP : endP]

        
    u_re = u_re.reshape((sqnodes,sqnodes))
    c_re = c_re.reshape((sqnodes,sqnodes))
    p_re = p_re.reshape((sqnodes,sqnodes))

    if show_plots is True and i%10 == 0:
        fig2 = plt.figure(figsize = (20, 5))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(1,4,1)
        im1 = plt.imshow(uhat_T_re)
        fig2.colorbar(im1)
        plt.title(f'Desired state for $u$ at t = {T}')
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(1,4,2)
        im1 = plt.imshow(u_re)
        fig2.colorbar(im1)
        plt.title(f'Computed state $u$ at t = {round(tU,5)}')
        ax2 = plt.subplot(1,4,3)
        im2 = plt.imshow(p_re)
        fig2.colorbar(im2)
        plt.title(f'Computed adjoint $p$ at t = {round(tP,5)}')
        ax2 = plt.subplot(1,4,4)
        im1 = plt.imshow(c_re)
        fig2.colorbar(im1)
        plt.title(f'Computed control $c$ at t = {round(tP,5)}')
        plt.show()
        
    print('------------------------------------------------------')

print(f'Exit:\n Stop. crit.: {RE_costfun}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')

