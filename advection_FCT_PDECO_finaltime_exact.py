from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags, triu, tril
from timeit import default_timer as timer
from datetime import timedelta
from scipy.integrate import simps
from helpers import *

# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the advection-diffusion equation
### with Flux-corrected transport method
## uses desired states only at final time
# min_{u,v,a,b} 1/2*||u(T)-\hat{u}_T||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt - eps*grad^2(u) + div(w*u) + gu = c + f       in Ωx[0,T]
#                         dot(grad u, n) = 0           on ∂Ωx[0,T]
#                                   u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                               w \dot n = 0           on ∂Ωx[0,T]

# Optimality conditions:
#         du/dt - eps*grad^2(u) + grad(w*u) + gu = c + f            in Ωx[0,T]
#    -dp/dt - eps*grad^2 p - w \dot grad(p) + gp = 0                in Ωx[0,T]
#                                  dp/dn = du/dn = 0                on ∂Ωx[0,T]
#                                           u(0) = u0(x)            in Ω
#                                           p(T) = uhat_T - u(T)    in Ω
# gradient equation:                     c = 1 \ beta * p
#                                        c = proj_[ca,cb] (1/beta*p) in Ωx[0,T]

# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1/2
intervals_line = round((a2-a1)/deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = 1
c_lower = 0
e1 = 1 #0.2
e2 = 1 #0.3
k1 = 1
k2 = 1
k3 = 1
k4 = 1
delta_ex = 0.1 #0.5 #1e-1
gamma = 0.1

# diffusion coefficient
eps = 0.0001

t0 = 0
dt = deltax**2 #0.01
T = 0.1
num_steps = round((T-t0)/dt)
tol = 10**-4 # !!!

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

def uex(t,X,Y,e1,k1,k2):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = np.exp(e1*t) * (np.cos(k1*np.pi*X) * np.cos(k2*np.pi*Y) + 1)
    return out

def pex(t,X,Y,e2,k3,k4):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = (np.exp(e2*T) - np.exp(e2*t)) * np.cos(k3*np.pi*X) * np.cos(k4*np.pi*Y)
    return out

def cex(t,X,Y,e2,k3,k4):
    # project the control on [u_a,u_b]
    out = np.clip(1/beta*pex(t,X,Y,e2,k3,k4),c_lower,c_upper)
    
    return out

def gex(t,X,Y,e2,k2,k3,k4,eps,gamma,delta_ex):
    
    out = - e2*np.exp(e2*t)/(np.exp(e2*T) - np.exp(e2*t*(1-delta_ex))) \
        - eps*(k3**2 + k4**2)*np.pi**2 \
            - gamma*np.pi*(k3*np.sin(k3*np.pi*X)**2 + k4*np.sin(k4*np.pi*Y)**2)
    
    # out = - e2*np.exp(e2*t)/(np.exp(e2*T) - np.exp(e2*t) + delta_ex) \
    #     - eps*(k3**2 + k4**2)*np.pi**2 \
    #         - gamma*np.pi*(k3*np.sin(k3*np.pi*X)**2 + k4*np.sin(k4*np.pi*Y)**2)
    
    return out

def fex(t,X,Y,e1,e2,k1,k2,k3,k4,eps,gamma,delta_ex):
    '''
    Source function for the exact solution in the convection-diffusion equation.
    '''
    u_ex = uex(t,X,Y,e1,k1,k2)
    c_ex = cex(t,X,Y,e2,k3,k4)
    g_ex = gex(t,X,Y,e2,k2,k3,k4,eps,gamma,delta_ex)
    wx,wy,_ = velocity(X,Y,a1,a2,k3,k4,gamma) # includes gamma

    term1 = e1 * u_ex
    term2 = eps * (k1**2 + k2**2) * np.pi**2 * (u_ex - np.exp(e1*t))
    term3 = gamma * np.pi * (k3 * np.cos(2*k3*np.pi*X) + k4 * np.cos(2*k4*np.pi*Y)) * u_ex
    term4 = -np.exp(e1*t) * np.pi * k1 * wx * np.sin(k1*np.pi*X) * np.cos(k2*np.pi*Y)
    term5 = -np.exp(e1*t) * np.pi * k2 * wy * np.cos(k1*np.pi*X) * np.sin(k2*np.pi*Y)
    term6 = g_ex * u_ex
    term7 = -c_ex
    
    out = term1 + term2 + term3 + term4 + term5 + term6 + term7
    # out = term1 + term2 + term4 + term5 + term6 + term7
    
    return out

def uhatex(t,X,Y,e1,e2,k1,k2,eps):
    '''
    Source function for the exact solution in the convection-diffusion equation.
    '''
    u_ex = uex(t,X,Y,e1,k1,k2)

    return u_ex

def velocity(X,Y,a1,a2,k3,k4,gamma):
    if a1==0 and a2==1: ## [0,1]^2
        wx = gamma*np.sin(k3*np.pi*X) * np.cos(k3*np.pi*X)
        wy = gamma*np.sin(k4*np.pi*Y) * np.cos(k4*np.pi*Y)
        
        wind = Expression(('gamma*sin(k3*pi*x[0])*cos(k3*pi*x[0])',
                           'gamma*sin(k4*pi*x[1])*cos(k4*pi*x[1])'), 
                          degree=4, k3=k3, k4=k4, pi=np.pi, gamma=gamma)
    else:
        raise ValueError("No velocity field defined for the domain specified.")

    return wx, wy, wind

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

_,_,wind = velocity(X,Y,a1,a2,k3,k4,gamma)
W = VectorFunctionSpace(mesh, "CG", 1)
wind_fun = Function(W)
wind_fun = project(wind, W)
# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(u * v * dx)

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)

# Advection matrices
Aa1 = assemble_sparse(dot(wind, grad(v))*u * dx) ## int_\Omega w*grad(v) *u*dx
Aa2 = assemble_sparse(div(wind_fun)*v*u * dx)    ## int_\Omega div(w) * v*u*dx

## System matrix for the state equation
# A_u = Aa1 + Aa2 - eps * Ad
## debug- new A_u
A_u = Aa1 - eps * Ad

## System matrix for the adjoint equation (opposite sign of transport matrix)
A_p = - Aa1 - Aa2 - eps * Ad

zeros = np.zeros(nodes)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time
f_orig = np.zeros(vec_length)
g_orig = np.zeros(vec_length)
c_ex_orig = np.zeros(vec_length)
for i in range(0, num_steps + 1): 
    start = i * nodes
    end = (i + 1) * nodes
    f_orig[start : end] = fex(i * dt, X, Y, e1, e2, k1, k2, k3, k4, eps, gamma, delta_ex).reshape(nodes)
    g_orig[start : end] = gex(i * dt, X, Y, e2, k2, k3, k4, eps, gamma, delta_ex).reshape(nodes)
    c_ex_orig[start : end] = cex(i * dt, X, Y, e2, k3, k4).reshape(nodes)
    
zeros_nt = np.zeros(vec_length)
uk = np.zeros(vec_length)
pk = np.zeros(vec_length)
ck = np.zeros(vec_length)#np.ones(vec_length) #np.zeros(vec_length)
dk = np.zeros(vec_length)
wk = np.zeros(vec_length)

f = reorder_vector_to_dof_time(f_orig, num_steps + 1, nodes, vertextodof)
g = reorder_vector_to_dof_time(g_orig, num_steps + 1, nodes, vertextodof)
u0 = reorder_vector_to_dof_time(uex(0, X, Y, e1, k1, k2).reshape(nodes), 1, nodes, vertextodof)
c_ex = reorder_vector_to_dof_time(c_ex_orig, num_steps + 1, nodes, vertextodof)
uhat_T = reorder_vector_to_dof_time(uhatex(T, X, Y, e1, e2, k1, k2, eps).reshape(nodes), 1, nodes, vertextodof)

uk[:nodes] = u0
wk[:nodes] = u0

c_mean_L2Q2 = L2_norm_sq_Q(c_ex, num_steps, dt, M)

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional_proj_FT(uk, zeros_nt, ck, zeros_nt, 0, uhat_T, np.zeros(nodes), num_steps, dt, M, c_lower, c_upper, beta) 
cost_fun_vals = []
cost_fidelity_vals = []
cost_control_vals = []

stop_crit = 5

print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
# while  (stop_crit >= tol and cost_fun_k >= 0.02) and it < 100:
while stop_crit >= tol and it < 4:

    it += 1
    print(f'\n{it=}')
        
    # In k-th iteration we solve for u^k, p^k using c^k (S1 & S2)
    # and calculate c^{k+1} (S5)
    
    ###########################################################################
    ############### 1. solve the state equation using FCT #####################
    ###########################################################################
    print('Solving state equation...')
    t=0
    uk[nodes:] = np.zeros(num_steps * nodes) # initialise uk, keep IC
    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        uk_n = uk[start - nodes : start] # uk(t_n), i.e. previous time step at k-th GD iteration
        ck_np1_fun = vec_to_function(ck[start : end], V)
        ### debug: use the exact solution for c in state equation
        # ck_np1_fun = vec_to_function(c_ex[start : end], V)
        
        f_np1_fun = vec_to_function(f[start : end], V) # f(t_{n+1})
        g_np1_fun = vec_to_function(g[start : end], V) # g(t_{n+1})
        
        # Mg = assemble_sparse(g_np1_fun*u*v*dx)
        # u_rhs = np.asarray(assemble((f_np1_fun + ck_np1_fun) * v * dx))
        # uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=Mg)
        
        ## IMEX -  for the source term use previous time step
        g_n_fun = vec_to_function(g[start - nodes : start], V) # g(t_n)
        Mg = assemble_sparse(g_n_fun*u*v*dx)
        u_rhs = np.asarray(assemble((f_np1_fun + ck_np1_fun) * v * dx))
        u_rhs -= Mg @ uk_n
        
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)

        
        ## debug: don't use FCT
        # A = assemble_sparse(dot(wind, grad(v))*u * dx)
        # A_u = -eps*Ad + A
        # matr = M - dt*(A_u - Mg)
        # rhs = M * uk_n + dt * u_rhs
        # uk[start:end] = spsolve(matr, rhs)
        
    ###########################################################################
    ############### 2. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) # includes the final-time condition
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    
    plt.imshow(reorder_vector_from_dof_time(pk[num_steps * nodes :], 1, nodes, vertextodof).reshape((sqnodes,sqnodes)))
    plt.colorbar()
    plt.title(f'p(T) at {it=}')
    plt.show()
    
    t=T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        
        # p_rhs =  np.zeros(nodes)
        # g_n_fun = vec_to_function(g[start : end], V) # g(t_{n+1})
        # Mg = assemble_sparse(g_n_fun*u*v*dx)
        # pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors, source_mat=Mg)
        
        ## IMEX -  for the source term use the next time step
        g_np1_fun = vec_to_function(g[start : end], V) # g(t_{n+1})
        Mg = assemble_sparse(g_np1_fun*u*v*dx)
        p_rhs =  np.zeros(nodes)
        p_rhs -= Mg @ pk_np1
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors)

        
        # # debug: don't use FCT
        # Aa1 = assemble_sparse(dot(wind, grad(v))*u * dx) ## int_\Omega w*grad(v) *u*dx
        # Aa2 = assemble_sparse(div(wind_fun)*v*u * dx)    ## int_\Omega div(w) * v*u*dx

        # A_p = -eps*Ad - Aa1 - Aa2
        # matr = M - dt*(A_p - Mg)
        # rhs = M * pk_np1 + dt * p_rhs
        # pk[start:end] = spsolve(matr, rhs)

    ###########################################################################
    ##################### 3. choose the descent direction #####################
    ###########################################################################

    dk = -(beta*ck - pk)
    
    ###########################################################################
    ########################## 4. step size control ###########################
    ###########################################################################
    
    print('Solving equation for move in u...')
    t=0
    wk[nodes:] = np.zeros(num_steps * nodes)
    for i in range(1, num_steps + 1):
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        wk_n = wk[start - nodes : start]
        dk_np1_fun = vec_to_function(dk[start : end], V)
        wk_n_fun = vec_to_function(wk_n, V)
        g_np1_fun = vec_to_function(g[start : end], V) # g(t_{n+1})
        Mg = assemble_sparse(g_np1_fun*u*v*dx)
        
        # uses the same advection matrix as u (A_u)
        # w_rhs = np.asarray(assemble(dk_np1_fun * v * dx))
        # wk[start:end] = FCT_alg(A_u, w_rhs, wk_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=Mg)
        
        ## IMEX -  for the source term use previous time step
        g_n_fun = vec_to_function(g[start - nodes : start], V) # g(t_n)
        Mg = assemble_sparse(g_n_fun*u*v*dx)
        w_rhs = np.asarray(assemble(dk_np1_fun * v * dx))
        w_rhs -= Mg @ wk_n
        
        wk[start:end] = FCT_alg(A_u, w_rhs, wk_n, dt, nodes, M, M_Lump, dof_neighbors)

    print(f'{cost_fun_k=}')
    print('\nStarting Armijo line search...')
    sk, u_inc = armijo_line_search(uk, pk, wk, ck, dk, uhat_T, num_steps, dt, 
                                   M, c_lower, c_upper, beta, cost_fun_k, 
                                   optim = 'finaltime')
    
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    # Check the cost functional - stopping criterion
    cost_fun_kp1 = cost_functional(u_inc, uhat_T, ckp1, num_steps, dt, M, beta,
                                   optim='finaltime')
    print(f'{cost_fun_kp1=}')

    cost_fun_vals.append(cost_fun_kp1)
    cost_fidelity_vals.append(L2_norm_sq_Omega(u_inc[num_steps*nodes:] - uhat_T, M))
    cost_control_vals.append(L2_norm_sq_Q(ckp1, num_steps, dt, M))
    
    stop_crit = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit=}')
    
    if it > 1:
        fig2 = plt.figure(figsize = (15,5))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(1,3,1)
        im1 = plt.plot(np.arange(1, it + 1), cost_fun_vals)
        plt.title('total cost')#' as sum of squared L2 norms')
        
        ax2 = plt.subplot(1,3,2)
        im2 = plt.plot(np.arange(1, it + 1), cost_fidelity_vals)
        plt.title('data fidelity norm in L2(Omega)^2')
        
        ax2 = plt.subplot(1,3,3)
        im3 = plt.plot(np.arange(1, it + 1), cost_control_vals)
        plt.plot(np.arange(1, it + 1), np.ones(len(cost_control_vals))*c_mean_L2Q2)
        plt.title('control norm in L2(Q)^2')
        
        plt.show()
    
###############################################################################    
# Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)

rel_errs_u, rel_errs_c, rel_errs_p = ([] for _ in range(3))
werrs_u, werrs_c, werrs_p = ([] for _ in range(3))
errs_u, errs_c, errs_p = ([] for _ in range(3))

min_u = min(np.amin(uk), np.amin(uex(T, X, Y, e1, k1, k2)))
min_c = min(np.amin(ck), np.amin(cex(0, X, Y, e2, k3, k4)))
min_p = min(np.amin(pk), np.amin(pex(0, X, Y, e2, k3, k4)))

max_u = max(np.amax(uk), np.amax(uex(0, X, Y, e1, k1, k2)))
max_c = max(np.amax(ck), np.amax(cex((num_steps - 1) * dt, X, Y, e2, k3, k4)))
max_p = max(np.amax(pk), np.amax(pex((num_steps - 1) * dt, X, Y, e2, k3, k4)))

for i in range(num_steps):
    startP = i * nodes
    endP = (i+1) * nodes
    tP = i * dt
    
    startU = (i+1) * nodes
    endU = (i+2) * nodes
    tU = (i+1) * dt
    
    u_re = uk_re[startU : endU]
    c_re = ck_re[startP : endP]
    p_re = pk_re[startP : endP]
        
    u_ex = uex(tU,X,Y,e1,k1,k2).reshape(nodes)
    ## adjoint variables and control are one step behind:
    c_ex = cex(tP,X,Y,e2,k3,k4).reshape(nodes)
    p_ex = pex(tP,X,Y,e2,k3,k4).reshape(nodes)
    
    E_u = np.linalg.norm(u_ex - u_re)
    E_c = np.linalg.norm(c_ex - c_re)
    E_p = np.linalg.norm(p_ex - p_re)

    RE_u = E_u / np.linalg.norm(u_ex)
    RE_c = E_c / np.linalg.norm(c_ex)
    RE_p = E_p / np.linalg.norm(p_ex)

    WE_u = deltax * E_u
    WE_c = deltax * E_c
    WE_p = deltax * E_p

    errs_u.append(E_u)
    errs_c.append(E_c)
    errs_p.append(E_p)
    
    rel_errs_u.append(RE_u)
    rel_errs_c.append(RE_c)
    rel_errs_p.append(RE_p)
    
    werrs_u.append(WE_u)
    werrs_c.append(WE_c)
    werrs_p.append(WE_p)

    u_re = u_re.reshape((sqnodes,sqnodes))
    c_re = c_re.reshape((sqnodes,sqnodes))
    p_re = p_re.reshape((sqnodes,sqnodes))
    
    u_ex = u_ex.reshape((sqnodes,sqnodes))
    c_ex = c_ex.reshape((sqnodes,sqnodes))
    p_ex = p_ex.reshape((sqnodes,sqnodes))
    
    if show_plots is True and i%5 == 0:
        fig2 = plt.figure(figsize = (10,15))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(3,2,1)
        im1 = plt.imshow(u_ex)#, vmin = min_u, vmax = max_u)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $u$ at t = {round(tU,5)}')
        ax2 = plt.subplot(3,2,2)
        im2 = plt.imshow(u_re)#, vmin = min_u, vmax = max_u)
        fig2.colorbar(im2)
        plt.title(f'Computed state $u$ at t = {round(tU,5)}')
        
        ax2 = plt.subplot(3,2,3)
        im1 = plt.imshow(c_ex)#, vmin = min_c, vmax = max_c)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $c$ at t = {round(tP,5)}')
        ax2 = plt.subplot(3,2,4)
        im2 = plt.imshow(c_re)#, vmin = min_c, vmax = max_c)
        fig2.colorbar(im2)
        plt.title(f'Computed state $c$ at t = {round(tP,5)}')
        
        ax2 = plt.subplot(3,2,5)
        im1 = plt.imshow(p_ex)#, vmin = min_p, vmax = max_p)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $p$ at t = {round(tP,5)}')
        ax2 = plt.subplot(3,2,6)
        im3 = plt.imshow(p_re)#, vmin = min_p, vmax = max_p)
        fig2.colorbar(im3)
        plt.title(f'Computed control $p$ at t = {round(tP,5)}')
        plt.show()
        
    print('------------------------------------------------------')

plt.imshow(uhatex(T, X, Y, e1, e2, k1, k2, eps))
plt.colorbar()
plt.show()

plt.plot(cost_fun_vals)
plt.show()

print(f'Exit:\n Stop. crit.: {stop_crit}\n Iterations: {it}\n dx={deltax}')
print(f'{dt=}, {T=}, {beta=}')
print('Relative errors')
print('u:', max(rel_errs_u))
print('c:', max(rel_errs_c))
print('p:', max(rel_errs_p))
print('Weighted errors')
print('u:', max(werrs_u))
print('c:', max(werrs_c))
print('p:', max(werrs_p))
print(max(rel_errs_u), ',', max(rel_errs_c), ',', max(rel_errs_p), ',', max(werrs_u), ',', max(werrs_c), ',', max(werrs_p), ',', it )

