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
# min_{u,v,a,b} 1/2*||u-\hat{u}||^2 + beta/2*||c||^2  (norms in L^2)
# subject to:
#  du/dt - eps*grad^2(u) + w \dot grad(u)) = c + g       in Ωx[0,T]
#                           dot(grad u, n) = 0           on ∂Ωx[0,T]
#                                    du/dn = 0           on ∂Ωx[0,T]
#                                     u(0) = u0(x)       in Ω

# w = velocity/wind vector with the following properties:
#                                 div (w) = 0           in Ωx[0,T]
#                                w \dot n = 0           on ∂Ωx[0,T]

# Optimality conditions:
#  du/dt - eps*grad^2(u) + w \dot grad(u)) = c + g                 in Ωx[0,T]
#    -dp/dt - eps*grad^2 p - w \dot grad(p)= \hat{u} - u           in Ωx[0,T]
#                            dp/dn = du/dn = 0                     on ∂Ωx[0,T]
#                                     u(0) = u0(x)                 in Ω
#                                     p(T) = 0                     in Ω
# gradient equation:                     c = 1 \ beta * p
#                                        c = proj_[ca,cb] (1/beta*p) in Ωx[0,T]
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
deltax = 0.1
intervals_line = round((a2-a1)/deltax)
beta = 0.1
# box constraints for c, exact solution is in [0,1]
c_upper = 0.5
c_lower = 0
e1 = 0.2
e2 = 0.3
k1 = 1
k2 = 1

# diffusion coefficient
eps = 0.001

t0 = 0
dt = deltax**2 #0.01
T = 1
num_steps = round((T-t0)/dt)
tol = 10**-5 # !!!

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

def uex(t,X,Y,e1,k1):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = np.exp(e1*t) * ( np.sin(k1*np.pi*X) * np.sin(k1*np.pi*Y) )**2
    return out

def pex(t,X,Y,e2,k2):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions), time
    '''
    out = (np.exp(e2*T) - np.exp(e2*t)) * ( np.sin(k2*np.pi*X) * np.sin(k2*np.pi*Y) )**2
    return out

def cex(t,X,Y,e2,k2):
    # project the control on [u_a,u_b]
    out = np.clip(1/beta*pex(t,X,Y,e2,k2),c_lower,c_upper)
    
    return out

def gex(t,X,Y,e1,e2,k1,k2,eps):
    '''
    Source function for the exact solution in the convection-diffusion equation.
    '''
    u_ex = uex(t,X,Y,e1,k1)
    c_ex = cex(t,X,Y,e2,k2)
    wx,wy,_ = velocity(X,Y,a1,a2)
    dudx = 2*k1*np.pi*np.exp(e1*t)*np.sin(k1*np.pi*X)*np.cos(k1*np.pi*X) * np.sin(k1*np.pi*Y)**2
    dudy = 2*k1*np.pi*np.exp(e1*t)*np.sin(k1*np.pi*X)**2 *np.sin(k1*np.pi*Y) *np.cos(k1*np.pi*Y) 
    du2dx2 = 2*(np.pi*k1)**2 * np.exp(e1*t) * np.cos(2*k1*np.pi*X) *  np.sin(k1*np.pi*Y)**2
    du2dy2 = 2*(np.pi*k1)**2 * np.exp(e1*t) * np.sin(k1*np.pi*X)**2 * np.cos(2*k1*np.pi*Y)
    
    out = e1*u_ex - eps*(du2dx2 + du2dy2) + wx*dudx + wy*dudy - c_ex
    return out

def uhatex(t,X,Y,e1,e2,k1,k2,eps):
    '''
    Source function for the exact solution in the convection-diffusion equation.
    '''
    u_ex = uex(t,X,Y,e1,k1)
    wx,wy,_ = velocity(X,Y,a1,a2)
    
    dpdt = -e2* np.exp(e2*t) * ( np.sin(k2*np.pi*X) * np.sin(k2*np.pi*Y) )**2
    dpdx = 2*k2*np.pi*(np.exp(e2*T) - np.exp(e2*t)) *np.sin(k2*np.pi*X)*np.cos(k2*np.pi*X) * np.sin(k2*np.pi*Y)**2
    dpdy = 2*k2*np.pi*(np.exp(e2*T) - np.exp(e2*t)) *np.sin(k2*np.pi*X)**2 *np.sin(k2*np.pi*Y) *np.cos(k2*np.pi*Y) 
    dp2dx2 = 2*(np.pi*k2)**2 * (np.exp(e2*T) - np.exp(e2*t)) * np.cos(2*k2*np.pi*X) *  np.sin(k2*np.pi*Y)**2
    dp2dy2 = 2*(np.pi*k2)**2 * (np.exp(e2*T) - np.exp(e2*t)) * np.sin(k2*np.pi*X)**2 * np.cos(2*k2*np.pi*Y)
    
    out = -dpdt - eps*(dp2dx2 + dp2dy2) - wx*dpdx - wy*dpdy + u_ex
    return out

def velocity(X,Y,a1,a2):
    
    if a1==0 and a2==1: ## [0,1]^2
        wx = 2*(Y-0.5)*X*(1-X)
        wy = -2*(X-0.5)*Y*(1-Y)
        wind = Expression(('2*(x[1]-0.5)*x[0]*(1-x[0])','-2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4)

    elif a1==-1 and a2==1: ## [-1,1]^2
        wx = 2*Y*(1+X)*(1-X)
        wy = -2*X*(1+Y)*(1-Y)
        wind = Expression(('2*x[1]*(1+x[0])*(1-x[0])','-2*x[0]*(1+x[1])*(1-x[1])'), degree=4)
    else:
        raise ValueError("No velocity field defined for the domain specified.")

    return wx, wy, wind

vertextodof = vertex_to_dof_map(V)
boundary_nodes, boundary_nodes_dof = generate_boundary_nodes(nodes, vertextodof)

mesh.init(0, 1)
dof_neighbors = find_node_neighbours(mesh, nodes, vertextodof)

_,_,wind = velocity(X,Y,a1,a2)

# ----------------------------------------------------------------------------

###############################################################################
################### Define the stationary matrices ###########################
###############################################################################

# Mass matrix
M = assemble_sparse_lil(assemble(u * v * dx))

# Row-lumped mass matrix
M_Lump = row_lump(M,nodes)

# Stiffness matrix
Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))

# Advection matrix
A = assemble_sparse(assemble(dot(wind, grad(v))*u * dx))

## System matrix for the state equation
A_u = A - eps * Ad

## System matrix for the adjoint equation (opposite sign of transport matrix)
A_p = - A - eps * Ad

zeros = np.zeros(nodes)

###############################################################################
########################### Initial guesses for GD ############################
###############################################################################

vec_length = (num_steps + 1)*nodes # include zero and final time
g_orig = np.zeros(vec_length)
uhat_orig = np.zeros(vec_length)
for i in range(0, num_steps + 1): 
    start = i * nodes
    end = (i + 1) * nodes
    g_orig[start : end] = gex(i * dt, X, Y, e1, e2, k1, k2, eps).reshape(nodes)
    uhat_orig[start : end] = uhatex(i * dt, X, Y, e1, e2, k1, k2, eps).reshape(nodes)

zeros_nt = np.zeros(vec_length)
uk = np.zeros(vec_length)
pk = np.zeros(vec_length)
ck = np.zeros(vec_length)
dk = np.zeros(vec_length)
uhat = np.zeros(vec_length)
wk = np.zeros(vec_length)

g = reorder_vector_to_dof_time(g_orig, num_steps + 1, nodes, vertextodof)
uhat = reorder_vector_to_dof_time(uhat_orig, num_steps + 1, nodes, vertextodof)
u0 = reorder_vector_to_dof_time(uex(0, X, Y, e1, k1).reshape(nodes), 1, nodes, vertextodof)

uk[:nodes] = u0
wk[:nodes] = u0

###############################################################################
###################### PROJECTED GRADIENT DESCENT #############################
###############################################################################

it = 0
cost_fun_k = 10*cost_functional_proj(uk, zeros_nt, ck, zeros_nt, 0, uhat, num_steps, dt, M, c_lower, c_upper, beta)

stop_crit = 5
stop_crit2 = 5 

print(f'dx={deltax}, {dt=}, {T=}, {beta=}')
print('Starting projected gradient descent method...')
# while (stop_crit >= tol ) and it<1000:
while ((stop_crit >= tol ) or (stop_crit2 >= tol)) and it < 1000:
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
        g_np1_fun = vec_to_function(g[start : end], V) # g(t_{n+1})

        u_rhs = np.asarray(assemble((g_np1_fun + ck_np1_fun) * v * dx))
        uk[start:end] = FCT_alg(A_u, u_rhs, uk_n, dt, nodes, M, M_Lump, dof_neighbors)
        
    ###########################################################################
    ############### 2. solve the adjoint equation using FCT ###################
    ###########################################################################

    pk = np.zeros(vec_length) # includes the final-time condition
    t=T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes] # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V) # uk(t_n)
        uhat_n_fun = vec_to_function(uhat[start : end], V) # uhat(t_n)
        
        p_rhs =  np.asarray(assemble((uhat_n_fun - uk_n_fun) * v * dx))
        pk[start:end] = FCT_alg(A_p, p_rhs, pk_np1, dt, nodes, M, M_Lump, dof_neighbors)
        
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
        
        # uses the same advection matrix as u (A_u)
        w_rhs = np.asarray(assemble(dk_np1_fun * v * dx))
        wk[start:end] = FCT_alg(A_u, w_rhs, wk_n, dt, nodes, M, M_Lump, dof_neighbors)

    print('Starting Armijo line search...')
    sk = armijo_line_search(uk, pk, wk, ck, dk, uhat, num_steps, dt, M, c_lower, c_upper, beta)
    
    ###########################################################################
    ## 5. Calculate new control and project onto admissible set
    ###########################################################################

    ckp1 = np.clip(ck + sk*dk,c_lower,c_upper)
    
    stop_crit = L2_norm_sq_Q(ckp1-ck, num_steps, dt, M) /  L2_norm_sq_Q(ck, num_steps, dt, M)

    # Check the cost functional - stopping criterion
    cost_fun_kp1 = cost_functional_proj(uk, wk, ckp1, dk, sk, uhat, num_steps, dt, M, c_lower, c_upper, beta)
    stop_crit2 = np.abs(cost_fun_k - cost_fun_kp1) / np.abs(cost_fun_k)
    
    cost_fun_k = cost_fun_kp1
    ck = ckp1
    print(f'{stop_crit=}')
    print(f'{stop_crit2=}')
    
###############################################################################    
# Mapping to order the solution vectors based on vertex indices
uk_re = reorder_vector_from_dof_time(uk, num_steps + 1, nodes, vertextodof)
ck_re = reorder_vector_from_dof_time(ck, num_steps + 1, nodes, vertextodof)
pk_re = reorder_vector_from_dof_time(pk, num_steps + 1, nodes, vertextodof)

rel_errs_u, rel_errs_c, rel_errs_p = ([] for _ in range(3))
werrs_u, werrs_c, werrs_p = ([] for _ in range(3))
errs_u, errs_c, errs_p = ([] for _ in range(3))

min_u = min(np.amin(uk), np.amin(uex(T, X, Y, e1, k1)))
min_c = min(np.amin(ck), np.amin(cex(0, X, Y, e2, k2)))
min_p = min(np.amin(pk), np.amin(pex(0, X, Y, e2, k2)))

max_u = max(np.amax(uk), np.amax(uex(0,X,Y,e1,k1)))
max_c = max(np.amax(ck), np.amax(cex((num_steps - 1) * dt, X, Y, e2, k2)))
max_p = max(np.amax(pk), np.amax(pex((num_steps - 1) * dt, X, Y, e2, k2)))

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
        
    u_ex = uex(tU,X,Y,e1,k1).reshape(nodes)
    ## adjoint variables and control are one step behind:
    c_ex = cex(tP,X,Y,e2,k2).reshape(nodes)
    p_ex = pex(tP,X,Y,e2,k2).reshape(nodes)
    
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
    
    if show_plots is True and i%10 == 0:
        fig2 = plt.figure(figsize = (10,15))
        fig2.tight_layout(pad = 3.0)
        ax2 = plt.subplot(3,2,1)
        im1 = plt.imshow(u_ex, vmin = min_u, vmax = max_u)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $u$ at t = {round(tU,5)}')
        ax2 = plt.subplot(3,2,2)
        im2 = plt.imshow(u_re, vmin = min_u, vmax = max_u)
        fig2.colorbar(im2)
        plt.title(f'Computed state $u$ at t = {round(tU,5)}')
        
        ax2 = plt.subplot(3,2,3)
        im1 = plt.imshow(c_ex, vmin = min_c, vmax = max_c)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $c$ at t = {round(tP,5)}')
        ax2 = plt.subplot(3,2,4)
        im2 = plt.imshow(c_re, vmin = min_c, vmax = max_c)
        fig2.colorbar(im2)
        plt.title(f'Computed state $c$ at t = {round(tP,5)}')
        
        ax2 = plt.subplot(3,2,5)
        im1 = plt.imshow(p_ex, vmin = min_p, vmax = max_p)
        fig2.colorbar(im1)
        plt.title(f'Exact solution $p$ at t = {round(tP,5)}')
        ax2 = plt.subplot(3,2,6)
        im3 = plt.imshow(p_re, vmin = min_p, vmax = max_p)
        fig2.colorbar(im3)
        plt.title(f'Computed control $p$ at t = {round(tP,5)}')
        plt.show()
        # filename = f'advection_FCT_PGD_upper05/plot_{i:03}.png'  # e.g., plot_001.png, plot_002.png, etc.
        # plt.savefig(filename)
        # plt.close()
        
    print('------------------------------------------------------')

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

