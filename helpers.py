import dolfin as df
from dolfin import dx, dot, grad, div, assemble
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, triu, tril, spdiags
from scipy.sparse.linalg import spsolve
# from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix, lil_matrix, spdiags,

# Contains functions used in all scripts

def reorder_vector_to_dof(vec, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for i in range(nodes):
        j = int(vertextodof[i])
        vec_dof[j] = vec[i]
    return vec_dof

def reorder_vector_to_dof_time(vec, num_steps, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + j] = temp[i] 
    return vec_dof

def reorder_vector_from_dof_time(vec, num_steps, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + i] = temp[j] 
    return vec_dof

def rel_err(new,old):
    '''
    Calculates the relative error between the new and the old value.
    '''
    return np.linalg.norm(new-old)/np.linalg.norm(old)

def assemble_sparse(M):
    '''
    M is a matrix assembled with dolfin of type dolfin.cpp.la.Matrix.
    The output is a sparse, csr matrix.
    '''
    mat = df.as_backend_type(M).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr

def assemble_sparse_lil(M):
    '''
    M is a matrix assembled with dolfin of type dolfin.cpp.la.Matrix.
    The output is a sparse, csr matrix.
    '''
    mat = df.as_backend_type(M).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return lil_matrix(csr)

def vec_to_function(vec, V):
    out = df.Function(V)
    out.vector().set_local(vec)
    return out

def ChebSI(vec, M, Md, cheb_iter, lmin, lmax):
    ymid = 0*vec
    yold = ymid
    omega = 0
    
    rho = (lmax - lmin) / (lmax + lmin)
    Md = (lmin + lmax) / 2 * Md
    
    for k in range(1,cheb_iter + 1):
        if k==2:
            omega = 1 / (1 - rho**2 / 2)
        else:
            omega = 1 / (1 -(omega * rho**2) / 4);
        r = vec - M*ymid #np.dot(M,ymid)                    #?  residual = b - Ax
        z = r / Md  # z = Md\r ? 
        ynew = omega * (z + ymid - yold) + yold
        yold = ymid
        ymid = ynew
    
    return ynew

def boundary(x, on_boundary):
    return on_boundary

def sparse_nonzero(H):
    '''
    Converts a sparse matrix to coo format and returns a table with the
    coordinates of nonzero entries and the values.
    Input: sparse matrix H
    Output: (dense) matrix with columns: i\j\data\+ve sign? (i=row, j=col)
    '''
    Hx = H.tocoo()
    out = np.transpose(np.array([Hx.row, Hx.col, Hx.data, Hx.data>0]))
    return out

def artificial_diffusion_mat(mat):
    '''
    Generates artificial diffusion matrix for a given flux matrix.
    '''
    neg_upper = -triu(mat, k=1, format='csr')
    neg_lower = -tril(mat, k=-1, format='csr')
    D_upper = neg_upper.maximum(0)
    D_lower = neg_lower.maximum(0)
    D = D_upper + D_lower
    D = D.maximum(D.transpose())  # Ensure D is symmetric
    D.setdiag(-D.sum(axis = 1)) # Set the diagonal entries to negative row sums
    return D

def generate_boundary_nodes(nodes, vertextodof):
    '''
    Generates the list of boundary nodes in vertex and dof ordering.
    We assume that the domain is a square.

    '''
    sqnodes = round(np.sqrt(nodes))
    boundary_nodes = []
    for n in range(nodes):
        if n % sqnodes in [0, sqnodes - 1] or n < sqnodes or n >= nodes-sqnodes: 
            boundary_nodes.append(n)        
    boundary_nodes = np.array(boundary_nodes)

    # Mapping to convert boundary node indices to dof indices
    boundary_nodes_dof = [] 
    for i in range(len(boundary_nodes)):
        j = int(vertextodof[boundary_nodes[i]])
        boundary_nodes_dof.append(j)
        
    return boundary_nodes, boundary_nodes_dof

def find_node_neighbours(mesh, nodes, vertextodof):
    '''
    Returns the list of neighbours for each node for a given mesh
    as a list of lists.

    '''
    # Initialize an empty list to store neighbors for each node
    node_neighbors = [[] for _ in range(mesh.num_vertices())]
    for vx in df.vertices(mesh):
        idx = vx.index()
        neighborhood = [df.Edge(mesh, i).entities(0) for i in vx.entities(1)]
        neighborhood = [node_index for sublist in neighborhood for node_index in sublist]

        # Remove own index from neighborhood
        neighborhood = [node_index for node_index in neighborhood if node_index != idx]
        neighborhood.append(idx)
        # Add the neighborhood to the list for the current node
        node_neighbors[idx] = neighborhood

    # Convert node_neighbors to dof_neighbors
    dof_neighbors = [[] for _ in range(nodes)]
    for i in range(nodes):
        j = vertextodof[i]
        dof_neighbors[j] = [vertextodof[node] for node in node_neighbors[i]]
    
    return dof_neighbors

def row_lump(mass_mat, nodes):
    '''
    Matrix lumping by row summing.
    '''
    return spdiags(data=np.transpose(mass_mat.sum(axis = 1)),diags = 0, \
                   m = nodes, n = nodes)

def L2_norm_sq_Q(phi, num_steps, dt, M):
    '''
    Calculates the squared norm in L^2-space of given vector phi using FEM in 
    space and trapezoidal rule in time.
    The vector is of the form phi = (phi0,phi1, phi2, .. ,phi_NT)
    (phi)^2_{L^2(Q)} = int_Q phi^2 dx dy dt = sum_k dt_k/2 phi_k^T M phi_k.
    '''
    v1 = np.split(phi,num_steps+1)
    trapez_coefs = np.ones(num_steps+1)
    trapez_coefs[0] = 0.5
    trapez_coefs[-1] = 0.5
    
    out = sum([trapez_coefs[i]*v1[i].transpose() @ M @ v1[i] for i in range(num_steps+1)]) *dt
    return out

def L2_norm_sq_Omega(phi, M):
    '''
    Calculates the squared norm in L^2-space of given vector phi for one time 
    step using FEM in space.
    (phi)^2_{L^2(Q)} = phi^T M phi.
    '''
    return phi.transpose() @ M @ phi

def cost_functional_proj(u, w, c, d, s, uhatvec, num_steps, dt, M,
                         c_lower, c_upper, beta):
    '''
    Evaluates the cost functional for given values of:
        u: state variable
        w: increment in state variable
        c: control variable
        s: stepsize
        d: direction vector
        uhatvec: desired state
    c is shifted to c_n+1 and projected onto the set of admissible solutions.
    Assume linear equation so increment in u can be precalculated.
    All-time optimization, i.e. desired state across the time interval (0,T].
    '''
    proj = np.clip(c + s*d,c_lower,c_upper)
    f = (L2_norm_sq_Q(u + s*w - uhatvec, num_steps, dt, M) \
         + beta*L2_norm_sq_Q(proj, num_steps, dt, M)) /2
    return f

def cost_functional_proj_FT(m, f, c, d, s, mhatvec, fhatvec, num_steps, 
                               dt, M, c_lower, c_upper, beta):
    '''
    Evaluates the cost functional for given values of 
        m,f: state variables
        c: control variable
        s: stepsize
        d: direction vector
        mhatvec, fhatvec: desired states
    c is shifted to c_n+1 and projected onto the set of admissible solutions.
    Assume non-linear equation, thus m, f for new c are inputs.
    Final-time optimization, i.e. desired states only at some final time T.
    '''
    proj = np.clip(c + s*d, c_lower, c_upper)
    n = mhatvec.shape # number of nodes
    f = (L2_norm_sq_Omega(m[num_steps * n :] - mhatvec, M)
         + L2_norm_sq_Omega(f[num_steps * n :] - fhatvec, M)
         + beta * L2_norm_sq_Q(proj, num_steps, dt, M)) /2
    return f
    
def armijo_line_search(u, p, w, c, d, uhatvec, num_steps, dt, M, c_lower, 
                        c_upper, beta, gam = 10**-4, max_iter = 5, s0 = 1,
                        optim = 'alltime'):
    '''
    Performs Projected Armijo Line Search and returns optimal step size.
    gam: parameter in tolerance for decrease in cost functional value
    max_iter: max number of armijo iterations
    s0: step size at the first iteration.
    optim = {'alltime', 'finaltime'}
    '''
    
    k = 0 # counter
    s = 1 # initial step size
    # Stationarity measure: Norm of the projected gradient (Hinze, p. 107)
    clip = np.clip(c + s * d, c_lower, c_upper)
    grad_costfun_L2 = L2_norm_sq_Q(np.clip(c + s * d, c_lower, c_upper) - c, 
                                  num_steps, dt, M)
    n = uhatvec.shape
    Z = np.zeros(u.shape)
    z = np.zeros(n)
    print(f'{grad_costfun_L2=}')
    if optim == 'alltime':
        # verify before deleting
        # costfun_init = (L2_norm_sq_Q(u - uhatvec, num_steps, dt, M)
        #                 + beta * L2_norm_sq_Q(c, num_steps, dt, M)) / 2
        costfun_init = cost_functional_proj(u, Z, c, d, s, uhatvec, num_steps, 
                                            dt, M, c_lower, c_upper, beta)
    elif optim == 'finaltime':
        costfun_init = cost_functional_proj_FT(u, Z, c, d, s, uhatvec, z, 
                                   num_steps, dt, M, c_lower, c_upper, beta)
    else:
        raise ValueError(f"The selected option {optim} is invalid.")
        
    armijo = 10**5 # initialise the difference in cost function norm decrease
    # note the negative sign in the condition comes from the descent direction
    while armijo > -gam / s * grad_costfun_L2 and k < max_iter:
        s = s0*( 1/2 ** k)
        # Calculate the incremented u using the new step size
        c_inc = np.clip(c - s * (beta * c - p), c_lower, c_upper)
        
        if optim == 'alltime':
            cost2 = cost_functional_proj(u, w, c_inc, d, s, uhatvec, num_steps, 
                                      dt, M, c_lower, c_upper, beta)
        elif optim == 'finaltime':
            u_inc = u[num_steps * n :] + s*w[num_steps * n :]
            cost2 = cost_functional_proj_FT(u_inc, Z, c_inc, d, s, uhatvec, z, 
                                      num_steps, dt, M, c_lower, c_upper, beta)
        armijo = cost2 - costfun_init
        grad_costfun_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)

        # print(f'{k=},{armijo=}')

        k += 1
        
    print(f'Armijo exit at {k=} with {s=}')
    return s



def armijo_line_search_chtxs(m, f, q, c, d, mhatvec, fhatvec, Mat_fq, chi, Dm, Df, num_steps,
                             dt, nodes, M, M_Lump, Ad, c_lower, c_upper, beta, V, dof_neighbors, gam = 10**-4, 
                             max_iter = 5, s0 = 1):
    '''
    Performs Projected Armijo Line Search and returns optimal step size.
    gam: parameter in tolerance for decrease in cost functional value
    max_iter: max number of armijo iterations
    s0: step size at the first iteration.
    m,f: state variables
    q: adjoint variable related to the control by the gradient equation
    mhatvec, fhatvec: desired states for m, f at final time T
    '''
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    k = 0 # counter
    s = 1 # initial step size
    # Stationarity measure: Norm of the projected gradient (Hinze, p. 107)
    grad_costfun_L2 = L2_norm_sq_Q(np.clip(c + s * d, c_lower, c_upper) - c,
                                 num_steps, dt, M)
    print(f'{grad_costfun_L2=}')
    costfun_init = cost_functional_proj_FT(m, f, c, d, s, mhatvec, fhatvec,
                              num_steps, dt, M, c_lower, c_upper, beta)
    
    armijo = 10**5 # initialise the difference in cost function norm decrease
    # note the negative sign in the condition comes from the descent direction
    while armijo > - gam / s * grad_costfun_L2 and k < max_iter:
        s = s0*( 1/2 ** k)
        # Calculate the incremented c using the new step size
        c_inc = np.clip(c - s * (beta * c - m * q), c_lower, c_upper)
        print(f'{k =}')
        ########## calculate new m,f corresponding to c_inc ###################
        print('Solving state equations...')
        t=0
        # initialise m,f and keep ICs
        f[nodes:] = np.zeros(num_steps * nodes)
        m[nodes:] = np.zeros(num_steps * nodes)
        for i in range(1,num_steps+1):    # solve for f(t_{n+1}), m(t_{n+1})
            start = i * nodes
            end = (i + 1) * nodes
            t += dt
            if i % 50 == 0:
                print('t =', round(t, 4))
                
            m_n = m[start - nodes : start]    # m(t_n) 
            m_n_fun = vec_to_function(m_n,V)
            c_inc_fun = vec_to_function(c_inc[start : end],V)
            f_n_fun = vec_to_function(f[start - nodes : start],V)
            
            f_rhs = rhs_chtx_f(f_n_fun, m_n_fun, c_inc_fun, dt, v)
            
            f[start : end] = spsolve(Mat_fq, f_rhs)
            
            f_np1_fun = vec_to_function(f[start : end], V)

            A_m = mat_chtx_m(f_np1_fun, m_n_fun, Dm, chi, u, v)
            m_rhs = rhs_chtx_m(m_n_fun, v)
            
            m[start : end] = FCT_alg(A_m, m_rhs, m_n, dt, nodes, M, M_Lump, 
                                      dof_neighbors)
        
        #######################################################################
        
        cost2 = cost_functional_proj_FT(m, f, c_inc, d, s, mhatvec, \
                           fhatvec, num_steps, dt, M, c_lower, c_upper, beta)
        armijo = cost2 - costfun_init
        grad_costfun_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)

        k += 1
        
    print(f'Armijo exit at {k=} with {s=}')
    return s

def rhs_chtx_m(m_fun, v):
    return np.asarray(assemble(4*m_fun * v * dx))

def rhs_chtx_f(f_fun, m_fun, c_fun, dt, v):
    return np.asarray(assemble(f_fun * v * dx  + dt * m_fun * c_fun * v * dx))

def rhs_chtx_p(c_fun, q_fun, v):
    return np.asarray(assemble(c_fun * q_fun * v * dx))

def rhs_chtx_q(q_fun, m_fun, p_fun, chi, dt, v):
    return np.asarray(assemble(q_fun * v * dx + 
                        dt * div(chi * m_fun * grad(p_fun)) * v * dx))
    
def mat_chtx_m(f_fun, m_fun, Dm, chi, u, v):
    Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))
    Aa = assemble_sparse(assemble(dot(grad(f_fun), grad(v)) * u * dx))
    Ar = assemble_sparse(assemble(m_fun * u * v * dx))
    return - Dm * Ad + chi * Aa + Ar

def mat_chtx_p(f_fun, m_fun, Dm, chi, u, v):
    Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))
    Aa = assemble_sparse(assemble(dot(grad(f_fun), grad(v)) * u * dx))
    Adf = assemble_sparse(assemble(div(grad(f_fun)) * u * v * dx))
    Ar = assemble_sparse(assemble((4 - 2 * m_fun) * u * v * dx))
    return - Dm * Ad - chi * Aa - chi * Adf + Ar
    


def FCT_alg(A, d, u_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat = None):

    D = artificial_diffusion_mat(A)
    M_diag = M.diagonal()
    M_Lump_diag = M_Lump.diagonal()

    ## 1. Calculate low-order solution u^{n+1} using previous time step solution
    if source_mat is None:
        Mat_u_Low = M_Lump - dt * (A + D)
    else:
        Mat_u_Low = M_Lump - dt * (A + D + source_mat)

    Rhs_u_Low = M_Lump @ u_n + dt * d
    u_Low = spsolve(Mat_u_Low, Rhs_u_Low)
    
    ## 2. Calculate raw antidiffusive flux
    # approximate the derivative du/dt using Chebyshev semi-iterative method
    rhs_du_dt = A @ u_Low + d
    du_dt = ChebSI(rhs_du_dt, M, M_diag, 20, 0.5, 2)
    
    # corrected flux calculation(only use neighbouring nodes):
    F = np.zeros((nodes,nodes))
    for i in range(nodes):
        for j in dof_neighbors[i]: # flux from node j to node i, j is a neighbour of i
            F[i,j] = M[i, j] * (du_dt[i] - du_dt[j]) + D[i, j] * (u_Low[i] - u_Low[j])
    F = csr_matrix(F)
    F.setdiag(np.zeros(nodes))
    
    ## 3. Calculate correction factor matrix Alpha 
    # ---------------------  Zalesak algorithm ------------------------------
    # (1) compute sum of pos/neg fluxes into node i (P)
    p_pos = np.ravel(F.maximum(0).sum(axis = 1))
    p_neg = np.ravel(F.minimum(0).sum(axis = 1))
    
    # (2) compute distance to local extremum of u_Low(Q) 
    u_Low_max = np.zeros(nodes)
    u_Low_min = np.zeros(nodes)
    for i in range(nodes):
        # Find the maximum value among the vector elements corresponding to 
        # the node and its neighbors
        max_value = max(u_Low[dof_index] for dof_index in dof_neighbors[i])
        min_value = min(u_Low[dof_index] for dof_index in dof_neighbors[i])
    
        u_Low_max[i] = max_value
        u_Low_min[i] = min_value
       
    q_pos = u_Low_max - u_Low
    q_neg = u_Low_min - u_Low
    
    # (3) compute nodal correction factors (R)
    r_pos = np.ones(nodes)
    r_neg = np.ones(nodes)
    r_pos[p_pos != 0] = np.minimum(1, M_Lump_diag[p_pos != 0]*q_pos[ p_pos!= 0]
                                 / (dt * p_pos[p_pos != 0]))
    r_neg[p_neg != 0] = np.minimum(1, M_Lump_diag[p_neg != 0]*q_neg[ p_neg!= 0]
                                  / (dt * p_neg[p_neg != 0]))
    
    # (4) limit the raw antidiffusive fluxes (calculate correction factors)    
    F_nz = sparse_nonzero(F)
    Fbar = np.zeros(nodes)
    for i in range(F_nz.shape[0]):
        flux_pos = min(r_pos[int(F_nz[i, 0])], r_neg[int(F_nz[i, 1])])*F_nz[i, 2]
        flux_neg = min(r_neg[int(F_nz[i, 0])], r_pos[int(F_nz[i, 1])]) * F_nz[i, 2]

        Fbar[int(F_nz[i, 0])] += F_nz[i, 3]*flux_pos + (1-F_nz[i,3])*flux_neg
    # -----------------------------------------------------------------------
    
    ## 4. Correct u_Low^{n+1} explicitly:
    u_np1 = u_Low + dt*Fbar/M_Lump_diag
    return u_np1