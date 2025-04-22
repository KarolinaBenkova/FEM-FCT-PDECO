def armijo_line_search_sbr_drift(u, p, c, d, uhatvec, eps, drift, num_steps, dt, nodes, M, M_Lump, Ad, Arot,
                                 c_lower, c_upper, beta, V, dof_neighbors, gam = 10**-4, max_iter = 5, s0 = 1,
                                 optim = "alltime"):
        
    """
    Performs Projected Armijo Line Search and Returns
  optimal step size.
    gam: parameter in tolerance for decrease in cost functional value
    max_iter: max number of armijo iterations
    s0: step size at the first iteration.
    m,f: state variables
    q: adjoint variable related to the control by the gradient equation
    mhatvec, fhatvec: desired states for m, f at final time T
    """
    

    k = 0 # counter
    s = 1 # initial step size
    n = uhatvec.shape[0]
    Z = np.zeros(u.shape)
    z = np.zeros(n)
    
    w = df.TrialFunction(V)
    v = df.TestFunction(V)
    
    # Stationarity measure: Norm of the projected gradient (Hinze, p. 107)
    grad_costfun_L2 = L2_norm_sq_Q(np.clip(c + s * d, c_lower, c_upper) - c,
                                 num_steps, dt, M)
    print(f"{grad_costfun_L2=}")
    
    if optim == "alltime":
        costfun_init = cost_functional_proj(u, Z, c, d, s, uhatvec, 
                               num_steps, dt, M, c_lower, c_upper, beta)
    elif optim == "finaltime":
        costfun_init = cost_functional_proj_FT(u, Z, c, d, s, uhatvec, z, 
                                    num_steps, dt, M, c_lower, c_upper, beta)
        
    armijo = 10**5 # initialise the difference in cost function norm decrease
    # note the negative sign in the condition comes from the descent direction
    while armijo > - gam / s * grad_costfun_L2 and k < max_iter:
        s = s0*( 1/2 ** k)
        # Calculate the incremented c using the new step size
        c_inc = np.clip(c + s * d, c_lower, c_upper)
        print(f"{k =}")
        ########## calculate new m,f corresponding to c_inc ###################
        print("Solving state equations...")
        t=0
        # initialise u and keep ICs
        u[nodes:] = np.zeros(num_steps * nodes)
        for i in range(1,num_steps+1):    # solve for f(t_{n+1}), m(t_{n+1})
            start = i * nodes
            end = (i + 1) * nodes
            t += dt
            if i % 20 == 0:
                print("t =", round(t, 4))
            
            u_n = u[start - nodes : start] # uk(t_n)
            c_inc_fun = vec_to_function(c_inc[start : end], V)

            u_rhs = np.zeros(nodes)
            
            Adrift1 = assemble_sparse(dot(drift, grad(c_inc_fun)) * w * v * dx) # pseudo-mass matrix
            Adrift2 = assemble_sparse(dot(drift, grad(v)) * c_inc_fun * w * dx) # pseudo-stiffness matrix
            
            ## System matrix for the state equation
            A_u = - eps * Ad + Arot + Adrift1 + Adrift2
            
            u[start:end] = FCT_alg(A_u, u_rhs, u_n, dt, nodes, M, M_Lump, dof_neighbors)
            
            
        #######################################################################
        if optim == "alltime":
            cost2 = cost_functional_proj(u, Z, c_inc, d, s, uhatvec, 
                                    num_steps, dt, M, c_lower, c_upper, beta)
        elif optim == "finaltime":   
            cost2 = cost_functional_proj_FT(u, Z, c_inc, d, s, uhatvec, z, 
                                        num_steps, dt, M, c_lower, c_upper, beta)
            
        armijo = cost2 - costfun_init
        grad_costfun_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)

        k += 1
        
    print(f"Armijo exit at {k=} with {s=}")
    return s, u

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
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    Aa = assemble_sparse(dot(grad(f_fun), grad(v)) * u * dx)
    Ar = assemble_sparse(m_fun * u * v * dx)
    return - Dm * Ad + chi * Aa + Ar

def mat_chtx_p(f_fun, m_fun, Dm, chi, u, v):
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    Aa = assemble_sparse(dot(grad(f_fun), grad(v)) * u * dx)
    Adf = assemble_sparse(div(grad(f_fun)) * u * v * dx)
    Ar = assemble_sparse((4 - 2 * m_fun) * u * v * dx)
    return - Dm * Ad - chi * Aa - chi * Adf + Ar
    


def FCT_alg(A, rhs, u_n, dt, nodes, M, M_lumped, dof_neighbors, source_mat=None):
    """
    Applies the Flux Corrected Transport (FCT) algorithm with linearized fluxes
    using antidiffusive fluxes and a correction factor matrix calculated with 
    Zalesak algorithm.
    
    Parameters
    A : scipy.sparse.spmatrix The flux matrix (e.g., advection or diffusion).
    rhs : numpy.ndarray The right-hand side (source) vector (not multiplied by dt).
    u_n : numpy.ndarray The solution vector at the current time step.
    dt : float The time step size.
    nodes : int Total number of nodes in the mesh.
    M : scipy.sparse.lil_matrix The mass matrix.
    M_lumped : scipy.sparse.spmatrix The lumped mass matrix.
    dof_neighbors (list of lists): A list of lists where each entry contains the neighboring DOFs for each node.
    source_mat (scipy.sparse.spmatrix, optional): The decoupled matrix terms from the RHS. Defaults to None.
    
    Returns
    numpy.ndarray: The flux-corrected solution vector at the next time step.
    """
    D = artificial_diffusion_mat(A)
    M_diag = M.diagonal()
    M_lumped_diag = M_lumped.diagonal()
    Mat_u_Low = lil_matrix(D.shape)
    
    ## 1. Calculate low-order solution u^{n+1} using previous time step solution
    if source_mat is None:
        Mat_u_Low[:, :] = M_lumped[:, :] - dt * (A[:, :] + D[:, :])

    else:
        Mat_u_Low[:, :] = M_lumped[:, :] - dt * (A[:, :] + D[:, :] - source_mat[:, :])

    rhs_u_Low = M_lumped @ u_n + dt * rhs
    u_Low = spsolve(csr_matrix(Mat_u_Low), rhs_u_Low)
    
    ## 2. Calculate raw antidiffusive flux
    # approximate the derivative du/dt using Chebyshev semi-iterative method
    rhs_du_dt = np.squeeze(np.asarray(A @ u_Low + rhs)) # flatten to vector array
    du_dt = ChebSI(rhs_du_dt, M, M_diag, 20, 0.5, 2)

    # corrected flux calculation(only use neighbouring nodes):
    F = lil_matrix((nodes, nodes))  # Use lil_matrix for efficient modifications
    for i in range(nodes):
        for j in dof_neighbors[i]: # flux from node j to node i, j is a neighbour of i
            F[i,j] = M[i, j] * (du_dt[i] - du_dt[j]) + D[i, j] * (u_Low[i] - u_Low[j])
    F.setdiag(0)
    
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
    r_pos[p_pos != 0] = np.minimum(1, M_lumped_diag[p_pos != 0]*q_pos[ p_pos!= 0]
                                  / (dt * p_pos[p_pos != 0]))
    r_neg[p_neg != 0] = np.minimum(1, M_lumped_diag[p_neg != 0]*q_neg[ p_neg!= 0]
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
    u_np1 = u_Low + dt*Fbar/M_lumped_diag
    
    return u_np1