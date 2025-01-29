import dolfin as df
from dolfin import dx, dot, grad, div, assemble, exp
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, triu, tril, spdiags
from scipy.sparse.linalg import spsolve

# Contains functions used in all scripts

def reorder_vector_to_dof(vec, nodes, vertextodof):
    """
    Reorders a vector to match the degrees of freedom (DoF) ordering in FEniCS.

    Parameters:
    vec (numpy.ndarray): Input vector of size equal to the number of nodes.
    nodes (int): Number of nodes in the mesh.
    vertextodof (numpy.ndarray): Mapping from vertex indices to DoF indices.

    Returns:
    numpy.ndarray: Reordered vector with values corresponding to DoF ordering.
    """
    vec_dof = np.zeros(vec.shape)
    for i in range(nodes):
        j = int(vertextodof[i])
        vec_dof[j] = vec[i]
    return vec_dof

def reorder_vector_to_dof_time(vec, num_steps, nodes, vertextodof):
    """
    Reorders a time-dependent vector to match the DoF ordering in FEniCS for each time step.

    Parameters:
    vec (numpy.ndarray): Input vector of size (num_steps * nodes).
    num_steps (int): Number of time steps.
    nodes (int): Number of nodes in the mesh.
    vertextodof (numpy.ndarray): Mapping from vertex indices to DoF indices.

    Returns:
    numpy.ndarray: Reordered vector with values corresponding to DoF ordering for all time steps.
    """
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + j] = temp[i] 
    return vec_dof

def reorder_vector_from_dof_time(vec, num_steps, nodes, vertextodof):
    """
    Reorders a time-dependent vector from the DoF ordering back to the node ordering.

    Parameters:
    vec (numpy.ndarray): Input vector of size (num_steps * nodes) in DoF order.
    num_steps (int): Number of time steps.
    nodes (int): Number of nodes in the mesh.
    vertextodof (numpy.ndarray): Mapping from vertex indices to DoF indices.

    Returns:
    numpy.ndarray: Reordered vector with values corresponding to the node ordering for all time steps.
    """
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + i] = temp[j] 
    return vec_dof

def rel_err(new,old):
    """
    Calculates the relative error between two vectors.

    Parameters:
    new (numpy.ndarray): New vector.
    old (numpy.ndarray): Old vector.

    Returns:
    float: Relative error defined as ||new - old|| / ||old||.
    """
    return np.linalg.norm(new - old)/np.linalg.norm(old)

def assemble_sparse(a):
    """
    Assembles a bilinear form into a sparse matrix in CSR format.
 
    Parameters:
    a (dolfin.Form): Bilinear form to be assembled.
 
    Returns:
    scipy.sparse.csr_matrix: Assembled sparse matrix in CSR format.
    """
    A = df.assemble(a)
    mat = df.as_backend_type(A).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr

def assemble_sparse_lil(a):
    """
    Assembles a bilinear form into a sparse matrix in LIL format.

    Parameters:
    a (dolfin.Form): Bilinear form to be assembled.

    Returns:
    scipy.sparse.lil_matrix: Assembled sparse matrix in LIL format.
    """
    csr = assemble_sparse(a)
    return lil_matrix(csr)

def vec_to_function(vec, V):
    """
    Converts a numpy vector to a FEniCS Function.

    Parameters:
    vec (numpy.ndarray): Input vector containing the DoF values.
    V (dolfin.FunctionSpace): Function space to define the output Function.

    Returns:
    dolfin.Function: FEniCS Function with values set from the input vector.
    """
    out = df.Function(V)
    out.vector().set_local(vec)
    return out

def ChebSI(vec, M, Md, cheb_iter, lmin, lmax):
    """
    Performs Chebyshev semi-iteration to approximate the solution of a linear system
    that is of the form Mx = b where M is the mass matrix.
    
    Parameters:
    vec (numpy.ndarray): Right-hand side vector (b in Ax = b).
    M (scipy.sparse.spmatrix): System matrix (M in Mx = b).
    Md (numpy.ndarray): Diagonal preconditioner or approximation to M's diagonal.
    cheb_iter (int): Number of Chebyshev iterations.
    lmin (float): Minimum eigenvalue estimate of M.
    lmax (float): Maximum eigenvalue estimate of M.
    
    Returns:
    numpy.ndarray: Approximated solution vector after iterations.
    """
    ymid = np.zeros_like(vec)
    yold = np.zeros_like(ymid)
    omega = 0
    
    rho = (lmax - lmin) / (lmax + lmin)
    Md = (lmin + lmax) / 2 * Md
    
    for k in range(1, cheb_iter + 1):
        if k==2:
            omega = 1 / (1 - rho**2 / 2)
        else:
            omega = 1 / (1 - (omega * rho**2) / 4);
        r = vec - M*ymid # residual = b - Mx
        z = r / Md 
        ynew = omega * (z + ymid - yold) + yold
        yold = ymid
        ymid = ynew
    
    return ynew

def sparse_nonzero(H):
    """
    Extracts nonzero elements and their indices from a sparse matrix.

    Parameters:
    H (scipy.sparse.spmatrix): Input sparse matrix.

    Returns:
    numpy.ndarray: Array with rows [row_index, col_index, value, is_positive]
        where is_positive is True for positive values.
    """
    Hx = H.tocoo()
    out = np.transpose(np.array([Hx.row, Hx.col, Hx.data, Hx.data > 0]))
    return out

def artificial_diffusion_mat(mat):
    """
    Constructs an artificial diffusion matrix to stabilize numerical schemes.
    
    Parameters:
    mat (scipy.sparse.lil_matrix): Input flux matrix, taken from the RHS of the equation.
    
    Returns:
    scipy.sparse.lil_matrix: Artificial diffusion matrix that cancels the negative 
        off-diagonal coefficients of mat.
    """
    # Flip the sign of upper and lower part of mat, ignoring the diagonal
    neg_upper = triu(mat, k=1, format='lil')  
    neg_lower = tril(mat, k=-1, format='lil')
    neg_upper[:,:] = -neg_upper[:,:]
    neg_lower[:,:] = -neg_lower[:,:]

    # Only keep the entries that were originally negative
    D_upper = lil_matrix(mat.shape)
    D_lower = lil_matrix(mat.shape)

    D_upper[:,:] = neg_upper.maximum(0)[:,:]
    D_lower[:,:] = neg_lower.maximum(0)[:,:]

    # Form D by adding lower and upper parts together
    D = lil_matrix(mat.shape)
    D[:, :] = D_upper[:, :] + D_lower[:, :]

    # Ensure D is symmetric
    D[:,:] = D.maximum(D.transpose())[:,:]

    # Set the diagonal entries to negative row sums
    D.setdiag(-D.sum(axis=1)) 

    return D

def generate_boundary_nodes(nodes, vertextodof):
    """
    Identifies boundary nodes in a square domain.
    
    Parameters:
    nodes (int): Total number of nodes in the mesh.
    vertextodof (numpy.ndarray): Mapping from vertex indices to DoF indices.
    
    Returns:
    tuple: (boundary_nodes, boundary_nodes_dof)
        - boundary_nodes (numpy.ndarray): Indices of boundary nodes in vertex ordering.
        - boundary_nodes_dof (numpy.ndarray): Indices of boundary nodes in DoF ordering.
   """
   
    sqnodes = round(np.sqrt(nodes))
    boundary_nodes = [
        n for n in range(nodes)
        if n % sqnodes in [0, sqnodes - 1] or n < sqnodes or n >= nodes - sqnodes
    ]
    boundary_nodes = np.array(boundary_nodes)
    boundary_nodes_dof = [int(vertextodof[n]) for n in boundary_nodes]
        
    return boundary_nodes, boundary_nodes_dof

def find_node_neighbours(mesh, nodes, vertextodof):
    """
    Finds neighbors for each node in the mesh.

    Parameters:
    mesh (dolfin.Mesh): Computational mesh.
    nodes (int): Total number of nodes in the mesh.
    vertextodof (numpy.ndarray): Mapping from vertex indices to DoF indices.

    Returns:
    list of lists: Each entry contains DoF indices of neighbors for a node.
    """
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

def row_lump(mat, nodes):
    """
    Applies row lumping to a matrix by summing its rows.
 
    Parameters:
    mass_mat (scipy.sparse.spmatrix): Input matrix.
    nodes (int): Total number of nodes in the mesh.
 
    Returns:
    scipy.sparse.dia_matrix: Row-lumped diagonal matrix.
    """
    return spdiags(data=np.transpose(mat.sum(axis=1)),diags=0, m=nodes, n=nodes)

def L2_norm_sq_Q(phi, num_steps, dt, M):
    '''
    Calculates the squared norm in L^2-space of given vector phi using FEM in 
    space and trapezoidal rule in time:
    \|phi\|^2_{L^2(Q)} = int_Q phi^2 dx dy dt = sum_k (dt_k/2)*phi_k^T*M*phi_k.

    
    Parameters:
    phi (numpy.ndarray): The vector of the form phi = (phi0, phi1, ..., phi_NT),
                         where each phi_i corresponds to a spatial discretization.
    num_steps (int): Number of time steps.
    dt (float): Time step size.
    M (numpy.ndarray): Mass matrix

    Returns:
    float: Squared L^2 norm of phi in spatiotemporal domain Q.
    '''
    phi_slices = np.split(phi,num_steps+1)
    trapez_coefs = np.ones(num_steps+1)
    trapez_coefs[0] = 0.5
    trapez_coefs[-1] = 0.5
    
    norm_sq = sum([trapez_coefs[i]*phi_slices[i].transpose() @ M @ phi_slices[i] for i in range(num_steps+1)]) *dt
    return norm_sq

def L2_norm_sq_Omega(phi, M):
    '''
    Calculates the squared norm in L^2-space of given vector phi for one time 
    step using FEM in space.
    \|phi\|^2_{L^2(Q)} = phi^T M phi.
    
    Parameters:
    phi (numpy.ndarray): The vector phi discretized in space of length 'nodes'.
    M (numpy.ndarray): Mass matrix

    Returns:
    float: Squared L^2 norm of phi in spatial domain \Omega.
    '''
    norm_sq = phi.transpose() @ M @ phi
    return norm_sq

def cost_functional(var1, var1_target, projected_control, num_steps,
                    dt, M, beta, optim,
                    var2=None, var2_target=None):
    """   
    Evaluates the cost functional for given state variables, control, and targets.
    Assumes given state variables correspond to the projected control variable.

    Parameters:
    var1 (numpy.ndarray): Primary state variable (e.g., m).
    var1_target (numpy.ndarray): Target state for var1.
    projected_control (numpy.ndarray): Control variable projected onto admissible solutions.
    num_steps (int): Number of time steps.
    dt (float): Time step size.
    M (numpy.ndarray): Mass matrix for spatial discretization.
    beta (float): Regularization parameter for the control term.
    optim (str): Optimization type ('alltime' or 'finaltime').
    var2 (numpy.ndarray, optional): Secondary state variable (e.g., f).
    var2_target (numpy.ndarray, optional): Target state for var2.

    Returns:
    float: Value of the cost functional.
    """
    # Calculate data fidelity norms
    if optim == 'alltime':
        print('Calculating L^2(Q)-norm...')
        func = 0.5 * L2_norm_sq_Q(var1 - var1_target, num_steps, dt, M)
        if var2 is not None and var2_target is not None:
            func += 0.5 * L2_norm_sq_Q(var2 - var2_target, num_steps, dt, M)
            
    elif optim == 'finaltime':
        print('Calculating L^2(\Omega)-norm...')
        nodes = var1_target.shape[0]
        func = 0.5 * L2_norm_sq_Omega(var1[num_steps * nodes:] - var1_target, M)
        
        if var2 is not None and var2_target is not None:
            func += 0.5 * L2_norm_sq_Omega(var2[num_steps * nodes:] - var2_target, M)
            
    else:
        raise ValueError(f"The selected option {optim} is invalid.")
    
    # Add the regularization term for the control
    func += beta / 2 * L2_norm_sq_Q(projected_control, num_steps, dt, M)
    
    return func
    
def armijo_line_search(var1, c, d, var1_target, num_steps, dt, M, c_lower, 
                        c_upper, beta, costfun_init, nodes, optim, gam = 1e-4,
                        max_iter=40, s0=1, w1=None, w2=None, example=None, V=None,
                        dof_neighbors=None, var2=None, var2_target=None):
    '''
    Performs Projected Armijo Line Search and returns optimal step size.
    Descent direction adjusted for a source control problem e.g. for the 
    advection equation.
    var1: the variable that is affected by a change in control variable in its 
        equation
    c: control variable
    d: descent direction, i.e. negative of the gradient 
    var1_target, var2_target: target state for var1, var2
    num_steps, dt: number of time steps and the corresponding step size
    M: mass matrix
    c_lower, c_upper: constants determining the admissible set for the control
    beta: regularisation parameter
    costfun_init: previous evaluation of the cost functional
    gam: parameter in tolerance for decrease in the cost functional
    max_iter: max number of armijo iterations
    s0: step size at the first iteration.
    w1, w2: in case of linear equations, increment amount in var1 and var2
    example: {'Schnak','nonlinear','chtxs'} name of the  problem to solve in case of nonlinear equations
    V: FunctionSpace on some mesh in case of nonlinear equations
    optim = {'alltime', 'finaltime'}
    var2: the second variable for systems of equations, to calculate cost fun.
    
    control_dif_L2: Stationarity measure defined as the norm of the projected 
        gradient (Hinze, p. 107)
        
    Note the negative sign in the armijo condition comes from the descent direction
    '''
    k = 0   # counter
    s = s0  # initial step size
    n = var1_target.shape[0]
    
    if w1 is None and w2 is None:
        print('Assuming the equation is nonlinear, the increments will be calculated at each armijo iteration')
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
        M_Lump = row_lump(M, nodes)

    elif w1 is not None and w2 is None:
        print('The increment in var1 is given.')
    elif w1 is not None and w2 is not None:
        print('The increments in var1 and var2 are given.')
    else:
        raise ValueError(f"The selected combination of parameters {w1=} and {w2=} is invalid.")
     
    control_dif_L2 = 1
    grad_costfun_L2 = L2_norm_sq_Q(np.clip(c + s * d, c_lower, c_upper) - c, 
                                    num_steps, dt, M)
    
    armijo = 10**5 # initialise the difference in cost functional norm decrease
    
    
    # while armijo > -gam / s * control_dif_L2 and k < max_iter:
    while k < max_iter: # hard constraint: number of iterations
    
            # check if condition has been reached
            if armijo > -gam / s * control_dif_L2 or armijo > 0: # not reached
                print(f'{k=}')
                s = s0*( 1/2 ** k)
                # Calculate the incremented in c using the new step size
                c_inc = np.clip(c + s * d, c_lower, c_upper)
                
                if w1 is None and example == 'Schnak': # solve the state equations for new values
                    Du = 1/10
                    Dv = 8.6676
                    c_b = 0.9
                    gamma = 230.82
                    omega1 = 100
                    omega2 = 0.6
                    var1[nodes :] = np.zeros(num_steps * nodes)
                    var2[nodes :] = np.zeros(num_steps * nodes)
                    t = 0
                    wind = df.Expression(('-(x[1]-0.5)*sin(2*pi*t)','(x[0]-0.5)*sin(2*pi*t)'), degree=4, pi = np.pi, t = t)
                    print('Using time-dep. velocity field')
                    for i in range(1, num_steps + 1):    # solve for uk(t_{n+1}), vk(t_{n+1})
                        start = i * nodes
                        end = (i + 1) * nodes
                        t += dt
                        wind.t = t
    
                        if i % 50 == 0:
                            print('t = ', round(t, 4))
                        
                        var1_n = var1[start - nodes : start]
                        var2_n = var2[start - nodes : start]
                        
                        # Define previous time-step solution as a function
                        var1_n_fun = vec_to_function(var1_n, V)
                        var2_n_fun = vec_to_function(var2_n, V)
                        c_inc_fun = vec_to_function(c_inc[start : end], V)
                        
                        # # Solve for u using FCT (advection-dominated equation)
                        # A = assemble_sparse(dot(wind, grad(u)) * v * dx)
                        # mat_var1 = -(Du*Ad + omega1*A)
                        # rhs_var1 = np.asarray(assemble((gamma*(c_np1_fun + var1_n_fun**2 * var2_n_fun))* v * dx))
                        # var1[start : end] = FCT_alg(mat_var1, rhs_var1, var1_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)
    
                        # var1_np1_fun = vec_to_function(var1[start : end], V)
                        # M_u2 = assemble_sparse(var1_np1_fun * var1_np1_fun * u * v *dx)
                        
                        # # Solve for v using a direct solver
                        # rhs_var2 = np.asarray(assemble((gamma*c_b)* v * dx))
                        # var2[start : end] = spsolve(M + dt*(Dv*Ad + omega2*A + gamma*M_u2), M@var2_n + dt*rhs_var2) 
                     
                        ## v2
                        # Solve for u using FCT (advection-dominated equation)
                        A = assemble_sparse(dot(wind, grad(v)) * u * dx)
                        mat_var1 = -(Du*Ad - omega1*A)
                        rhs_var1 = np.asarray(assemble((gamma*(c_inc_fun + var1_n_fun**2 * var2_n_fun))* v * dx))
                        var1[start : end] = FCT_alg(mat_var1, rhs_var1, var1_n, dt, nodes, M, M_Lump, dof_neighbors, source_mat=gamma*M)
    
                        var1_np1_fun = vec_to_function(var1[start : end], V)
                        M_u2 = assemble_sparse(var1_np1_fun * var1_np1_fun * u * v *dx)
                        
                        # Solve for v using a direct solver
                        rhs_var2 = np.asarray(assemble((gamma*c_b)* v * dx))
                        var2[start : end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@var2_n + dt*rhs_var2) 
                     
                    cost2 = cost_functional(var1, var1_target, c_inc, num_steps, 
                                         dt, M, beta, optim=optim, 
                                         var2 = var2, var2_target = var2_target)
                
                elif w1 is None and example == 'nonlinear':
                    eps = 0.0001
                    speed = 1
                    k1 = 2
                    k2 = 2
                    wind = df.Expression(('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
                              'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'), degree=4, speed = speed)
                    source_fun_expr = df.Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)
                    A = assemble_sparse(dot(wind, grad(v))*u * dx)
                    mat_var1 = -eps * Ad + A
                    t=0
                    var1[nodes:] = np.zeros(num_steps * nodes) # initialise uk, keep IC
                    for i in range(1,num_steps + 1):    # solve for uk(t_{n+1})
                        start = i * nodes
                        end = (i + 1) * nodes
                        t += dt
                        if t%50==0:
                            print('t = ', round(t, 4))
                        
                        var1_n = var1[start - nodes : start]
                        var1_n_fun = vec_to_function(var1_n, V) 
                        c_inc_fun = vec_to_function(c_inc[start : end], V)

                        M_u2 = assemble_sparse(var1_n_fun * var1_n_fun * u * v *dx)

                        var1_rhs = np.asarray(assemble(c_inc_fun*v*dx))
                        
                        var1[start:end] = FCT_alg(mat_var1, var1_rhs, var1_n, dt, nodes, M, M_Lump, 
                                        dof_neighbors, source_mat = -M + 1/3*M_u2)
                        
                    cost2 = cost_functional(var1, var1_target, c_inc, num_steps, 
                                         dt, M, beta, optim=optim)

                elif w1 is not None and w2 is None: # increment in var1 is given as w1
                    var1_inc = var1 + s*w1 # assuming one variable only
                    cost2 = cost_functional(var1_inc, var1_target, c_inc, num_steps, 
                                            dt, M, beta, optim=optim)
                
                elif w1 is not None and w2 is not None:
                    var1_inc = var1 + s*w1
                    var2_inc = var2 + s*w2
                    cost2 = cost_functional(var1_inc, var1_target, c_inc, num_steps, 
                                        dt, M, beta, optim=optim, 
                                        var2 = var2_inc, var2_target = var2_target)
                else:
                    raise ValueError(f"The selected combination of parameters {w1=} and {example=} is invalid.")

                armijo = cost2 - costfun_init
                control_dif_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)
                
                k += 1
                print(f'{control_dif_L2=}')
                print(f'{armijo=}')
                print(f'{cost2=}, {costfun_init=}')
                
            else: # reached
                break

    print(f'\nArmijo exit at {k=} with {s=}')

    if armijo < -gam / s * control_dif_L2:
        print('Converged: Armijo condition satisfied.')
    elif k >= max_iter:
        print(f'Stopped: Maximum iterations exceeded for {max_iter=}.')
    
    if var2 is None and example is None:
        return s, var1_inc
    elif example == 'nonlinear':
        return s, var1, k
    elif example == 'Schnak':
        return s, var1, var2
    elif example == 'chtxs':
        return s, var1, var2

def solve_schnak_system(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solver for the advective Schnakenberg system
        du/dt - Du*grad^2(u) + om1*w⋅grad(u) + gamma*(u-u^2v) = gamma*c  in Ωx[0,T]
        dv/dt - Dv*grad^2(v) + om2*w⋅grad(v) + gamma*(u^2v-b) = 0        in Ωx[0,T]
                                                du/dn = dv/dn = 0        on ∂Ωx[0,T]
                                                         u(0) = u0(x)    in Ω
                                                         v(0) = v0(x)    in Ω

    Parameters:
        c (np.ndarray): Control variable.
        var1 (np.ndarray): The state variable influenced by the control variable.
        var2 (np.ndarray): The second state variable.
        V (FunctionSpace, optional): Finite element function space.
        nodes (int): Number of nodes in the spatial discretization.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        dof_neighbors (list): Degrees of freedom neighbors for FCT.

    Returns:
        tuple: Solutions for the state variables var1 and var2.
    """
    Du = 1/10
    Dv = 8.6676
    c_b = 0.9
    gamma = 230.82
    omega1 = 100
    omega2 = 0.6
    wind = df.Expression(('-(x[1]-0.5)*sin(2*pi*t)','(x[0]-0.5)*sin(2*pi*t)'), degree=4, pi=np.pi, t=0)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M = assemble_sparse(u * v * dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    
    # Reset variables but keep initial conditions
    var1[nodes :] = np.zeros(num_steps * nodes)
    var2[nodes :] = np.zeros(num_steps * nodes)
    t = 0
    print('Solving the system of advective Schnakenberg state equations...')
    for i in range(1, num_steps + 1):   # Solve for var1(t_{n+1}), var2(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        wind.t = t

        if i % 50 == 0:
            print('t = ', round(t, 4))
        
        var1_n = var1[start - nodes : start]
        var2_n = var2[start - nodes : start]
        
        # Define previous time-step solution as a function
        var1_n_fun = vec_to_function(var1_n, V)
        var2_n_fun = vec_to_function(var2_n, V)
        control_fun = vec_to_function(control[start : end], V)
        
        # Solve for u using FCT (advection-dominated equation)
        A = assemble_sparse(dot(wind, grad(v)) * u * dx)
        mat_var1 = -(Du*Ad - omega1*A)
        rhs_var1 = np.asarray(assemble((gamma*(control_fun + var1_n_fun**2 * var2_n_fun))* v * dx))
        var1[start : end] = FCT_alg(mat_var1, rhs_var1, var1_n, dt, nodes, M, M_lumped, dof_neighbors, source_mat=gamma*M)

        var1_np1_fun = vec_to_function(var1[start : end], V)
        M_u2 = assemble_sparse(var1_np1_fun * var1_np1_fun * u * v *dx)
        
        # Solve for v using a direct solver
        rhs_var2 = np.asarray(assemble((gamma*c_b)* v * dx))
        var2[start : end] = spsolve(M + dt*(Dv*Ad - omega2*A + gamma*M_u2), M@var2_n + dt*rhs_var2) 
     
    return var1, var2

def nonlinear_equation_IC(a1, a2, deltax, nodes, vertextodof):
    """
    Computes the initial condition for a nonlinear equation on a 2D square mesh grid.

    Parameters:
        a1 (float): Left endpoint of the spatial domain [a1,a2]x[a1,a2].
        a2 (float): Right endpoint of the spatial domain [a1,a2]x[a1,a2].
        deltax (float): Grid spacing in both X and Y directions.

    Returns:
        np.ndarray: 2D array representing the initial condition over the mesh grid.
    """
    X = np.arange(a1, a2 + deltax, deltax)
    Y = np.arange(a1, a2 + deltax, deltax)
    X, Y = np.meshgrid(X,Y)
    
    kk = 4
    init_cond = 5*Y*(Y-1)*X*(X-1)*np.sin(kk*X*np.pi)
    init_cond_dof = reorder_vector_to_dof_time(init_cond.reshape(nodes), 1, nodes, vertextodof)

    return init_cond_dof

def get_nonlinear_eqns_params():
    """
    Returns the shared parameters for the nonlinear state and adjoint equations:
        du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c   in Ωx[0,T]
        dp/dt + div(-eps*grad(p) + w*p) + u^2*p - p = 0     in Ω × [0,T]
    """
    eps = 0.0001
    speed = 1
    k1 = 2
    k2 = 2
    wind = df.Expression(
        ('speed*2*(x[1]-0.5)*x[0]*(1-x[0])',
         'speed*2*(x[0]-0.5)*x[1]*(1-x[1])'),
        degree=4, speed=speed
    )
    return eps, speed, k1, k2, wind

def solve_nonlinear_equation(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solver for the nonlinear equation:
        du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c      in Ωx[0,T]
                                                du/dn = 0      on ∂Ωx[0,T]
                                                 u(0) = u0(x)  in Ω
    where div (w) = 0  in Ωx[0,T]
          w⋅n = 0      on ∂Ωx[0,T]                                             
          
    Parameters:
        c (np.ndarray): Control variable.
        var1 (np.ndarray): The state variable influenced by the control variable.
        var2 (np.ndarray): Placeholder for a second state variable.
        V (FunctionSpace, optional): Finite element function space.
        nodes (int): Number of nodes in the spatial discretization.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        dof_neighbors (list): Degrees of freedom neighbors for FCT.

    Returns:
        tuple: Solution for the state variable var1 and None for compatibility 
            with usage in armijo_line_search that requires a tuple.
    """
    
    eps, speed, k1, k2, wind = get_nonlinear_eqns_params()
    # source_fun_expr = df.Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M_lil = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M_lil, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(v)) * dx)
    A = assemble_sparse_lil(dot(wind, grad(v))*u * dx)
    Mat_var1 = lil_matrix(A.shape)
    Mat_var1[:,] = A[:,:] - eps * Ad[:,:]
    
    # Reset variable but keep initial conditions
    var1[nodes:] = np.zeros(num_steps * nodes)
    t = 0
    print('Solving nonlinear state equation...')
    for i in range(1,num_steps + 1):           # Solve for var1(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 20 == 0:
            print('t = ', round(t, 4))

        var1_n = var1[start - nodes : start]
        var1_n_fun = vec_to_function(var1_n, V) 
        control_fun = vec_to_function(control[start : end], V)

        M_u2 = assemble_sparse_lil(var1_n_fun**2 * u * v *dx)
        Mat_rhs = lil_matrix(A.shape)
        Mat_rhs[:,:] = -M_lil[:,:] + 1/3*M_u2[:,:]
        var1_rhs = np.asarray(assemble(control_fun * v *dx))
        var1[start:end] = FCT_alg(Mat_var1, var1_rhs, var1_n, dt, nodes, M_lil, M_lumped, 
                        dof_neighbors, source_mat = Mat_rhs)
    print(var1)
    return var1, None

def solve_adjoint_nonlinear_equation(uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solves the adjoint equation:
        dp/dt + div(-eps*grad(p) + w*p) + u^2*p - p =  0   in Ω × [0,T]
                                               dp/dn = 0         on ∂Ω × [0,T]
                                                p(T) = uhat - u         in Ω
    corresponding to the nonlinear state equation:
        du/dt + div(-eps*grad(u) + w*u) - u + (1/3) * u^3 = c    in Ω × [0,T]
                                                   du/dn = 0    on ∂Ω × [0,T]
                                                    u(0) = u0(x) in Ω
    where the velocity field satisfies:
        div(w) = 0   in Ω × [0,T]
           w⋅n = 0    on ∂Ω × [0,T]

    Parameters:
        uk (np.ndarray): The state variable influenced by the control variable.
        uhat_T (np.ndarray): The desired target state at the final time T.
        pk (np.ndarray): The adjoint variable, initialized at final time T.
        T (float): Final time of the simulation.
        V (FunctionSpace): Finite element function space.
        nodes (int): Number of spatial nodes in the discretization.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        dof_neighbors (list): Degrees of freedom neighbors for FCT.

    Returns:
        np.ndarray: The computed adjoint variable pk at all time steps.
    """
    
    eps, speed, k1, k2, wind = get_nonlinear_eqns_params()
    # source_fun_expr = df.Expression('sin(k1*pi*x[0])*sin(k2*pi*x[1])', degree=4, pi=np.pi, k1=k1, k2=k2)
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M_lil = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M_lil, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(v)) * dx)
    A = assemble_sparse_lil(dot(wind, grad(v))*u * dx)
    Mat_p = lil_matrix(A.shape)
    Mat_p[:,] = -A[:,:] - eps * Ad[:,:]

    # Reset variable but keep initial conditions
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    t = T
    print('Solving adjoint equation...')
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
            
        pk_np1 = pk[end : end + nodes]                  # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V)  # uk(t_n)
        M_u2 = assemble_sparse_lil(uk_n_fun**2 * u * v *dx)
        Mat_rhs = lil_matrix(A.shape)
        Mat_rhs[:,:] = M_u2[:,:] - M_lil[:,:] 
        p_rhs = np.zeros(nodes)
        pk[start:end] = FCT_alg(Mat_p, p_rhs, pk_np1, dt, nodes, M_lil, M_lumped, 
                                dof_neighbors, source_mat = Mat_rhs)
    return pk

def solve_chtxs_system(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solver for the chemotaxis system
    du/dt - Du*grad^2(u) + om1*w⋅grad(u) + gamma*(u-u^2v) = gamma*c  in Ω
    dv/dt - Dv*grad^2(v) + om2*w⋅grad(v) + gamma*(u^2v-b) = 0        in Ω
    dm/dt - Dm*grad^2(m) + div(chi*m*grad(f)) = RHS       in Ωx[0,T]
    df/dt - Df*grad^2(f)) + delta*f = c*m          in Ωx[0,T]
                                            du/dn = dv/dn = 0        on ∂Ωx[0,T]
                                                     u(0) = u0(x)    in Ω
                                                     v(0) = v0(x)    in Ω

    Parameters:
        c (np.ndarray): Control variable.
        var1 (np.ndarray): The state variable u.
        var2 (np.ndarray): The second state variable, v, influenced by the control variable.
        V (FunctionSpace, optional): Finite element function space.
        nodes (int): Number of nodes in the spatial discretization.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        dof_neighbors (list): Degrees of freedom neighbors for FCT.

    Returns:
        tuple: Solutions for the state variables var1 and var2.
    """
    delta = 100
    Dm = 0.05
    Df = 0.05
    chi = 0.25
    gamma = 100
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M = assemble_sparse(u*v*dx)
    M_lil = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    Mat_var2 = M + dt * (Df * Ad + delta * M)

    # Reset variables but keep initial conditions
    var1[nodes :] = np.zeros(num_steps * nodes)
    var2[nodes :] = np.zeros(num_steps * nodes)
    t = 0
    print('Solving the system of chemotaxis state equations...')
    for i in range(1, num_steps + 1):  # Solve for var2(t_{n+1}), var1(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 50 == 0:
            print('t = ', round(t, 4))
                
        var1_n = var1[start - nodes : start]
        var1_n_fun = vec_to_function(var1_n, V)
        var2_n_fun = vec_to_function(var2[start - nodes : start], V)
        control_fun = vec_to_function(control[start : end], V)
           
        var2_rhs = np.asarray(assemble(var2_n_fun * v * dx  + dt * control_fun * var1_n_fun * v * dx))
    
        var2[start : end] = spsolve(Mat_var2, var2_rhs)
            
        var2_np1_fun = vec_to_function(var2[start : end], V)
    
        beta = 0.5
        Aa = assemble_sparse(exp(-beta*var1_n_fun)*dot(grad(var2_np1_fun), grad(v)) * u * dx)
        A_var1 = - Dm * Ad + chi*Aa
        var1_rhs = np.zeros(nodes)
    
        var1[start : end] =  FCT_alg(A_var1, var1_rhs, var1_n, dt, nodes, M_lil, M_lumped, dof_neighbors)
    
    return var1, var2
    
### refactored armijo code
def armijo_line_search_ref(var1, c, d, var1_target, num_steps, dt, c_lower, 
                       c_upper, beta, costfun_init, nodes, optim, V, gam=1e-4, 
                       max_iter=10, s0=1, nonlinear_solver=None, dof_neighbors=None, 
                       var2=None, var2_target=None, w1=None, w2=None):
    """
    Performs Projected Armijo Line Search to find the optimal step size for a source control problem.

    Parameters:
        var1 (np.ndarray): The state variable influenced by the control variable.
        c (np.ndarray): Control variable.
        d (np.ndarray): Descent direction (negative gradient).
        var1_target (np.ndarray): Target state for var1.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        c_lower (float): Lower bound for the control variable.
        c_upper (float): Upper bound for the control variable.
        beta (float): Regularization parameter.
        costfun_init (float): Initial cost function value.
        nodes (int): Number of nodes in the spatial discretization.
        optim (str): Optimization type ('alltime' or 'finaltime').
        V (FunctionSpace, optional): Finite element function space.
        
        Optional:
        gam (float): Armijo condition parameter (default: 1e-4).
        max_iter (int): Maximum number of iterations (default: 10).
        s0 (float): Initial step size (default: 1).
        nonlinear_solver (callable, optional): Function to solve a nonlinear 
            problem (state equation(s)).
            Parameters: (control, var1, var2, V, nodes, num_steps, dt, dof_neighbors)
            Returns: var1, var2
        dof_neighbors (list, optional): Degrees of freedom neighbors for FCT.
        var2 (np.ndarray, optional): Second state variable in coupled systems.
        var2_target (np.ndarray, optional): Target state for var2 in coupled systems.
        w1 (np.ndarray, optional): Increment for var1 in linear problems.
        w2 (np.ndarray, optional): Increment for var2 in linear problems.

    Returns:
        tuple: Optimal step size and updated states (var1, var2 if applicable).
    
    Notes:
    - control_dif_L2: Stationarity measure defined as the norm of the projected 
        gradient (Hinze, p. 107)
    - the negative sign in the armijo condition comes from the descent direction
    """
    def initialize_problem():
        if w1 is None and w2 is None:
            u = df.TrialFunction(V)
            v = df.TestFunction(V)
            M = assemble_sparse(u*v*dx)
            return u, v, M
        return None, None, None

    def compute_cost(var1, var2, c):
        return cost_functional(var1, var1_target, c, num_steps, dt, M, beta, 
                               optim=optim, var2=var2, var2_target=var2_target)

    def update_control(c, s, d):
        return np.clip(c + s * d, c_lower, c_upper)

    u, v, M = initialize_problem()

    s = s0
    control_dif_L2 = 1
    
    # Initialise the difference in cost functional norm decrease
    armijo = float('inf') 

    for k in range(max_iter): 
        c_inc = update_control(c, s, d)
        if w1 is None:
            var1, var2 = nonlinear_solver(c_inc, var1, var2, V, nodes, num_steps, dt, dof_neighbors)
            cost2 = compute_cost(var1, var2, c_inc)
        else:
            # Assumes a linear problem and state increments given
            var1_inc = var1 + s * w1
            if w2 is not None:
                var2_inc = var2 + s * w2
            else:
                var2_inc = None
            cost2 = compute_cost(var1_inc, var2_inc, c_inc)
            
        armijo = cost2 - costfun_init
        control_dif_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)
        
        print(f'Updated cost={cost2}, Orig. cost={costfun_init}')
        print(f'Cost difference={armijo}')
        print(f'Threshold value: {-gam/s*control_dif_L2=}')
        
        # Check Armijo stopping condition
        if armijo <= -gam / s * control_dif_L2:
            print('Converged: Armijo condition satisfied.')
            break

        s /= 2

    if armijo > -gam / s * control_dif_L2:
        print(f"Stopped: Maximum iterations ({max_iter}) exceeded.")

    return (s, var1, var2, c_inc) if var2 is not None else (var1, c_inc, k)


# def armijo_line_search_sbr_drift(u, p, c, d, uhatvec, eps, drift, num_steps, dt, nodes, M, M_Lump, Ad, Arot,
#                                  c_lower, c_upper, beta, V, dof_neighbors, gam = 10**-4, max_iter = 5, s0 = 1,
#                                  optim = 'alltime'):
        
#     '''
#     Performs Projected Armijo Line Search and returns optimal step size.
#     gam: parameter in tolerance for decrease in cost functional value
#     max_iter: max number of armijo iterations
#     s0: step size at the first iteration.
#     m,f: state variables
#     q: adjoint variable related to the control by the gradient equation
#     mhatvec, fhatvec: desired states for m, f at final time T
#     '''
    

#     k = 0 # counter
#     s = 1 # initial step size
#     n = uhatvec.shape[0]
#     Z = np.zeros(u.shape)
#     z = np.zeros(n)
    
#     w = df.TrialFunction(V)
#     v = df.TestFunction(V)
    
#     # Stationarity measure: Norm of the projected gradient (Hinze, p. 107)
#     grad_costfun_L2 = L2_norm_sq_Q(np.clip(c + s * d, c_lower, c_upper) - c,
#                                  num_steps, dt, M)
#     print(f'{grad_costfun_L2=}')
    
#     if optim == 'alltime':
#         costfun_init = cost_functional_proj(u, Z, c, d, s, uhatvec, 
#                                num_steps, dt, M, c_lower, c_upper, beta)
#     elif optim == 'finaltime':
#         costfun_init = cost_functional_proj_FT(u, Z, c, d, s, uhatvec, z, 
#                                     num_steps, dt, M, c_lower, c_upper, beta)
        
#     armijo = 10**5 # initialise the difference in cost function norm decrease
#     # note the negative sign in the condition comes from the descent direction
#     while armijo > - gam / s * grad_costfun_L2 and k < max_iter:
#         s = s0*( 1/2 ** k)
#         # Calculate the incremented c using the new step size
#         c_inc = np.clip(c + s * d, c_lower, c_upper)
#         print(f'{k =}')
#         ########## calculate new m,f corresponding to c_inc ###################
#         print('Solving state equations...')
#         t=0
#         # initialise u and keep ICs
#         u[nodes:] = np.zeros(num_steps * nodes)
#         for i in range(1,num_steps+1):    # solve for f(t_{n+1}), m(t_{n+1})
#             start = i * nodes
#             end = (i + 1) * nodes
#             t += dt
#             if i % 20 == 0:
#                 print('t =', round(t, 4))
            
#             u_n = u[start - nodes : start] # uk(t_n)
#             c_inc_fun = vec_to_function(c_inc[start : end], V)

#             u_rhs = np.zeros(nodes)
            
#             Adrift1 = assemble_sparse(dot(drift, grad(c_inc_fun)) * w * v * dx) # pseudo-mass matrix
#             Adrift2 = assemble_sparse(dot(drift, grad(v)) * c_inc_fun * w * dx) # pseudo-stiffness matrix
            
#             ## System matrix for the state equation
#             A_u = - eps * Ad + Arot + Adrift1 + Adrift2
            
#             u[start:end] = FCT_alg(A_u, u_rhs, u_n, dt, nodes, M, M_Lump, dof_neighbors)
            
            
#         #######################################################################
#         if optim == 'alltime':
#             cost2 = cost_functional_proj(u, Z, c_inc, d, s, uhatvec, 
#                                     num_steps, dt, M, c_lower, c_upper, beta)
#         elif optim == 'finaltime':   
#             cost2 = cost_functional_proj_FT(u, Z, c_inc, d, s, uhatvec, z, 
#                                         num_steps, dt, M, c_lower, c_upper, beta)
            
#         armijo = cost2 - costfun_init
#         grad_costfun_L2 = L2_norm_sq_Q(c_inc - c, num_steps, dt, M)

#         k += 1
        
#     print(f'Armijo exit at {k=} with {s=}')
#     return s, u

# def rhs_chtx_m(m_fun, v):
#     return np.asarray(assemble(4*m_fun * v * dx))

# def rhs_chtx_f(f_fun, m_fun, c_fun, dt, v):
#     return np.asarray(assemble(f_fun * v * dx  + dt * m_fun * c_fun * v * dx))

# def rhs_chtx_p(c_fun, q_fun, v):
#     return np.asarray(assemble(c_fun * q_fun * v * dx))

# def rhs_chtx_q(q_fun, m_fun, p_fun, chi, dt, v):
#     return np.asarray(assemble(q_fun * v * dx + 
#                         dt * div(chi * m_fun * grad(p_fun)) * v * dx))
    
# def mat_chtx_m(f_fun, m_fun, Dm, chi, u, v):
#     Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
#     Aa = assemble_sparse(dot(grad(f_fun), grad(v)) * u * dx)
#     Ar = assemble_sparse(m_fun * u * v * dx)
#     return - Dm * Ad + chi * Aa + Ar

# def mat_chtx_p(f_fun, m_fun, Dm, chi, u, v):
#     Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
#     Aa = assemble_sparse(dot(grad(f_fun), grad(v)) * u * dx)
#     Adf = assemble_sparse(div(grad(f_fun)) * u * v * dx)
#     Ar = assemble_sparse((4 - 2 * m_fun) * u * v * dx)
#     return - Dm * Ad - chi * Aa - chi * Adf + Ar
    


def FCT_alg(A, rhs, u_n, dt, nodes, M, M_lumped, dof_neighbors, source_mat=None):
    """
    Applies the Flux Corrected Transport (FCT) algorithm with linearized fluxes
    using antidiffusive fluxes and a correction factor matrix calculated with 
    Zalesak algorithm.
    
    Parameters:
    A (scipy.sparse.spmatrix): The flux matrix (e.g., advection or diffusion).
    rhs (numpy.ndarray): The right-hand side (source) vector (not multiplied by dt).
    u_n (numpy.ndarray): The solution vector at the current time step.
    dt (float): The time step size.
    nodes (int): Total number of nodes in the mesh.
    M (scipy.sparse.lil_matrix): The mass matrix.
    M_lumped (scipy.sparse.spmatrix): The lumped mass matrix.
    dof_neighbors (list of lists): A list of lists where each entry contains the neighboring DOFs for each node.
    source_mat (scipy.sparse.spmatrix, optional): The decoupled matrix terms from the RHS. Defaults to None.
    
    Returns:
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
        Mat_u_Low[:, :] = D[:, :] - dt * (A[:, :] + D[:, :] - source_mat[:, :])

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

def import_data_final(file_path, file_name, nodes, vertextodof):
    '''
    Loads the target data: 
        hat{m} if input is "m", or hat{f} if input is "f"
    Size of the loaded array should be sqnodes x sqnodes.
    Output = numpy array sqnodes x sqnodes.
    '''
    sqnodes = round(np.sqrt(nodes))
    data = np.genfromtxt(file_path + file_name + '.csv', delimiter=',')
    
    data_re = reorder_vector_from_dof_time(data, 1, nodes, vertextodof)
    data_re = data_re.reshape((sqnodes,sqnodes))
    return data_re, data















