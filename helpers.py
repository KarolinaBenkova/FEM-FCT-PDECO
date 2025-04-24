import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, triu, tril, spdiags
from scipy.sparse.linalg import spsolve

import dolfin as df
from dolfin import dx, dot, grad, assemble, exp

def reorder_vector_to_dof(vec, num_steps, nodes, vertex_to_dof):
    """
    Reorders a time-dependent vector to match the DoF ordering in FEniCS for each time step.

    Parameters
    ----------
    vec : numpy.ndarray
        Input vector of size (num_steps * nodes).
    num_steps : int 
        Number of time steps.
    nodes : int
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    vec_dof : numpy.ndarray
        Reordered vector with values corresponding to DoF ordering for all time steps.
    """
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertex_to_dof[i])
            vec_dof[n*nodes + j] = temp[i] 
    return vec_dof

def reorder_vector_from_dof(vec_dof, num_steps, nodes, vertex_to_dof):
    """
    Reorders a time-dependent vector from the DoF ordering back to the node ordering.

    Parameters
    ----------
    vec_dof : numpy.ndarray 
        Input vector of size (num_steps * nodes) in DoF order.
    num_steps : int
        Number of time steps.
    nodes : int 
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray 
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    vec : numpy.ndarray
        Reordered vector with values corresponding to the node ordering for all time steps.
    """
    vec = np.zeros(vec_dof.shape)
    for n in range(num_steps):
        temp = vec_dof[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertex_to_dof[i])
            vec[n*nodes + i] = temp[j] 
    return vec

def rel_err(new,old):
    """
    Calculates the relative error between two vectors.

    Parameters
    ----------
    new : numpy.ndarray 
        New vector.
    old : numpy.ndarray 
        Old vector.

    Returns
    -------
    float : 
        Relative error defined as ||new - old|| / ||old||.
    """
    return np.linalg.norm(new - old)/np.linalg.norm(old)

def assemble_sparse(a):
    """
    Assembles a bilinear form into a sparse matrix in CSR format.
 
    Parameters
    ----------
    a : dolfin.Form
        Bilinear form to be assembled.
 
    Returns
    -------
    csr_mat: scipy.sparse.csr_matrix
        Assembled sparse matrix in CSR format.
    """
    A = df.assemble(a)
    mat = df.as_backend_type(A).mat()
    csr_mat = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr_mat

def assemble_sparse_lil(a):
    """
    Assembles a bilinear form into a sparse matrix in LIL format.

    Parameters
    ----------
    a : dolfin.Form
        Bilinear form to be assembled.

    Returns
    -------
    lil_mat : scipy.sparse.lil_matrix
        Assembled sparse matrix in LIL format.
    """
    lil_mat = lil_matrix(assemble_sparse(a))
    return lil_mat

def vec_to_function(vec, V):
    """
    Converts a numpy vector to a FEniCS Function.

    Parameters
    ----------
    vec : numpy.ndarray
        Input vector containing the DoF values.
    V : dolfin.FunctionSpace
        Function space to define the output Function.

    Returns
    -------
    out : dolfin.Function
        FEniCS Function with values set from the input vector.
    """
    out = df.Function(V)
    out.vector().set_local(vec)
    return out

def ChebSI(vec, M, Md, cheb_iter=20, lmin=0.5, lmax=2):
    """
    Performs Chebyshev semi-iteration to approximate the solution of a linear system
    that is of the form Mx = b where M is the mass matrix.
    
    Parameters
    ----------
    vec : numpy.ndarray
        Right-hand side vector (b in Ax = b).
    M : scipy.sparse.spmatrix 
        Mass matrix (M in Mx = b).
    Md : numpy.ndarray 
        Diagonal preconditioner or approximation to M's diagonal.
    cheb_iter : int, default 20
        Number of Chebyshev iterations.
    lmin : float, default 0.5
        Minimum eigenvalue estimate of M.
    lmax : float, default 2 
        Maximum eigenvalue estimate of M.
    
    Returns
    -------
    ynew : numpy.ndarray
        Approximated solution vector after iterations.
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

    Parameters
    ----------
    H : scipy.sparse.spmatrix 
        Input sparse matrix.

    Returns
    -------
    out : numpy.ndarray
        Array with rows [row_index, col_index, value, is_positive] 
        where is_positive is True for positive values.
    """
    Hx = H.tocoo()
    out = np.transpose(np.array([Hx.row, Hx.col, Hx.data, Hx.data > 0]))
    return out

def artificial_diffusion_mat(mat):
    """
    Constructs an artificial diffusion matrix to stabilize numerical schemes.
    
    Parameters
    ----------
    mat : scipy.sparse.lil_matrix 
        Input flux matrix.
    
    Returns
    -------
    D : scipy.sparse.lil_matrix
        Artificial diffusion matrix that cancels the negative off-diagonal coefficients of mat.
    """
    # Flip the sign of upper and lower part of mat, ignoring the diagonal
    neg_upper = triu(mat, k=1, format="lil")  
    neg_lower = tril(mat, k=-1, format="lil")
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

def generate_boundary_nodes(nodes, vertex_to_dof):
    """
    Identifies boundary nodes in a square domain.
    
    Parameters
    ----------
    nodes : int 
        Total number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.
    
    Returns
    -------
    (boundary_nodes, boundary_nodes_dof) : tuple (list, list)
        boundary_nodes: Indices of boundary nodes in vertex ordering.
        boundary_nodes_dof: Indices of boundary nodes in DoF ordering.
   """
   
    sqnodes = round(np.sqrt(nodes))
    boundary_nodes = [
        n for n in range(nodes)
        if n % sqnodes in [0, sqnodes - 1] or n < sqnodes or n >= nodes - sqnodes
    ]
    boundary_nodes_dof = [int(vertex_to_dof[n]) for n in boundary_nodes]
        
    return boundary_nodes, boundary_nodes_dof

def find_node_neighbours(mesh, nodes, vertex_to_dof):
    """
    Finds neighbors for each node in the mesh.

    Parameters
    ----------
    mesh : dolfin.Mesh
        Computational mesh.
    nodes : int 
        Total number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    dof_neighbors : list 
        List where entry "i" contains DoF indices of neighbors of node "i".
    """
    # Initialize an empty list to store neighbors for each node
    node_neighbors = [[] for _ in range(mesh.num_vertices())]
    for vx in df.vertices(mesh): # Loop over mesh vertices
        idx = vx.index()
        neighborhood = [df.Edge(mesh, i).entities(0) for i in vx.entities(1)]
        neighborhood = [node_index for sublist in neighborhood for node_index in sublist]

        # Remove own index from neighborhood as its included several times
        neighborhood = [node_index for node_index in neighborhood if node_index != idx]
        neighborhood.append(idx) # Add own index again
        # Add the neighborhood to the list for the current node
        node_neighbors[idx] = neighborhood

    # Convert node_neighbors to dof_neighbors
    dof_neighbors = [[] for _ in range(nodes)]
    for i in range(nodes):
        j = vertex_to_dof[i]
        dof_neighbors[j] = [vertex_to_dof[node] for node in node_neighbors[i]]
    return dof_neighbors

def row_lump(mat, nodes):
    """
    Applies row lumping to a matrix by summing its rows.
 
    Parameters
    ----------
    mat : scipy.sparse.lil_matrix 
        Input matrix.
    nodes : int 
        Number of nodes in the mesh.
 
    Returns
    -------
    lumped_matrix : scipy.sparse.lil_matrix
        Row-lumped diagonal matrix.
    """
    
    lumped_matrix = lil_matrix((nodes, nodes))
    lumped_matrix.setdiag(mat.sum(axis=1).A1)  
    return lumped_matrix
    
def L2_norm_sq_Q(phi, num_steps, dt, M):
    """
    Calculates the squared norm in L^2-space of given vector phi using FEM in 
    space and trapezoidal rule over Q = [0,T] x Omega.
    \|phi\|^2_{L^2(Q)} = int_Q phi^2 dx dy dt = sum_k (dt_k/2)*phi_k^T*M*phi_k.

    
    Parameters
    ----------
    phi : numpy.ndarray 
        The vector of the form phi = (phi0, phi1, ..., phi_NT) where each phi_i 
        corresponds to a spatial discretization.
    num_steps : int 
        Number of time steps.
    dt : float 
        Time step size.
    M : scipy.sparse.spmatrix
        Mass matrix.

    Returns
    -------
    norm_sq : float
        Squared L^2 norm of phi in spatiotemporal domain Q.
    """
    phi_slices = np.split(phi,num_steps+1)
    trapez_coefs = np.ones(num_steps+1)
    trapez_coefs[0] = 0.5
    trapez_coefs[-1] = 0.5
    
    norm_sq = sum([trapez_coefs[i]*phi_slices[i].transpose() @ M @ phi_slices[i] for i in range(num_steps+1)]) *dt
    return norm_sq

def L2_norm_sq_Omega(phi, M):
    """
    Calculates the squared norm in L^2-space of given vector phi for one time 
    step using FEM in space over the spatial domain Omega.
    \|phi\|^2_{L^2(Q)} = phi^T M phi.
    
    Parameters
    ----------
    phi : numpy.ndarray 
        The vector phi discretized in space of length "nodes".
    M : scipy.sparse.spmatrix
        Mass matrix.

    Returns
    -------
    norm_sq : float
        Squared L^2 norm of phi in spatial domain Omega.
    """
    norm_sq = phi.transpose() @ M @ phi
    return norm_sq

def cost_functional(var1, var1_target, projected_control, num_steps, dt, M, 
                    beta, optim, var2=None, var2_target=None):
    """   
    Evaluates the cost functional for given state variables, control, and targets.
    Assumes given state variables correspond to the projected control variable.

    Parameters
    ----------
    var1 : numpy.ndarray 
        State variable.
    var1_target : numpy.ndarray 
        Target state for var1.
    projected_control : numpy.ndarray 
        Control variable projected onto admissible solutions.
    num_steps : int 
        Number of time steps.
    dt : float 
        Time step size.
    M : scipy.sparse.spmatrix
        Mass matrix.
    beta : float 
        Regularization parameter for the control term.
    optim : str {"alltime","finaltime"}
        Optimization type.
    var2 : numpy.ndarray [optional]
        Secondary state variable.
    var2_target : numpy.ndarray [optional]
        Target state for var2.

    Returns
    -------
    func : float
        Value of the cost functional.
    """
    valid_options = ["alltime", "finaltime"]
    if optim not in valid_options:
        raise ValueError(f"Invalid value for 'optim': '{optim}'. Must be one of {valid_options}.")

    # Calculate data fidelity norms
    if optim == "alltime":
        print("Calculating L^2(Q)-norm...")
        func = 0.5 * L2_norm_sq_Q(var1 - var1_target, num_steps, dt, M)
        if var2 is not None and var2_target is not None:
            func += 0.5 * L2_norm_sq_Q(var2 - var2_target, num_steps, dt, M)
            
    elif optim == "finaltime":
        print("Calculating L^2(\Omega)-norm...")
        nodes = var1_target.shape[0]
        func = 0.5 * L2_norm_sq_Omega(var1[num_steps * nodes:] - var1_target, M)
        
        if var2 is not None and var2_target is not None:
            func += 0.5 * L2_norm_sq_Omega(var2[num_steps * nodes:] - var2_target, M)
            
    else:
        raise ValueError(f"The selected option {optim} is invalid.")
    
    # Add the regularization term for the control
    func += beta / 2 * L2_norm_sq_Q(projected_control, num_steps, dt, M)
    return func
   
def schnak_sys_IC(a1, a2, deltax, nodes, vertex_to_dof):
    """
    Computes the initial condition for the advective Schnakenberg system on a 2D square mesh grid.

    Parameters
    ----------
    a1 : float 
        Left endpoint of the spatial domain [a1,a2]x[a1,a2].
    a2 : float 
        Right endpoint of the spatial domain [a1,a2]x[a1,a2].
    deltax : float 
        Grid spacing in both X and Y directions.
    nodes : int
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray 
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    u_init_dof, v_init_dof : tuple (numpy.ndarray, numpy.ndarray)
        Tuple of flattened 2D arrays representing the initial condition of each
        variable over the mesh grid.
    """
    X = np.arange(a1, a2 + deltax, deltax)
    Y = np.arange(a1, a2 + deltax, deltax)
    X, Y = np.meshgrid(X,Y)
    
    _, _, c_a, c_b, _, _, _, _ = get_schnak_sys_params()

    con = 0.1
    u_init = c_a + c_b + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))

    v_init = c_b / pow(c_a + c_b, 2) + con * np.cos(2 * np.pi * (X + Y)) + \
    0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))                  
    
    u_init_dof = reorder_vector_to_dof(
        u_init.reshape(nodes), 1, nodes, vertex_to_dof)
    v_init_dof = reorder_vector_to_dof(
        v_init.reshape(nodes), 1, nodes, vertex_to_dof)
    return u_init_dof, v_init_dof

def get_schnak_sys_params():
    """
    Returns the 7 shared parameters (floats, dolfin.Expression) for the advective 
    Schnakenberg state and adjoint equations:
    du/dt + div(-Du*grad(u) + ω1*w*u) + γ*(u-u^2v) = γ*c     in Ω × [0,T]
    dv/dt + div(-Dv*grad(v) + ω2*w*v) + γ*(u^2v-b) = 0       in Ω × [0,T]
    -dp/dt + div(-Du*grad(p) - ω1*w*p) + γ*p + 2*γ*u*v*(q-p) = 0  in Ωx[0,T]
    -dq/dt + div(-Dv*grad(q) - ω2*w*q) + γ*u^2*(q-p) = 0          in Ωx[0,T]
    assuming 
     div(w) = 0  in Ω × [0,T]
    Note: assumes equation for v is diffusion-dominated, i.e. Dv >> ω2
    """
    # Setup used in Garzon-Alvarado et al (2011)
    Du = 1/100      # Diffusion coefficient for u
    Dv = 8.6676     # Diffusion coefficient for v
    c_a = 0.1       # Constant "a" (control)
    c_b = 0.9       # Constant "b"
    gamma = 230.82  # Reaction term parameter
    omega1 = 100    # Wind strength for u
    omega2 = 0.6    # Wind strength for v
    # Stationary velocity field
    wind = df.Expression(("1 * (x[1] - 0.5) * x[0] * (1 - x[0])",
                         "-1 * (x[0] - 0.5) * x[1] * (1 - x[1])"),
                         degree=4, t=0)
    return Du, Dv, c_a, c_b, gamma, omega1, omega2, wind

def solve_schnak_system(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors,
                        control_fun=None, rescaling=1):
    """
    Solver for the advective Schnakenberg system
    du/dt + div(-Du*grad(u) + ω1*w*u) + γ*(u-u^2v) = γ*(c/r)     in Ω × [0,T]
    dv/dt + div(-Dv*grad(v) + ω2*w*v) + γ*(u^2v-b) = 0       in Ω × [0,T]
                         (-Du*grad(u) + ω1*w*u) ⋅ n = 0     on ∂Ω × [0,T]
                         (-Dv*grad(v) + ω2*w*v) ⋅ n = 0     on ∂Ω × [0,T]
                                                   u(0) = u0(x)   in Ω
                                                   v(0) = v0(x)   in Ω

    Parameters
    ----------
    control : numpy.ndarray 
        Control variable.
    var1 : numpy.ndarray 
        The state variable influenced by the control variable.
    var2 : numpy.ndarray
        The second state variable.
    V : FunctionSpace
        Finite element function space.
    nodes : int 
        Number of nodes in the mesh.
    num_steps : int 
        Number of time steps.
    dt : float 
        Time step size.
    dof_neighbors : list 
        List where entry "i" contains DoF indices of neighbors of node "i".
    control_fun : dolfin.Expression [optional]
         Control as a given expression, used to generate target data.
    rescaling : float, default 1
        Parameter to rescale the control parameter.

    Returns
    -------
    var1, var2 : tuple
        Solutions for the state variables var1 and var2.
    """
    Du, Dv, _, c_b, gamma, omega1, omega2, wind = get_schnak_sys_params()
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M = assemble_sparse_lil(u * v * dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(v)) * dx)
    
    # Reset variables but keep initial conditions
    var1[nodes :] = np.zeros(num_steps * nodes)
    var2[nodes :] = np.zeros(num_steps * nodes)
    t = 0
    print("Solving the system of advective Schnakenberg state equations...")
    for i in range(1, num_steps + 1):   # Solve for var1(t_{n+1}), var2(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        wind.t = t

        if i % 50 == 0:
            print("t = ", round(t, 4))
        
        var1_n = var1[start - nodes : start]
        var2_n = var2[start - nodes : start]
        
        # Define previous time-step solution as a function
        var1_n_fun = vec_to_function(var1_n, V)
        var2_n_fun = vec_to_function(var2_n, V)
        if control_fun is None:
            control_fun = vec_to_function(control[start : end], V)
        
        # Solve for u using FCT (advection-dominated equation)
        A = assemble_sparse_lil(dot(wind, grad(v)) * u * dx)
        Mat_var1 = lil_matrix(A.shape)
        Mat_var1[:,:] = Du*Ad[:,:] - omega1*A[:,:]
        rhs_var1 = np.asarray(assemble((
            gamma/rescaling*control_fun + gamma*(var1_n_fun**2 * var2_n_fun))* v * dx))

        var1[start : end] = FCT_alg_ref(Mat_var1, rhs_var1, var1_n, dt, nodes, M, 
                                    M_lumped, dof_neighbors, non_flux_mat=gamma*M)

        var1_np1_fun = vec_to_function(var1[start : end], V)
        M_u2 = assemble_sparse(var1_np1_fun **2 * u * v *dx)
        
        # Solve for v using a direct solver (assumes Dv >> ω2)
        rhs_var2 = np.asarray(assemble((gamma*c_b)* v * dx))
        Mat_var2 = M + dt*(Dv*Ad - omega2*A + gamma*M_u2)
        var2[start : end] = spsolve(Mat_var2, M@var2_n + dt*rhs_var2) 
    return var1, var2

def solve_adjoint_schnak_system(uk, vk, uhat_T, vhat_T, pk, qk, T, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solves the adjoint system equations corresponding to final-time optimization:
    -dp/dt + div(-Du*grad(p) - ω1*w*p) + γ*p + 2*γ*u*v*(q-p) = 0   in Ωx[0,T]
    -dq/dt + div(-Dv*grad(q) - ω2*w*q) + γ*u^2*(q-p) = 0           in Ωx[0,T]
                                       dp/dn = dq/dn = 0           on ∂Ωx[0,T]
                                                p(T) = û_T - u(T)  in Ω
                                                q(T) = v̂_T - v(T)  in Ω

    corresponding to the advective Schnakenberg system:
    du/dt + div(-Du*grad(u) + ω1*w*u) + γ(u-u^2v) = γ*c     in Ω × [0,T]
    dv/dt + div(-Dv*grad(v) + ω2*w*v) + γ(u^2v-b) = 0       in Ω × [0,T]
                         (-Du*grad(u) + ω1*w*u)⋅n = 0       on ∂Ω × [0,T]
                         (-Dv*grad(v) + ω2*w*v)⋅n = 0       on ∂Ω × [0,T]
                                             u(0) = u0(x)   in Ω
                                             v(0) = v0(x)   in Ω
    where the velocity field satisfies:
    div(w) = 0   in Ω × [0,T]

    Parameters
    ----------
    uk : numpy.ndarray 
        The state variable influenced by the control variable.
    vk : numpy.ndarray 
        The second state variable.
    uhat_T : numpy.ndarray 
        The desired target state for u at the final time T.
    vhat_T : numpy.ndarray 
        The desired target state for v at the final time T.
    pk : numpy.ndarray 
        The adjoint variable, initialized at final time T.
    qk : numpy.ndarray 
        Second adjoint variable, initialized at final time T.
    T : float 
        Final time of the simulation.
    V : FunctionSpace 
        Finite element function space.
    nodes : int 
        Number of nodes in the mesh.
    num_steps : int 
        Number of time steps.
    dt : float 
        Time step size.
    dof_neighbors : list 
        List where entry "i" contains DoF indices of neighbors of node "i".

    Returns
    -------
    pk, qk : tuple (numpy.ndarray, numpy.ndarray)
        The computed adjoint variables through space and over all time steps.
    """
    
    Du, Dv, _, _, gamma, omega1, omega2, wind = get_schnak_sys_params()
    
    u = df.TrialFunction(V)
    w = df.TestFunction(V)  
    M = assemble_sparse_lil(u*w*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(w)) * dx)

    # Set final-time conditions
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    qk[num_steps * nodes :] = vhat_T - vk[num_steps * nodes :]
    t = T
    print("\nSolving adjoint equation...")
    for i in reversed(range(0, num_steps)): # solve for pk(t_n), qk(t_n)
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print("t = ", round(t, 4))
            
        q_np1 = qk[end : end + nodes] # qk(t_{n+1})
        p_np1 = pk[end : end + nodes] # pk(t_{n+1})
    
        p_np1_fun = vec_to_function(p_np1, V) 
        u_n_fun = vec_to_function(uk[start : end], V)      # uk(t_n)
        v_n_fun = vec_to_function(vk[start : end], V)      # vk(t_n)
        
        wind.t = t
        
        # First solve for q, then for p
        A = assemble_sparse_lil(dot(wind,grad(u)) * w * dx)

        M_u2 = assemble_sparse_lil(u_n_fun **2 * u * w *dx)
        rhs_q = np.asarray(assemble(gamma * p_np1_fun * u_n_fun**2 * w * dx))
        Mat_q = M + dt*(Dv*Ad - omega2*A + gamma*M_u2)
        qk[start:end] = spsolve(Mat_q, M@q_np1 + dt*rhs_q) 
        
        q_n_fun = vec_to_function(qk[start:end], V)
        
        Mat_p = lil_matrix(A.shape)
        Mat_p[:,:]  = Du*Ad[:,:] - omega1*A[:,:] 
        M_uv = assemble_sparse_lil(u_n_fun * v_n_fun * u * w *dx)
        rhs_p = np.asarray(assemble(-2 * gamma * u_n_fun * v_n_fun * q_n_fun * w * dx))
        Mat_rhs = lil_matrix(A.shape)
        Mat_rhs[:,:] = gamma*M[:,:] - 2*gamma*M_uv[:,:]
        pk[start:end] = FCT_alg_ref(Mat_p, rhs_p, p_np1, dt, nodes, M, M_lumped, 
                                dof_neighbors, non_flux_mat=Mat_rhs)
    return pk, qk

def plot_two_var_solution(uk, vk, pk, qk, ck, uhat_re, vhat_re, T_data, it,
                          nodes, num_steps, dt, out_folder, vertex_to_dof, optim,
                          step_freq=20):
    """
    Produces plots of the computed state variables, adjoint variables, and control variable.
    Parameters
    ----------
    uk : numpy.ndarray 
        The first state variable in dof ordering.
    vk : numpy.ndarray
        The second state variable in dof ordering.
    pk : numpy.ndarray
        The first adjoint variable in dof ordering.
    qk : numpy.ndarray
        The second adjoint variable in dof ordering.
    ck : numpy.ndarray
        The control variable in dof ordering.
    uhat_re : numpy.ndarray
        The desired target state for u at the final time T in vertex ordering
    vhat_re : numpy.ndarray
        The desired target state for v at the final time T in vertex ordering.
    T_data : float
        Final time of the simulation.
    it : int
        Current iteration number.
    nodes : int
        Number of nodes in the mesh.
    num_steps : int
        Number of time steps.
    dt : float
        Time step size.
    out_folder : str
        Output folder for saving plots.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.
    optim : str ["alltime","finaltime"]
        Optimization type.
    step_freq : int, default 20
        Frequency of steps to plot.
    
    Returns
    -------
    None
        The function saves the plots to the specified folder and does not return any value.

    """
    valid_options = ["alltime", "finaltime"]
    if optim not in valid_options:
        raise ValueError(f"Invalid value for 'optim': '{optim}'. Must be one of {valid_options}.")
    
    sqnodes = round(np.sqrt(nodes))
    uk_re = reorder_vector_from_dof(uk, num_steps + 1, nodes, vertex_to_dof)
    vk_re = reorder_vector_from_dof(vk, num_steps + 1, nodes, vertex_to_dof)
    ck_re = reorder_vector_from_dof(ck, num_steps + 1, nodes, vertex_to_dof)
    pk_re = reorder_vector_from_dof(pk, num_steps + 1, nodes, vertex_to_dof)
    qk_re = reorder_vector_from_dof(qk, num_steps + 1, nodes, vertex_to_dof)
    
    for i in range(num_steps):
        startP = i * nodes
        endP = (i+1) * nodes
        tP = i * dt
        
        startU = (i+1) * nodes
        endU = (i+2) * nodes
        tU = (i+1) * dt
        
        u_re = uk_re[startU : endU].reshape((sqnodes,sqnodes))
        v_re = vk_re[startU : endU].reshape((sqnodes,sqnodes))
        c_re = ck_re[startP : endP].reshape((sqnodes,sqnodes))
        p_re = pk_re[startP : endP].reshape((sqnodes,sqnodes))
        q_re = qk_re[startP : endP].reshape((sqnodes,sqnodes))
        
        if optim == "alltime":
            uhat_t_re = uhat_re[startU : endU].reshape((sqnodes,sqnodes))
            vhat_t_re = vhat_re[startU : endU].reshape((sqnodes,sqnodes))
        else:
            uhat_t_re = uhat_re
            vhat_t_re = vhat_re
            
        if i % step_freq == 0 or i == num_steps-1:
            
            fig = plt.figure(figsize=(20, 10))

            ax = fig.add_subplot(2, 4, 1)
            im1 = ax.imshow(uhat_t_re)
            cb1 = fig.colorbar(im1, ax=ax)
            if optim == "alltime":
                ax.set_title(f"{it=}, Desired state for $u$ at t={round(tU, 5)}")
            else:
                ax.set_title(f"{it=}, Desired state for $u$ at t={T_data}")
        
            ax = fig.add_subplot(2, 4, 2)
            im2 = ax.imshow(u_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f"Computed state $u$ at t={round(tU, 5)}")
        
            ax = fig.add_subplot(2, 4, 3)
            im3 = ax.imshow(p_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f"Computed adjoint $p$ at t={round(tP, 5)}")
        
            ax = fig.add_subplot(2, 4, 4)
            im4 = ax.imshow(c_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f"Computed control $c$ at t={round(tP, 5)}")
            
            ax = fig.add_subplot(2, 4, 5)
            im5 = ax.imshow(vhat_t_re)
            cb5 = fig.colorbar(im5, ax=ax)
            if optim == "alltime":
                ax.set_title(f"{it=}, Desired state for $v$ at t={round(tU, 5)}")
            else:
                ax.set_title(f"{it=}, Desired state for $v$ at t={T_data}")
        
            ax = fig.add_subplot(2, 4, 6)
            im6 = ax.imshow(v_re)
            cb6 = fig.colorbar(im6, ax=ax)
            ax.set_title(f"Computed state $v$ at t={round(tU, 5)}")
        
            ax = fig.add_subplot(2, 4, 7)
            im7 = ax.imshow(q_re)
            cb7 = fig.colorbar(im7, ax=ax)
            ax.set_title(f"Computed adjoint $q$ at t={round(tP, 5)}")
        
            fig.tight_layout(pad=3.0)
            plt.savefig(out_folder + f"/it_{it}_plot_{i:03}.png")
        
            ax.clear()       # Clear axes
            for cb in [cb1, cb2, cb3, cb4, cb5, cb6, cb7]:
                cb.remove() # Remove colorbars
            del im1, im2, im3, im4, im5, im6, im7, cb1, cb2, cb3, cb4, cb5, cb6, cb7
            fig.clf()
            plt.close(fig) 
    return None
    
def nonlinear_equation_IC(a1, a2, deltax, nodes, vertex_to_dof):
    """
    Computes the initial condition for a nonlinear equation on a 2D square mesh grid.

    Parameters
    ----------
    a1 : float 
        Left endpoint of the spatial domain [a1,a2]x[a1,a2].
    a2 : float 
        Right endpoint of the spatial domain [a1,a2]x[a1,a2].
    deltax : float 
        Grid spacing in both X and Y directions.
    nodes : int
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray 
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    init_condition_dof : numpy.ndarray
        Flattened 2D array representing the initial condition over the mesh grid.
    """
    X = np.arange(a1, a2 + deltax, deltax)
    Y = np.arange(a1, a2 + deltax, deltax)
    X, Y = np.meshgrid(X,Y)
    
    kk = 4
    init_condition = 5*Y*(Y-1)*X*(X-1)*np.sin(kk*X*np.pi)
    init_condition_dof = reorder_vector_to_dof(
        init_condition.reshape(nodes), 1, nodes, vertex_to_dof)
    return init_condition_dof

def get_nonlinear_eqns_params():
    """
    Returns the 3 shared parameters (floats, dolfin.Expression) for the nonlinear state and adjoint equations:
        du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c   in Ωx[0,T]
        dp/dt + div(-eps*grad(p) + w*p) + u^2*p - p = 0     in Ω × [0,T]
    """
    eps = 1e-4  # Diffusion parameter
    speed = 1   # Wind speed, used to scale the velocity field
    # Velocity field
    wind = df.Expression(("speed * 2 * (x[1] - 0.5) * x[0] * (1 - x[0])",
                         "-speed * 2 * (x[0] - 0.5) * x[1] * (1 - x[1])"),
                         degree=4, speed=speed)
    return eps, speed, wind

def solve_nonlinear_equation(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors,
                             control_fun=None, show_plots=False, vertex_to_dof=None):
    """
    Solver for the nonlinear equation:
    du/dt + div(-eps*grad(u) + w*u) - u + 1/3*u^3 = c      in Ωx[0,T]
                                            du/dn = 0      on ∂Ωx[0,T]
                                             u(0) = u0(x)  in Ω
    where 
    div (w) = 0  in Ωx[0,T]
    w⋅n = 0      on ∂Ωx[0,T]                                             
          
    Parameters
    ----------
    control : numpy.ndarray 
        Control variable.
    var1 : numpy.ndarray 
        The state variable influenced by the control variable.
    var2 : numpy.ndarray 
        Placeholder for a second state variable, input needs to be None.
    V : FunctionSpace : 
        Finite element function space.
    nodes : int
        Number of nodes in the mesh.
    num_steps : int 
        Number of time steps.
    dt : float 
        Time step size.
    dof_neighbors : list 
        List where entry "i" contains DoF indices of neighbors of node "i".
    control_fun : dolfin.Expression [optional]
        Control as a given expression, used to generate target data.
    show_plots : bool, default False
        Choose whether to plot the solution.
    vertex_to_dof : numpy.ndarray [optional]
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    var1, None : tuple (numpy.ndarray, None)
        Solution for the state variable var1 and None for compatibility with 
        usage in armijo_line_search that requires a tuple.
    """
    if var2 is not None:
        warnings.warn("Warning: 'var2' is not None. Ensure this is intentional.")
        
    eps, _, wind = get_nonlinear_eqns_params()
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    sqnodes = round(np.sqrt(nodes))
    M = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(v)) * dx)
    A = assemble_sparse_lil(dot(wind, grad(v))*u * dx)
    Mat_var1 = lil_matrix(A.shape)
    Mat_var1[:,] = A[:,:] - eps * Ad[:,:]
    
    # Reset variable but keep initial conditions
    var1[nodes:] = np.zeros(num_steps * nodes)
    t = 0
    print("\nSolving nonlinear state equation...")
    for i in range(1,num_steps + 1):           # Solve for var1(t_{n+1})
        start = i * nodes
        end = (i + 1) * nodes
        t += dt
        if i % 20 == 0:
            print("t = ", round(t, 4))

        var1_n = var1[start - nodes : start]
        var1_n_fun = vec_to_function(var1_n, V) 
        if control_fun is None:
            control_fun = vec_to_function(control[start : end], V)
            
        M_u2 = assemble_sparse_lil(var1_n_fun**2 * u * v *dx)
        Mat_rhs = lil_matrix(A.shape)
        Mat_rhs[:,:] = -M[:,:] + 1/3*M_u2[:,:]
        var1_rhs = np.asarray(assemble(control_fun * v *dx))
        var1[start:end] = FCT_alg_ref(-Mat_var1, var1_rhs, var1_n, dt, nodes, M, M_lumped, 
                        dof_neighbors, non_flux_mat = Mat_rhs)
        
        if show_plots is True and i % 100 == 0:
            var1_re = reorder_vector_from_dof(var1[start:end],1, nodes, vertex_to_dof)
            plt.imshow(var1_re.reshape((sqnodes,sqnodes)))
            plt.colorbar()
            plt.title(f"Computed state $u$ at t = {round(t,5)}")
            plt.show()
    return var1, None

def solve_adjoint_nonlinear_equation(uk, uhat_T, pk, T, V, nodes, num_steps, dt, dof_neighbors):
    """
    Solves the adjoint equation corresponding to final-time optimization:
    dp/dt + div(-eps*grad(p) + w*p) + u^2*p - p =  0         in Ω × [0,T]
                                          dp/dn = 0          on ∂Ω × [0,T]
                                           p(T) = uhat - u   in Ω
    corresponding to the nonlinear state equation:
    du/dt + div(-eps*grad(u) + w*u) - u + (1/3)*u^3 = c      in Ω × [0,T]
                                              du/dn = 0      on ∂Ω × [0,T]
                                               u(0) = u0(x)  in Ω
    where the velocity field satisfies:
    div(w) = 0   in Ω × [0,T]
    w⋅n = 0      on ∂Ω × [0,T]

    Parameters
    ----------
    uk : numpy.ndarray 
        The state variable influenced by the control variable.
    uhat_T : numpy.ndarray 
        The desired target state at the final time T.
    pk : numpy.ndarray 
        The adjoint variable, initialized at final time T.
    T : float 
        Final time of the simulation.
    V : FunctionSpace
        Finite element function space.
    nodes : int 
        Number of nodes in the mesh.
    num_steps : int 
        Number of time steps.
    dt : float
        Time step size.
    dof_neighbors : list
        List where entry "i" contains DoF indices of neighbors of node "i".

    Returns
    -------
    pk: numpy.ndarray
        The computed adjoint variable pk at all time steps.
    """
    eps, _, wind = get_nonlinear_eqns_params()
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    M = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse_lil(dot(grad(u), grad(v)) * dx)
    A = assemble_sparse_lil(dot(wind, grad(v))*u * dx)
    Mat_p = lil_matrix(A.shape)
    Mat_p[:,] = -A[:,:] - eps * Ad[:,:]

    # Set final-time conditions
    pk[num_steps * nodes :] = uhat_T - uk[num_steps * nodes :]
    t = T
    print("\nSolving adjoint equation...")
    for i in reversed(range(0, num_steps)):
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 50 == 0:
            print("t = ", round(t, 4))
            
        pk_np1 = pk[end : end + nodes]                  # pk(t_{n+1})
        uk_n_fun = vec_to_function(uk[start : end], V)  # uk(t_n)
        M_u2 = assemble_sparse_lil(uk_n_fun**2 * u * v *dx)
        Mat_rhs = lil_matrix(A.shape)
        Mat_rhs[:,:] = M_u2[:,:] - M[:,:] 
        p_rhs = np.zeros(nodes)
        pk[start:end] = FCT_alg_ref(-Mat_p, p_rhs, pk_np1, dt, nodes, M, M_lumped, 
                                dof_neighbors, non_flux_mat = Mat_rhs)
    return pk

def plot_nonlinear_solution(uk, pk, ck, uhat_T_re, T_data, it, nodes, num_steps, 
                            dt, out_folder, vertex_to_dof, step_freq=20):
    """
    Produces plots for the computed state variable, adjoint variable, and control variable 
    of the final-time nonlinear equation PDECO problem.
    Parameters
    ----------
    uk : numpy.ndarray 
        The state variable in dof ordering.
    pk : numpy.ndarray
        The adjoint variable in dof ordering.
    ck : numpy.ndarray
        The control variable in dof ordering.
    uhat_T_re : numpy.ndarray
        The desired target state for u at the final time T in vertex ordering
    T_data : float
        Final time of the simulation.
    it : int
        Current iteration number.
    nodes : int
        Number of nodes in the mesh.
    num_steps : int
        Number of time steps.
    dt : float
        Time step size.
    out_folder : str
        Output folder for saving plots.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.
    step_freq : int, default 20
        Frequency of steps to plot.
    
    Returns
    -------
    None
        The function saves the plots to the specified folder and does not return any value.
    """
    sqnodes = round(np.sqrt(nodes))
    uk_re = reorder_vector_from_dof(uk, num_steps + 1, nodes, vertex_to_dof)
    ck_re = reorder_vector_from_dof(ck, num_steps + 1, nodes, vertex_to_dof)
    pk_re = reorder_vector_from_dof(pk, num_steps + 1, nodes, vertex_to_dof)
    
    for i in range(num_steps):
        startP = i * nodes
        endP = (i+1) * nodes
        tP = i * dt
        
        startU = (i+1) * nodes
        endU = (i+2) * nodes
        tU = (i+1) * dt
        
        u_re = uk_re[startU : endU].reshape((sqnodes,sqnodes))
        c_re = ck_re[startP : endP].reshape((sqnodes,sqnodes))
        p_re = pk_re[startP : endP].reshape((sqnodes,sqnodes))
            
        if i % step_freq == 0 or i == num_steps-1:
            
            fig = plt.figure(figsize=(20, 5))

            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(uhat_T_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f"{it=}, Desired state for $u$ taken at t={T_data}")
        
            ax = fig.add_subplot(1, 4, 2)
            im2 = ax.imshow(u_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f"Computed state $u$ at t={round(tU, 5)}")
        
            ax = fig.add_subplot(1, 4, 3)
            im3 = ax.imshow(p_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f"Computed adjoint $p$ at t={round(tP, 5)}")
        
            ax = fig.add_subplot(1, 4, 4)
            im4 = ax.imshow(c_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f"Computed control $c$ at t={round(tP, 5)}")
            
            fig.tight_layout(pad=3.0)
            plt.savefig(out_folder + f"/it_{it}_plot_{i:03}.png")
        
            # Clear and remove objects explicitly
            ax.clear()      # Clear axes
            cb1.remove()    # Remove colorbars
            cb2.remove()
            cb3.remove()
            cb4.remove()
            del im1, im2, im3, im4, cb1, cb2, cb3, cb4
            fig.clf()
            plt.close(fig) 
    return None

def plot_progress(cost_fun_vals, cost_fidel_vals, cost_c_vals, it, out_folder, 
                  cost_fidel_vals2=None, v1_name="u", v2_name="v"):

    """
    Produces plots to track progress of an optimization algorithm:
    1. Cost functional values over iterations.
    2. Data fidelity norm(s) over iterations.
    3. Regularization norm in L2(Q)^2 over iterations.
    
    Parameters:
    -----------
    cost_fun_vals : list or array-like
        Values of the cost functional at each iteration.
    cost_fidel_vals : list or array-like
        Values of the data fidelity term at each iteration.
    cost_c_vals : list or array-like
        Values of the regularization term at each iteration.
    it : int
        Current iteration number.
    out_folder : str
        Path to the folder where the plots will be saved.
    cost_fidel_vals2 : list or array-like [optional]
        Values of a second data fidelity term..
    v1_name : str, default "u" [optional]
        Label for the first data fidelity term.
    v2_name : str, default "v" [optional]
        Label for the second data fidelity term.

    Returns:
    --------
    None
        The function saves the plots to the specified folder and does not return any value.
    """
    fig2 = plt.figure(figsize=(15, 5))

    ax2 = fig2.add_subplot(1, 3, 1)
    im1 = plt.plot(np.arange(0, it + 2), cost_fun_vals)
    plt.yscale('log')
    plt.title(f"{it=} Cost functional")
    
    ax2 = fig2.add_subplot(1, 3, 2)
    im2_u = plt.plot(np.arange(1, it + 2), cost_fidel_vals, label=v1_name)
    if cost_fidel_vals2 is not None:
        im2_v = plt.plot(np.arange(1, it + 2), cost_fidel_vals2, label=v2_name)
        if v2_name is not None:
            plt.legend()
    plt.title("Data fidelity norms in L2(Omega)^2")
    
    ax2 = fig2.add_subplot(1, 3, 3)
    im3 = plt.plot(np.arange(1, it + 2), cost_c_vals)
    plt.title("Regularisation norm in L2(Q)^2")
    
    fig2.tight_layout(pad=3.0)
    plt.savefig(out_folder + "/progress_plot.png")
    
    # Clear and remove objects explicitly
    ax2.clear()      # Clear axes
    del im1, im2_u, im3
    if cost_fidel_vals2 is not None:
        del im2_v
    fig2.clf()
    plt.close(fig2)
    return None
    
def get_chtxs_sys_params():
    """
    Returns the shared parameters (floats) for the chemotaxis state and adjoint equations:
     du/dt + ∇⋅(-Dm*∇u + χ*u*exp(-ηu)*∇v) = 0                  in Ω × [0,T]
                 dv/dt + ∇⋅(-Df*∇v) + δ*v = γ*u                 in Ω × [0,T]
    -dp/dt + ∇⋅(-Dm*∇p) - χ*(1 - η*u)*exp(-ηu)*∇p⋅∇v = γ*q     in Ω x [0,T]
         -dq/dt + ∇⋅(-Df*∇q + χ*u*exp(-ηu)*∇p) + δ*q = 0        in Ω x [0,T]
    """
    delta = 100     # Decay parameter for v
    Dm = 0.05       # Diffusion parameter for u
    Df = 0.05       # Diffusion parameter for v
    chi = 0.25      # Chemotaxis strength
    gamma = 100     # The control parameter, can be rescaled by rewriting term as γ*u/r 
    eta = 0.5       # Parameter in chemotactic term
    return delta, Dm, Df, chi, gamma, eta

def chtxs_sys_IC(a1, a2, deltax, nodes, vertex_to_dof):
    """
    Computes the initial condition for the chemotaxis system on a 2D square mesh grid.
    We get u(0)=v(0)

    Parameters
    ----------
    a1 : float 
        Left endpoint of the spatial domain [a1,a2]x[a1,a2].
    a2 : float 
        Right endpoint of the spatial domain [a1,a2]x[a1,a2].
    deltax : float 
        Grid spacing in both X and Y directions.
    nodes : int
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray 
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    u_init_dof, v_init_dof : tuple (numpy.ndarray, numpy.ndarray)
        Flattened 2D array representing the initial condition over the mesh grid.
    """
    X = np.arange(a1, a2 + deltax, deltax)
    Y = np.arange(a1, a2 + deltax, deltax)
    X, Y = np.meshgrid(X,Y)
    
    delta, Dm, Df, chi, gamma, eta = get_chtxs_sys_params()
    sqnodes = round(np.sqrt(nodes))
    np.random.seed(5)
    u_init = 1.5 + 0.1*(0.5 - np.random.rand(sqnodes,sqnodes))
    
    u_init_dof = reorder_vector_to_dof(
        u_init.reshape(nodes), 1, nodes, vertex_to_dof)
    v_init_dof = u_init_dof
    return u_init_dof, v_init_dof

def solve_chtxs_system(control, var1, var2, V, nodes, num_steps, dt, dof_neighbors,
                       control_fun=None, show_plots=False, vertex_to_dof=None,
                       generation_mode=False, output_dir=None, rescaling=1/10):
    """
    Solver for the chemotaxis system
    du/dt + ∇⋅(-Dm*∇u + χ*u*exp(-ηu)*∇v) = 0    in Ω × [0,T]
    dv/dt + ∇⋅(-Df*∇v) + δ*v = c*(u/r)          in Ω × [0,T]
    (-Dm*∇u + χ*u*exp(-ηu)*∇v)⋅n = 0            on ∂Ω × [0,T]
                            ∇v⋅n = 0            on ∂Ω × [0,T]
                            u(0) = u0(x)        in Ω
                            v(0) = v0(x)        in Ω

    Parameters
    ----------
    control : numpy.ndarray 
        Control variable.
    var1 : numpy.ndarray 
        The state variable u.
    var2 : numpy.ndarray 
        The second state variable, v, influenced by the control variable.
    V : FunctionSpace 
        Finite element function space.
    nodes : int
        Number of nodes in the mesh.
    num_steps : int
        Number of time steps.
    dt : float 
        Time step size.
    dof_neighbors : list
        List where entry "i" contains DoF indices of neighbors of node "i".
    control_fun : dolfin.Expression [optional]
         Control as a given expression, used to generate target data.
    show_plots : bool, default False
        Choose whether to plot the solution.
    vertex_to_dof : numpy.ndarray [optional]
        Mapping from vertex indices to DoF indices.
    generation_mode : bool, default False
        Used to generate target states when memory constrains the possible size 
        of the output vectors. If True, chooses short vectors of length "nodes" 
        for storage.
    output_dir : str [optional]
        Directory to save the output files when in generation mode.
    rescaling : float, default 1
        Parameter to rescale the control parameter.
    
    Returns
    -------
    var1, var2: tuple (numpy.ndarray,numpy.ndarray)
        Solutions for the state variables var1 and var2.
    """
    delta, Dm, Df, chi, c_gamma, eta = get_chtxs_sys_params()
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    sqnodes = round(np.sqrt(nodes))
    M = assemble_sparse_lil(u*v*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    Mat_var2 = M + dt * (Df * Ad + delta * M)

    # Reset variables but keep initial conditions
    if not generation_mode:
        var1[nodes :] = np.zeros(num_steps * nodes)
        var2[nodes :] = np.zeros(num_steps * nodes)
    else:
        if len(var1) != nodes or len(var2) != nodes or len(control) != nodes:
            raise ValueError(f"Generation mode, the input vectors should be of length {nodes}")
        var1_n = var1
        var2_n = var2
    t = 0
    print("Solving the system of chemotaxis state equations...")
    for i in range(1, num_steps + 1):  # Solve for var2(t_{n+1}), var1(t_{n+1})
        t += dt
        if i % 10 == 0:
            print("t = ", round(t, 4))

        if not generation_mode:
            start = i * nodes
            end = (i + 1) * nodes
                    
            var1_n = var1[start - nodes : start]
            var2_n = var2[start - nodes : start]
            if control_fun is None:
                control_fun = vec_to_function(control[start : end], V)
            
        var1_n_fun = vec_to_function(var1_n, V)
        var2_n_fun = vec_to_function(var2_n, V)
        
        # Solve R-D equation for the variable "v" with a direct solver
        var2_rhs = np.asarray(assemble(
            var2_n_fun * v * dx  + dt * control_fun * var1_n_fun / rescaling * v * dx))
        
        var2_np1 = spsolve(Mat_var2, var2_rhs)
        
        if not generation_mode:
            var2[start : end] = var2_np1
            
        var2_np1_fun = vec_to_function(var2_np1, V)
        
        # Solve convection equation for the variable "u" with FCT
        Aa = assemble_sparse(
            exp(-eta*var1_n_fun)*dot(grad(var2_np1_fun), grad(v)) * u * dx)
        A_var1 = Dm * Ad - chi*Aa
        var1_rhs = np.zeros(nodes)
    
        var1_np1 = FCT_alg_ref(
            A_var1, var1_rhs, var1_n, dt, nodes, M, M_lumped, dof_neighbors)
        
        if not generation_mode:
            var1[start : end] = var1_np1
        else:
            var1_n = var1_np1
            var2_n = var2_np1
            if output_dir is not None and i % 100 == 0:
                output_filename_m = output_dir / f"chtxs_m_t{round(t,2)}.csv"
                output_filename_f = output_dir / f"chtxs_f_t{round(t,2)}.csv"
                var1_np1.tofile(output_filename_m, sep=",")
                var2_np1.tofile(output_filename_f, sep=",")

        if show_plots is True and i % 100 == 0:
            var1_re = reorder_vector_from_dof(var1_np1, 1, nodes, vertex_to_dof)
            var2_re = reorder_vector_from_dof(var2_np1, 1, nodes, vertex_to_dof)
            
            fig2 = plt.figure(figsize=(10, 5), dpi=100)
            
            ax2 = plt.subplot(1,2,1)
            im1 = plt.imshow(var1_re.reshape((sqnodes,sqnodes)), cmap ="gray")
            fig2.colorbar(im1)
            plt.title(f"Computed state $m$ at t={round(t,5)}")
            
            ax2 = plt.subplot(1,2,2)
            im2 = plt.imshow(var2_re.reshape((sqnodes,sqnodes)), cmap="gray")
            fig2.colorbar(im2)
            plt.title(f"Computed state $f$ at t={round(t,5)}")
            plt.show()
    return var1, var2

def solve_adjoint_chtxs_system(uk, vk, uhat, vhat, pk, qk, control, T, V, 
                               nodes, num_steps, dt, dof_neighbors, optim,
                               show_plots=None, vertex_to_dof=None, out_folder=None,
                               mesh=None, deltax=None, rescaling=1/10):
    """
    Solves the adjoint system equations:
    -dp/dt + ∇⋅(-Dm*∇p) - χ*(1-η*u)*exp(-ηu)*∇p⋅∇v = c*q + (1-σ)(û-u)   in Ω x [0,T]
    -dq/dt + ∇⋅(-Df*∇q + χ*u*exp(-ηu)*∇p) + δ*q = (1-σ)(v̂-v)            in Ω x [0,T]
                                           ∇p⋅n = ∇q⋅n = 0              on ∂Ω x [0,T]
                                           p(T) = σ(û_T - u(T))         in Ω
                                           q(T) = σ(v̂_T - v(T))      in Ω
    (indicator σ=0 for all-time optimization and σ=1 for final-time optimization)
    corresponding to the chemotaxis system:
    du/dt + ∇⋅(-Dm*∇u + χ*u*exp(-ηu)*∇v) = 0    in Ω × [0,T]
    dv/dt + ∇⋅(-Df*∇v) + δ*v = c*u              in Ω × [0,T]
    (-Dm*∇u + χ*u*exp(-ηu)*∇v)⋅n = 0            on ∂Ω × [0,T]
                            ∇v⋅n = 0            on ∂Ω × [0,T]
                            u(0) = u0(x)        in Ω
                            v(0) = v0(x)        in Ω
                                       
    Parameters
    ----------
    uk : numpy.ndarray 
        The state variable influenced by the control variable.
    vk : numpy.ndarray 
        The second state variable.
    uhat : numpy.ndarray 
        The desired target state for u.
    vhat : numpy.ndarray 
        The desired target state for v.
    pk : numpy.ndarray 
        The adjoint variable, initialized at final time T.
    qk : numpy.ndarray 
        Second adjoint variable, initialized at final time T.
    control : numpy.ndarray 
        Control variable.
    T : float 
        Final time of the simulation.
    V : FunctionSpace 
        Finite element function space.
    nodes : int 
        Number of nodes in the mesh.
    num_steps : int
        Number of time steps.
    dt : float
        Time step size.
    dof_neighbors : list 
        List where entry "i" contains DoF indices of neighbors of node "i".
    optim : str {"alltime","finaltime"}
        Optimization type.
    show_plots : bool, default False
        Choose whether to plot the solution.
    vertex_to_dof : numpy.ndarray [optional]
        Mapping from vertex indices to DoF indices.
    out_folder : str [optional]
        Directory to save the output files.
    mesh : dolfin.Mesh [optional]
        Computational mesh.
    deltax : float [optional]
        Grid spacing in both X and Y directions.
    rescaling : float, default 1
        Parameter to rescale the control parameter.

    Returns
    -------
    pk, qk : numpy.ndarray
        The computed adjoint variables through space and over all time steps.
    """
    valid_options = ["alltime", "finaltime"]
    if optim not in valid_options:
        raise ValueError(f"Invalid value for 'optim': '{optim}'. Must be one of {valid_options}.")

    ## For gradient smoothing
    # V_vec = df.VectorFunctionSpace(mesh, "CG", 1)  # continuous vector field
    # V_dg0 = df.VectorFunctionSpace(mesh, "DG", 0)  # discontinuous gradient projection
    # u = df.TrialFunction(V_vec)
    # v = df.TestFunction(V_vec)
    # print("Using gradient smoothing for p")
    
    delta, Dm, Df, chi, _, eta = get_chtxs_sys_params()
    
    p = df.TrialFunction(V)
    w = df.TestFunction(V)
    M = assemble_sparse_lil(p*w*dx)
    M_lumped = row_lump(M, nodes)
    Ad = assemble_sparse_lil(dot(grad(p), grad(w)) * dx)

    # Set final-time conditions
    if optim=="finaltime":
        pk[num_steps * nodes :] = uhat - uk[num_steps * nodes :]
        qk[num_steps * nodes :] = vhat - vk[num_steps * nodes :]
    t = T
    print("\nSolving adjoint equation...")
    for i in reversed(range(0, num_steps)): # solve for pk(t_n), qk(t_n)
        start = i * nodes
        end = (i + 1) * nodes
        t -= dt
        if i % 10 == 0:
            print("t = ", round(t, 4))
            
        q_np1 = qk[end : end + nodes] # qk(t_{n+1})
        p_np1 = pk[end : end + nodes] # pk(t_{n+1})
    
        p_np1_fun = vec_to_function(p_np1, V) 
        q_np1_fun = vec_to_function(q_np1, V) 

        u_n_fun = vec_to_function(uk[start : end], V)      # uk(t_n)
        v_n_fun = vec_to_function(vk[start : end], V)      # vk(t_n)

        control_fun = vec_to_function(control[start : end], V) # ck(t_n)
        
        # Solve R-D equation for the variable "q" with a direct solver        
        Aa = assemble_sparse(
            (1-eta*u_n_fun) * exp(-eta*u_n_fun) * dot(grad(p), grad(v_n_fun)) * w * dx)
        
        Mat_p = lil_matrix(Ad.shape)
        Mat_p[:,:]  = Dm*Ad[:,:] - chi*Aa[:,:]
        
        rhs_p = np.asarray(assemble(control_fun * q_np1_fun / rescaling * w * dx))
        if optim == "alltime":
            rhs_p += uhat[start : end] - uk[start : end]
            
        p_new = FCT_alg_ref(Mat_p, rhs_p, p_np1, dt, nodes, M, M_lumped, 
                                dof_neighbors, vertex_to_dof=vertex_to_dof)
        ## Option: rescale boundary nodes and smooth corners (also uncomment vector spaces above and code for q below)
        # p_new = rescale_boundary_nodes(p_new, vertex_to_dof, a1=0, a2=1, deltax=deltax)
        # p_new = smooth_corners_on_boundary(p_new, V, vertex_to_dof, a1=0, a2=1, deltax=deltax)
        # print("Using rescaling and smoothing for p,q")
        pk[start:end] = p_new
        
        p_n_fun = vec_to_function(pk[start:end], V)
        
        ## Option: Gradient smoothing:
        # # Project raw gradient of p_n_fun into DG(0)
        # grad_p_raw = df.project(df.grad(p_n_fun), V_dg0)
        # # Solve a variational problem to smooth grad_p_raw into CG(1)
        # a = df.inner(u, v) * df.dx
        # L = df.inner(grad_p_raw, v) * df.dx
        # grad_p_smooth = df.Function(V_vec)
        # df.solve(a == L, grad_p_smooth)
        # rhs_q = np.asarray(df.assemble(
        #     chi * u_n_fun * df.exp(-eta * u_n_fun) * dot(grad_p_smooth, grad(w)) * df.dx))
        
        # if using gradient smoothing, disable rhs_q below
        rhs_q = np.asarray(assemble(
            chi*u_n_fun * exp(-eta*u_n_fun) * dot(grad(p_n_fun), grad(w))*dx))
        if optim == "alltime":
            rhs_q += vhat[start : end] - vk[start : end]
            
        Mat_q = M + dt*(Df*Ad + delta*M)
        
        q_new = spsolve(Mat_q, M@q_np1 + dt*rhs_q) 
        ## Option: rescale boundary nodes and smooth corners
        # q_new = rescale_boundary_nodes(q_new, vertex_to_dof, a1=0, a2=1, deltax=deltax)
        # q_new = smooth_corners_on_boundary(q_new, V, vertex_to_dof, a1=0, a2=1, deltax=deltax)
        qk[start:end] = q_new
        
        if show_plots: # to use for debugging and just adjoint equation solver
            sqnodes = round(np.sqrt(nodes))
            pk_re = reorder_vector_from_dof(pk[start:end], 1, nodes, vertex_to_dof).reshape((sqnodes,sqnodes))
            qk_re = reorder_vector_from_dof(qk[start:end], 1, nodes, vertex_to_dof).reshape((sqnodes,sqnodes))
            sc_uhat_re = 0.2*reorder_vector_from_dof(uhat[start:end], 1, nodes, vertex_to_dof).reshape((sqnodes,sqnodes))
            sc_vhat_re = 0.2*reorder_vector_from_dof(vhat[start:end], 1, nodes, vertex_to_dof).reshape((sqnodes,sqnodes))
                           
            fig = plt.figure(figsize=(20, 5))
            ax = fig.add_subplot(1, 4, 1)
            im1 = ax.imshow(pk_re)
            cb1 = fig.colorbar(im1, ax=ax)
            ax.set_title(f"Computed adjoint $p$ at t={round(t, 5)}")
        
            ax = fig.add_subplot(1, 4, 2)
            im2 = ax.imshow(qk_re)
            cb2 = fig.colorbar(im2, ax=ax)
            ax.set_title(f"Computed adjoint $q$ at t={round(t, 5)}")
            
            ax = fig.add_subplot(1, 4, 3)
            im3 = ax.imshow(sc_uhat_re)
            cb3 = fig.colorbar(im3, ax=ax)
            ax.set_title(f"$0.2*\hat u$ at t={round(t, 5)}")
        
            ax = fig.add_subplot(1, 4, 4)
            im4 = ax.imshow(sc_vhat_re)
            cb4 = fig.colorbar(im4, ax=ax)
            ax.set_title(f"$0.2*\hat v$ at t={round(t, 5)}")
            plt.show()
            fig.tight_layout(pad=3.0)
            plt.savefig(os.path.join(out_folder, f"/adj_{i:03}.png"))
            # Clear and remove objects explicitly
            ax.clear()      # Clear axes
            cb1.remove()    # Remove colorbars
            cb2.remove()
            del im1, im2, cb1, cb2
            fig.clf()
            plt.close(fig) 
    return pk, qk
    
def armijo_line_search_ref(var1, c, d, var1_target, num_steps, dt, c_lower, 
                       c_upper, beta, costfun_init, nodes, optim, V, gam=1e-4, 
                       max_iter=10, s0=1, nonlinear_solver=None, dof_neighbors=None, 
                       var2=None, var2_target=None, w1=None, w2=None):
    """
    Performs Projected Armijo Line Search to find the optimal step size for a source control problem.

    Parameters
    ----------
    var1 : numpy.ndarray 
        The state variable influenced by the control variable.
    c : numpy.ndarray 
        Control variable.
    d : numpy.ndarray
        Descent direction (negative gradient).
    var1_target : numpy.ndarray 
        Target state for var1.
    num_steps : int 
        Number of time steps.
    dt : float
        Time step size.
    c_lower : float 
        Lower bound for the control variable.
    c_upper : float 
        Upper bound for the control variable.
    beta : float 
        Regularization parameter.
    costfun_init : float
        Initial cost function value.
    nodes : int
        Number of nodes in the mesh.
    optim : str {"alltime","finaltime"}
        Optimization type.
    V : FunctionSpace 
        Finite element function space.
    gam : float, default 1e-4
        Armijo condition parameter.
    max_iter : int. default 10
        Maximum number of iterations.
    s0 : float, default 1
        Initial step size.
    nonlinear_solver : callable [optional] 
        Not to be used when solving a linear problem.
        Function to solve a nonlinear problem (state equation(s)).
        Parameters: (control, var1, var2, V, nodes, num_steps, dt, dof_neighbors)
        Returns: var1, var2
    dof_neighbors : list [optional]
        List where entry "i" contains DoF indices of neighbors of node "i".
    var2 : numpy.ndarray [optional]
        Second state variable in coupled systems.
    var2_target : numpy.ndarray [optional]
         Target state for var2 in coupled systems.
    w1 : numpy.ndarray [optional]
        Increment for var1 in linear problems.
    w2 : numpy.ndarray [optional]
        Increment for var2 in linear problems.

    Returns
    -------
    (var1, var2, c_inc, k + 1) : tuple (numpy.ndarray,numpy.ndarray,numpy.ndarray,int)
        Returned if var2 is given. Includs updated states, control and optimal 
        step size.
    (var1, c_inc, k + 1) : tuple (numpy.ndarray,numpy.ndarray,int)
        Returned if var2 is not given. Includs updated state, control and optimal 
        step size.

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
    
    valid_options = ["alltime", "finaltime"]
    if optim not in valid_options:
        raise ValueError(f"Invalid value for 'optim': '{optim}'. Must be one of {valid_options}.")

    u, v, M = initialize_problem()

    s = s0
    control_dif_L2 = 1
    
    # Initialise the difference in cost functional norm decrease
    armijo = float("inf") 

    for k in range(max_iter): 
        print(f"{k=}")
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
        
        print(f"Updated cost={cost2}, Orig. cost={costfun_init}")
        print(f"Cost difference={armijo}")
        print(f"Threshold value: {-gam/s*control_dif_L2=}")
        
        # Check Armijo stopping condition
        if armijo <= -gam / s * control_dif_L2:
            print(f"Converged in {k+1} iterations: Armijo condition satisfied.")
            break

        s /= 2

    if armijo > -gam / s * control_dif_L2:
        print(f"Stopped: Maximum number of iterations reached ({max_iter}) .")

    return (var1, var2, c_inc, k + 1) if var2 is not None else (var1, c_inc, k + 1)

def FCT_alg_ref(A, rhs, u_n, dt, nodes, M, M_lumped, dof_neighbors, non_flux_mat=None,
                vertex_to_dof=None):
    """
    Applies the Flux Corrected Transport (FCT) algorithm with linearized fluxes
    to the equation
        [ M + dt*(A + non_flux_mat)] u^{n+1} = M u^n + dt*rhs.
    Uses antidiffusive fluxes and a correction factor matrix calculated with 
    Zalesak algorithm. The artificial diffusion matrix D cancelS the negative
    off-diagonal elements of the flux matrix "-A" in the "high-order" equation:
        M du/dt = -A u - non_flux_mat u + rhs.
    The "low-order" equation is:
        M_lumped du/dt = (D-A) u + rhs,
    from which we obtain the transported and diffused (low-order) solution.
    The flux is the residual between the low- and high-order equations:
        f = (M_lumped - M) du/dt - D u
    where the first term simplifies to 
        sum_{i\neq j} M_{ij} ( (du/dt)_j - (du/dt)_i)
    and the second term simplifies to
        sum_{i\neq j} D_{ij} (u_j - u_i)
    The derivative is approximated using the Chebyshev semi-iterative method 
    applied to the equation:
        M du/dt = -A u^{low} + rhs, where u^{low} is the low-order solution.
    The final solution is obtained by correcting the low-order solution explicitly:
        U^{n+1}_i = u^{low}_i + dt * Fbar_i
    where Fbar_i = sum_{j\neq i} α_{ij} f_{ij}.
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix 
        The flux matrix (e.g., advection or diffusion).
    rhs : numpy.ndarray 
        The right-hand side (source) vector (not multiplied by dt).
    u_n : numpy.ndarray
        The solution vector at the current time step.
    dt : float
        The time step size.
    nodes : int
        Number of nodes in the mesh.
    M : scipy.sparse.lil_matrix 
        The mass matrix.
    M_lumped : scipy.sparse.spmatrix 
        The lumped mass matrix.
    dof_neighbors : list
        List where entry "i" contains DoF indices of neighbors of node "i".
    non_flux_mat : scipy.sparse.spmatrix [optional]
        Matrix terms that do not involve flux, from the RHS. Defaults to None.
    vertex_to_dof : numpy.ndarray [optional]
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    u_np1 : numpy.ndarray
        The flux-corrected solution vector at the next time step.
    """
    D = artificial_diffusion_mat(-A)
    M_diag = M.diagonal()
    M_lumped_diag = M_lumped.diagonal()
    Mat_u_Low = csr_matrix(D.shape)
    
    ## 1. Calculate low-order solution u^{n+1} using previous time step solution
    Mat_u_Low = M_lumped + dt * (A - D)

    if non_flux_mat is not None:
        Mat_u_Low += dt * non_flux_mat

    rhs_u_Low = M_lumped @ u_n + dt * rhs
    
    u_Low = spsolve(Mat_u_Low, rhs_u_Low)
    
    # ---------------- Check the 3 M-matrix properties ----------------------
    ## 1. all diag. coefs are positive (prints True if correct)
    # diag_Mat_u_Low = Mat_u_Low.diagonal()
    # print("1:",np.all(diag_Mat_u_Low > 0))

    ## 2. no positive off-diagonal entries (prints True if correct)
    # diag_Mat_u_Low = spdiags(diag_Mat_u_Low, diags=0, m=nodes, n=nodes)
    # Mat_u_Low_nondiag = Mat_u_Low - diag_Mat_u_Low # get a matrix without a diagonal
    # Mat_u_Low_flat = Mat_u_Low_nondiag.todense().reshape(nodes**2)
    # print("2:", np.all(Mat_u_Low_flat <= 0))
        
    ## 3. diagonally dominant (prints True if correct, False if not along with bounds on dt)
    Mat_u_Low_rowsums = np.sum(Mat_u_Low.todense(), axis=1)
    # print("3:", np.all(Mat_u_Low_rowsums > 0))
    if np.all(Mat_u_Low_rowsums > 0) == False:
        print("3:", np.all(Mat_u_Low_rowsums > 0))
        lower_bounds_dt = []
        upper_bounds_dt = []
        row_sums_A = [ A[i, :].sum() for i in range(A.shape[0]) ]
        for i in range(len(row_sums_A)):
            if row_sums_A[i] < 0:
                upper_bounds_dt.append(-M_lumped_diag[i] / row_sums_A[i])
            elif row_sums_A[i] > 0:
                lower_bounds_dt.append(-M_lumped_diag[i] / row_sums_A[i])
        print("Upper bound on dt:", min(upper_bounds_dt))
        print("Lower bound on dt:", max(max(lower_bounds_dt), 0))
    # -----------------------------------------------------------------------

    ## 2. Calculate raw antidiffusive flux
    # approximate the derivative du/dt using Chebyshev semi-iterative method
    rhs_du_dt = np.squeeze(np.asarray(-A @ u_Low + rhs)) # flatten to vector array
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
    
    
    # ## IDEA: reset r to 1 on boundary nodes (normally used with non-zero-flux BCs)
    # _, boundary_nodes_dof = generate_boundary_nodes(nodes, vertex_to_dof)
    # r_pos[boundary_nodes_dof] = 1
    # r_neg[boundary_nodes_dof] = 1

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

def import_data_final(file_path, nodes, vertex_to_dof, num_steps=0,
                      time_dep=False):
    """
    Parameters
    ----------
    file_path : str
        Path to data file.
    nodes : int 
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.
    num_steps : int [optional, default=0]
        Number of time steps.
    time_dep : bool, default False
        If True, the function assumes the data should be extracted for 
        target states in all-time optimization (num_steps must be specified).
        If False, use default value for num_steps for use with final-time
        optimization problems.
        
    Returns
    -------
    data_re, data : touple (numpy.ndarray, numpy.ndarray)
        data_re: data in vertex ordering given as 2D array (for plotting).
        data: data in DoF ordering as 1D array (for computations).
    """        
    sqnodes = round(np.sqrt(nodes))
    data = np.genfromtxt(file_path, delimiter=",")
    if time_dep: # for use with all-time optimization
        # Subset the data
        data = data[:(num_steps + 1) * nodes]
    data_re = reorder_vector_from_dof(data, num_steps + 1, nodes, vertex_to_dof)

    if not time_dep: # only loads one frame for final-time target state
        data_re = data_re.reshape((sqnodes,sqnodes))
        
    return data_re, data

def extract_data(file_path, file_name, T, dt, nodes, vertex_to_dof):
    """
    Extracts the solution vector at a specific time step from a CSV file and 
    saves it as a new CSV file.
    
    The input CSV file contains the solution vector over the time interval [0, T], 
    with dimensions ((Nt + 1) * Nx), where Nt is the number of time steps and Nx 
    is the number of spatial nodes. The function extracts the solution at time T 
    and saves it separately.
    
    Parameters
    ----------
    file_path : str
        Directory containing the input CSV file.
    file_name : str
        Name of the input CSV file (without extension).
    T : float
        Time at which the solution is extracted.
    dt : float 
        Time step size.
    nodes : int
        Number of nodes in the mesh.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.

    Returns
    -------
    None : 
        Saves the extracted data as a new CSV file.
    """
    
    idx = round(T / dt)
    start_col = idx * nodes
    end_col = (idx + 1) * nodes
    
    input_file = os.path.join(file_path, f"{file_name}.csv")
    output_file = os.path.join(file_path, f"{file_name}_T{T}.csv")
    
    # Load only the required columns
    data = pd.read_csv(input_file, header=None, usecols=range(start_col, end_col), nrows=1)
    
    np.savetxt(output_file, data.to_numpy().flatten(), delimiter=",")
    print(f"Extracted data at {T=} into {output_file}.")
    return None

def norm_true_control(example, T, dt, M, V, c_a=None):
    """
    Calculates the squared norm in L^2(Q) of the true control for a given
    PDECO problem and final time T where Q = Ω × [0,T].
    
    Parameters
    ----------
    example : string {"nonlinear","Schnak","chtxs"}
        Selected problem.
    T : int 
        Final time for the problem used to generate target states.
    dt : float 
        Time step size.
    M : numpy.ndarray
        Mass matrix.
    V : FunctionSpace
        Finite element function space.
    c_a : float [optional]
        True control parameter, used in "Schnak" or "chtxs" problems.

    Returns
    -------
    float: Squared L^2-norm of the control in spatiotemporal domain Q.
    """
    valid_options = ["nonlinear", "Schnak", "chtxs"]
    if example not in valid_options:
        raise ValueError(f"Invalid value for 'example': '{example}'. Must be one of {valid_options}.")
    
    num_steps = round(T / dt)

    if example == "nonlinear":
        k, l = 2, 2
        control_fun = df.Expression("sin(k1 * pi * x[0]) * sin(k2 * pi * x[1])",
            degree=4, pi=np.pi, k1=k, k2=l)
    
        # Find L^2-norm of true control over Ω × [0,T]
        control_vector = df.interpolate(control_fun, V).vector().get_local()
        control_vector_td = np.tile(control_vector, num_steps + 1)
    elif example == "Schnak":
        vec_length = (num_steps + 1) * M.shape[0]
        control_vector_td = c_a * np.ones(vec_length)
    control_norm = L2_norm_sq_Q(control_vector_td, num_steps, dt, M)
    
    return control_norm
    
def smooth_corners_on_boundary(vec, V, vertex_to_dof, a1, a2, deltax):
    """
    Replace the value at each corner DoF with the average of its boundary neighbors.

    Parameters
    ----------
    vec : numpy.ndarray 
        A 1D vector of a flattened 2D array over a square mesh in DoF ordering.
    V : FunctionSpace 
        Finite element function space.
    vertex_to_dof : numpy.ndarray 
        Mapping from vertex indices to DoF indices.

    a1, a2 : float 
        Domain bounds (assumed square).
    deltax : float 
        Grid spacing in both X and Y directions.

    Returns
    -------
    new_vec : numpy.ndarray
        Modified vec with smoothed corners.
    """
    sqnodes = round((a2 - a1) / deltax) + 1
    corner_vertices = {
        'bl': 0,
        'br': sqnodes - 1,
        'tl': (sqnodes - 1) * sqnodes,
        'tr': sqnodes * sqnodes - 1
    }

    # Get corresponding DoF indices
    corner_dofs = {k: int(vertex_to_dof[v]) for k, v in corner_vertices.items()}

    # Neighboring boundary vertices (horizontal and vertical) for each corner
    boundary_neighbors = {
        'bl': [1, sqnodes],               # right and top neighbor
        'br': [sqnodes - 2, 2*sqnodes - 1], # left and top neighbor
        'tl': [(sqnodes - 2) * sqnodes, (sqnodes - 1) * sqnodes + 1],  # bottom and right
        'tr': [(sqnodes - 1) * sqnodes + sqnodes - 2, sqnodes * (sqnodes - 1) - 1]  # bottom and left
    }

    new_vec = vec.copy()

    for corner, neighbors in boundary_neighbors.items():
        neighbor_dofs = [int(vertex_to_dof[v]) for v in neighbors]
        avg = np.mean([vec[d] for d in neighbor_dofs])
        new_vec[corner_dofs[corner]] = avg

    return new_vec
    
def rescale_boundary_nodes(u_vec, vertex_to_dof, a1=0, a2=1, deltax=0.025):
    """
    Linearly rescales boundary node values to fall within the range of
    the adjacent inner row/column of nodes.

    Parameters
    ----------
    u_vec : numpy.ndarray
        The vector of nodal values (DoFs).
    V : FunctionSpace 
        Finite element function space.
    vertex_to_dof : numpy.ndarray
        Mapping from vertex indices to DoF indices.
    a1: float, default 0
        Domain bound (start), square domain assumed.
    a2 : float, default 1
        Domain bounds (end), square domain assumed.
    deltax : float, default 0.025
        Grid spacing in both X and Y directions.
        
    Returns
    -------
    new_u : numpy.ndarray
        Rescaled DoF vector.
    """
    new_u = u_vec.copy()
    sqnodes = round((a2 - a1) / deltax) + 1  # number of nodes per side

    def index(i, j):
        return i * sqnodes + j  # row-major ordering

    boundaries = {
        "bottom": [(0, j) for j in range(sqnodes)],
        "top": [(sqnodes - 1, j) for j in range(sqnodes)],
        "left": [(i, 0) for i in range(sqnodes)],
        "right": [(i, sqnodes - 1) for i in range(sqnodes)],
    }

    adjacents = {
        "bottom": [(1, j) for j in range(sqnodes)],
        "top": [(sqnodes - 2, j) for j in range(sqnodes)],
        "left": [(i, 1) for i in range(sqnodes)],
        "right": [(i, sqnodes - 2) for i in range(sqnodes)],
    }

    global_min = np.min(u_vec)
    global_max = np.max(u_vec)
    eps = 1e-12  # for safety in division

    for side in boundaries:
        b_indices = [index(i, j) for i, j in boundaries[side]]
        a_indices = [index(i, j) for i, j in adjacents[side]]

        b_dofs = [int(vertex_to_dof[v]) for v in b_indices]
        a_dofs = [int(vertex_to_dof[v]) for v in a_indices]

        interior_values = [u_vec[d] for d in a_dofs]
        u_min_adj = min(interior_values)
        u_max_adj = max(interior_values)

        # Avoid division by zero if global range is flat
        scale_denominator = max(global_max - global_min, eps)

        for d in b_dofs:
            u_b = u_vec[d]
            t = (u_b - global_min) / scale_denominator
            new_u[d] = u_min_adj + t * (u_max_adj - u_min_adj)
    return new_u












