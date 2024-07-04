import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from dolfin import dx, dot, grad, div, assemble
from helpers import assemble_sparse
# Contains functions used for importing experimental data

# def get_data_array(var, final_time):
#     '''
#     Loads the target data: 
#         hat{m} if input is "m", or hat{f} if input is "f"
#     Size of the loaded array should be sqnodes x sqnodes.
#     Output = numpy array sqnodes x sqnodes.
#     '''
#     data = np.genfromtxt('data/mimura_tsujikawa_t' + str(final_time) + '_' + str(var) + '.csv', delimiter=',')

#     return data

def m_initial_condition(a1, a2, deltax):
    X = np.arange(a1, a2 + deltax, deltax)
    Y = np.arange(a1, a2 + deltax, deltax)
    X, Y = np.meshgrid(X,Y)
    
    # Initialize the condition array with zeros
    m0_array = np.ones(X.shape)
    
    # Add the perturbation
    radius_squared = 1.5 ** 2
    center_x, center_y = 8, 8
    condition = (X - center_x)**2 + (Y - center_y)**2 <= radius_squared
    # generate array of random numbers between (-0.5, 0.5)
    random_values = 0.05*np.random.rand(*X.shape)# - 0.5
    
    m0_array[condition] += random_values[condition]
    
    # same number everywhere:
    # m0_array[condition] += (np.random.rand()-0.5)
    
    return m0_array

def rhs_chtx_m(m_fun, v):
    # modified for the reaction term: m^2(1-m)
    return np.asarray(assemble(m_fun **2 * v * dx))

def rhs_chtx_f(f_fun, m_fun, dt, v):
    # modified to have c = 1 (multiplying m)
    return np.asarray(assemble(f_fun * v * dx  + dt * m_fun * v * dx))

def mat_chtx_m(f_fun, m_fun, Dm, chi, u, v):
    # modified for the reaction term: m^2(1-m)
    Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))
    Aa = assemble_sparse(assemble(dot(grad(f_fun), grad(v)) * u * dx))
    # Ar = assemble_sparse(assemble(m_fun **2 * u * v * dx))
    ## using the same reaction term matrix as in FEniCS sim.:
    Ar = assemble_sparse(assemble(m_fun * (1 - m_fun) * u * v * dx))

    return - Dm * Ad + chi * Aa + Ar

def mat_chtx_p(f_fun, m_fun, Dm, chi, u, v):
    # modified for the reaction term: m^2(1-m)
    Ad = assemble_sparse(assemble(dot(grad(u), grad(v)) * dx))
    Aa = assemble_sparse(assemble(dot(grad(f_fun), grad(v)) * u * dx))
    Adf = assemble_sparse(assemble(div(grad(f_fun)) * u * v * dx))
    Ar = assemble_sparse(assemble(m_fun*(2 - 3 * m_fun) * u * v * dx))
    return - Dm * Ad - chi * Aa - chi * Adf + Ar
