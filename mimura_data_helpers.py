import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from dolfin import dx, dot, grad, div, assemble, exp
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
    
    # # Initialize the condition array with zeros
    # m0_array = np.ones(X.shape)
    
    # # Add the perturbation
    # radius_squared = 1.5 ** 2
    # center_x, center_y = 8, 8
    # condition = (X - center_x)**2 + (Y - center_y)**2 <= radius_squared
    # # generate array of random numbers between (-0.5, 0.5)
    # random_values = 0.05*np.random.rand(*X.shape)# - 0.5
    
    # m0_array[condition] += random_values[condition]
    
    # same number everywhere:
    # m0_array[condition] += (np.random.rand()-0.5)
    
    
    ## mimura:
    # m0_array = np.exp(-5*((X-8)**2 + (Y-8)**2))
    
    
    ## painter ptashnyk headon 2021 
    n = X.shape[0]
    # con = 0.1
    # m0_array = np.ones((n,n)) + 0.5*(np.random.rand(n,n)-0.5*np.ones((n,n)))
    # m0_array = np.ones((n,n)) + con * np.cos(2 * np.pi * (X + Y)) + \
    # 0.01 * (sum(np.cos(2 * np.pi * X * i) for i in range(1, 9)))
    
    # m0_array = 1 + 0.2*np.cos(2*np.pi*X/5)*np.cos(2*np.pi*Y/5)
    
    
    # simplified feathers model
    np.random.seed(5)
    # k = 20
    # # m0_array = 1.5 + 0.6*(0.5 - np.random.rand(n,n)) 
    # m0_array = 1.5*np.ones((n,n)) + con * np.cos(2 * np.pi / k * (X + Y)) + \
    # 0.01 * (sum(np.cos(2 * np.pi / k * X * i) for i in range(1, 9)))
    
    m0_array = 1.5 + 0.1*(0.5 - np.random.rand(n,n))

    return m0_array

def rhs_chtx_m(m_fun, v):
    # modified for the reaction term: m^2(1-m)
    # return np.asarray(assemble(m_fun **2 * v * dx))
    
    # using IMEX so that the reaction term is on the RHS
    return np.asarray(assemble(m_fun **2 * (1 - m_fun)* v * dx))
    
def rhs_chtx_f(f_fun, m_fun, dt, v):
    
    ## mimura
    # modified to have c = 1 (multiplying m)
    # return np.asarray(assemble(f_fun * v * dx  + dt * m_fun * v * dx))

    ## painter ptashnyk headon 2021 and simplified feathers model
    return np.asarray(assemble(f_fun * v * dx  + dt * m_fun * v * dx))


def mat_chtx_m(f_fun, m_fun, Dm, chi, u, v):
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    
    ##mimura
    # Aa = assemble_sparse(assemble(dot(grad(f_fun), grad(v)) * u * dx))
    
    # modified for the reaction term: m^2(1-m)
    # Ar = assemble_sparse(assemble(m_fun **2 * u * v * dx))
    ## using the same reaction term matrix as in FEniCS sim.:
    # Ar = assemble_sparse(assemble(m_fun * (1 - m_fun) * u * v * dx))
    # used the above but the scale of the images wasn't right.
    Ar = np.zeros(Ad.shape)

    # return - Dm * Ad + chi * Aa + Ar
    ## painter ptashnyk headon 2021 and simplified feathers model
    beta = 0.5
    Aa = assemble_sparse(exp(-beta*m_fun)*dot(grad(f_fun), grad(v)) * u * dx)
    
    return - Dm * Ad + chi*Aa

def mat_chtx_p(f_fun, m_fun, Dm, chi, u, v):
    # modified for the reaction term: m^2(1-m)
    Ad = assemble_sparse(dot(grad(u), grad(v)) * dx)
    Aa = assemble_sparse(dot(grad(f_fun), grad(v)) * u * dx)
    Adf = assemble_sparse(div(grad(f_fun)) * u * v * dx)
    # Ar = assemble_sparse(m_fun*(2 - 3 * m_fun) * u * v * dx)
    Ar = np.zeros(Ad.shape)
    return - Dm * Ad - chi * Aa - chi * Adf + Ar
