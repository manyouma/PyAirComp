import numpy as np
from numpy import linalg as LA
import cvxpy as cp

# compute the partial derivative of h_1 and h_2 w.r.t. M
def M_partial(M_1):
    # First get the eigenvalue decompositon
    eigen_value,eigen_vec = LA.eigh(M_1)
    # print(eigen_value)
    u = np.expand_dims(eigen_vec[:,-1],axis=1)
    M_partial = u@u.transpose().conjugate()
    return M_partial



# compute the partial derivative of h_1 w.r.t. x
def x_partial(x, c_k):
    x_partial = np.zeros_like(x)
    x_rank = (-np.abs(x)).argsort()
    x_partial[x_rank[:c_k]] =  np.sign(x[x_rank[:c_k]])
    return x_partial
    
