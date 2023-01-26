import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cvxpy as cp
from util import M_partial

def feasibility_DC(H, gamma, param):
    N,K = H.shape
    cur_obj = 0
    M_cur = np.random.randn(N,N) + 1j *np.random.randn(N,N)
    M_cur = M_cur@M_cur.conjugate().transpose()
    M_p = M_partial(M_cur)
    for i_iter in np.arange(20):
        M = cp.Variable((N,N),hermitian=True)
        obj = cp.real(cp.trace((-M_p.transpose().conjugate())@M+M))
        Objective = cp.Minimize(obj)

        constraints = []    
        for i_k in np.arange(K):
            constraints += [cp.real(cp.trace(M))-gamma*cp.real((H[:,i_k].transpose().conjugate()@M)@H[:,i_k] ) <= 0]

        constraints += [cp.real(cp.trace(M)) >= 1]
        constraints += [M >> 0]
        prob = cp.Problem(Objective, constraints)
        cvx_obj = prob.solve()

        if prob.status == 'infeasible':
            feasibility = 0
            m = []
            return m, feasibility

        err = np.absolute(cvx_obj - cur_obj)
        if param["verbosity"] > 2:
            print(f"iter:{i_iter} err:{err} obj: {cvx_obj}")
        M_cur = M.value
        if err < 1e-9 or cvx_obj < 1e-7:
            break
        M_p = M_partial(M_cur)
        cur_obj = cvx_obj
        
    u, singular_value, v = LA.svd(M_cur, hermitian=True)
    m = np.expand_dims(u[:,0],axis=1)
    feasibility = singular_value[1:].sum() < 1e-6
    if feasibility:
        for i_k in np.arange(K):
            if np.power(LA.norm(m,2),2)/np.power(LA.norm(m.conjugate().transpose()@H[:,i_k],2),2) >= gamma:
                feasibility = 0
                print(i_k)
                break
    return m, feasibility


def feasibility_SDR(H, gamma, param):
    N,K = H.shape 
    M = cp.Variable((N,N),hermitian=True)
    Objective = cp.Minimize(0)

    constraints = []    
    for i_k in np.arange(K):
        constraints += [cp.real(cp.trace((np.eye(N)-gamma*H[:,i_k]@H[:,i_k].transpose().conjugate())@M)) <=0]
   
    constraints += [cp.real(cp.trace(M)) >= 1]
    constraints += [M >> 0]
    prob = cp.Problem(Objective, constraints)
    cvx_obj = prob.solve()
    
    eigen_value,eigen_vec = LA.eigh(M.value)
    m = np.expand_dims(eigen_vec[:,-1],axis=1)
    
    
    # Use the randomization method to find a feasible m
    if LA.matrix_rank(M.value, 1e-6, hermitian=True) > 1:
        zi = SLA.cholesky(M.value, lower=True)
        xi = (np.random.randn(N,N) + 1j *np.random.randn(N,N))/np.sqrt(2)
        xi = zi@xi
        min_eig = 1e1000
        for i_k in np.arange(K):
            min_eig = min(min_eig, LA.norm(xi.transpose().conjugate()@H[:,i_k]),2)
            m = xi/min_eig

        feasibility = 1

        for i_k in np.arange(K):
            if np.power(LA.norm(m,2),2)/np.power(LA.norm(m.conjugate().transpose()@H[:,i_k],2),2) >= gamma:
                feasibility = 0
                break
            
    return m, feasibility