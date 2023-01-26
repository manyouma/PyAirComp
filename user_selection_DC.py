import numpy as np
from numpy import linalg as LA
import cvxpy as cp
from util import M_partial, x_partial
from feasibility_DC import feasibility_DC, feasibility_SDR

def user_selection_DC(H, gamma, param):
    N,K = H.shape
    for c_k in np.arange(K):
        if param["verbosity"]>3:
            print(f"Current C: {c_k}")
        cur_obj = 0
        M_cur = np.random.randn(N,N) + 1j *np.random.randn(N,N)
        M_cur = M_cur@M_cur.conjugate().transpose()
        x_cur = np.random.randn(K)

        M_p = M_partial(M_cur)
        x_p = np.expand_dims(x_partial(x_cur, c_k),axis=1)

        for i_iter in np.arange(param["maxIter"]):
            M = cp.Variable((N,N),hermitian=True)
            x = cp.Variable(K) 
            obj = cp.norm(x,1) - x_p.transpose().conjugate()@x + cp.real(cp.trace((-M_p.transpose().conjugate())@M+M))
            Objective = cp.Minimize(obj)

            constraints = []    
            for i_k in np.arange(K):       
                constraints += [cp.real(cp.trace(M))-gamma*cp.real((H[:,i_k].transpose().conjugate() @ M)@H[:,i_k] ) <= x[i_k]]

            constraints += [cp.real(cp.trace(M)) >= 1]
            constraints += [x >= 0]
            constraints += [M >> 0]
            prob = cp.Problem(Objective, constraints)
            cvx_obj = prob.solve(solver=cp.MOSEK, verbose=False)

            x_cur = x.value
            M_cur = M.value

            err = np.abs(cvx_obj-cur_obj)
            x_abs = np.abs(x_cur)
            if param["verbosity"]>3:
                print(f"iter:{i_iter} err:{err} obj: {cvx_obj}")
            #print(M.value)
            cur_obj = cvx_obj
            if err < 1e-9 or cvx_obj < 1e-7:
                break

            x_p = np.expand_dims(x_partial(x_cur, c_k),axis=1)
            M_p = M_partial(M_cur)

        u, singular_value, v = LA.svd(M.value)
        if singular_value[1:].sum() < 1e-6:
            break

    ind = x.value.argsort()

    for i_k in np.arange(K):
        active_user_num = K-i_k
        active_users = ind[:active_user_num]
        H_part = H[:,active_users]
        m, feasibility= feasibility_DC(H_part, gamma, param)
        if feasibility:
            num_of_users = active_user_num
            break
        if ~feasibility:
            num_of_users = 0
            active_users = []
            m = []
    return m, num_of_users, active_users


def user_selection_SDR_L1(H, gamma, param):
    N,K = H.shape 
    M = cp.Variable((N,N),hermitian=True)
    x = cp.Variable(K) 
    obj = cp.sum(x)
    Objective = cp.Minimize(obj)

    constraints = []    
    for i_k in np.arange(K):       
        constraints += [cp.real(cp.trace(M))-gamma*cp.real((H[:,i_k].transpose().conjugate() @ M)@H[:,i_k] ) <= x[i_k]]

    constraints += [cp.real(cp.trace(M)) >= 1]
    constraints += [x >= 0]
    constraints += [M >> 0]
    prob = cp.Problem(Objective, constraints)
    cvx_obj = prob.solve(solver=cp.MOSEK, verbose=False)

    ind = x.value.argsort()

    for i_k in np.arange(K):
        active_user_num = K-i_k
        active_users = ind[:active_user_num]
        H_part = H[:,active_users]
        m, feasibility= feasibility_SDR(H_part, gamma, param)
        if feasibility:
            num_of_users = active_user_num
            break
        if ~feasibility:
            num_of_users = 0
            active_users = []
            m = []
    return m, num_of_users, active_users