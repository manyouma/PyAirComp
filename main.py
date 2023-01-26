import numpy as np
from numpy import linalg as LA
import cvxpy as cp
from scipy.io import savemat, loadmat
from feasibility_DC import feasibility_DC
from user_selection_DC import user_selection_DC, user_selection_SDR_L1
import matplotlib.pyplot as plt


N = 6      # Number of antennas
K = 20     # Number of users

gamma_set = np.arange(-10,35,5)
gamma_num = len(gamma_set)
test_num = 20
user_num_DC = np.zeros((gamma_num, test_num))
user_num_SDR_l1 = np.zeros((gamma_num, test_num))

param = {
    "maxIter": 50, 
    "epsilon": 0.0001,
    "verbosity": 2,
    "p": 0.5
}

# Construct a random matrix
H = 1j*np.random.randn(N,K)
for i_user in np.arange(K):
    H[:,i_user] = np.random.randn(N,)/np.sqrt(2)+1j*np.random.randn(N,)/np.sqrt(2)


for i_gamma in np.arange(gamma_num):
    gamma = np.power(10,gamma_set[i_gamma]/10)
    if param["verbosity"] > 1:
            print(f'i_Gamma: {i_gamma}, gamma:{gamma}')
    for i_test in np.arange(test_num):
        try:
            m, num_of_users, active_users = user_selection_DC(H, gamma, param)
            user_num_DC[i_gamma, i_test] = num_of_users
            m, num_of_users, active_users = user_selection_SDR_L1(H, gamma, param)
            user_num_SDR_l1[i_gamma, i_test] = num_of_users
            if param["verbosity"] > 1:
                print(f'Working on test case: {i_test}')
        except:
            print("error occur")

# Make the plots
plt.plot(gamma_set, user_num_DC.mean(axis=1), label='DC')
plt.plot(gamma_set, user_num_SDR_l1.mean(axis=1), label='SDR')
plt.legend()