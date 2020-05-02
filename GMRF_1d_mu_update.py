# GMRF 1D to GF comparision

import numpy as np
import matplotlib.pyplot as pl
import time


def kernel(a, b):  # kernel for GF, note that GMRF is based on Matern kernel
    dist = 0.7
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * (1 / dist) * sqdist)


# Field to explore
field = lambda x: 2*np.sin(0.5*x)+0.2*np.sin(2.5*x)


N = 25         # number of training/measurement points.
n = 500         # number of test/Field points.
sig = 0.005    # noise variance of measurements.

# Gaussian Field Regression
X_training = np.linspace(-5, 5, N).reshape(-1, 1)  # N random testpoints
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
Y_training = field(X_training) + sig*np.random.randn(N, 1)  # measurements

# GMRF
N_MRF = 40
s = np.linspace(-5, 5, N_MRF)  # GMRF grid points
a = 2  # Precision Qii
Q = a * np.identity(N_MRF, int) - np.eye(N_MRF, k=1) - np.eye(N_MRF, k=-1)

# Neumann Boundary condition
Q[0, 0] = Q[0, 0]-0.9  # linkes Ende
Q[N_MRF-1, N_MRF-1] = Q[N_MRF-1, N_MRF-1]-0.9  # rechtes Ende

# plot one draw GPR

for i in range(len(X_training)):

    K = kernel(X_training[0:i+1], X_training[0:i+1])
    Lk = np.linalg.cholesky(K+sig*np.eye(i+1))

    Ks = kernel(X_training[0:i+1], X_test)
    Kss = kernel(X_test, X_test)

    mu_av = np.sum(Y_training[0:i+1])/len(Y_training[0:i+1])

    mu_GF = mu_av**np.ones(n).reshape(-1, 1) + np.dot(Ks.T, np.linalg.solve(Lk.T, np.linalg.solve(Lk, Y_training[0:i+1] - mu_av*np.ones(i+1).reshape(-1, 1))))  # GF expectation

    v = np.linalg.solve(Lk, Ks)  # for better calculation
    Sigma_GF = Kss - np.dot(v.T, v)

    sq = np.sqrt(np.diag(Sigma_GF)).reshape(-1, 1)

    p = X_training[0:i+1]

    # measurement points, same as GF
    # Continous mapping
    M = np.zeros((np.size(p), np.size(s)))
    L = s[1]-s[0]
    ind = np.searchsorted(s, p)

    for j in range(len(p)):  # constructing M for mapping
        M[j, ind[j]-1] = np.linalg.norm(s[ind[j]-1] - p[j])/L
        M[j, ind[j]] = np.linalg.norm(s[ind[j]] - p[j])/L

    Qinv = np.linalg.inv(Q)
    mu_MRF = mu_av*np.ones(N_MRF).reshape(-1, 1) + np.dot(Qinv, np.dot(M.T, np.dot(np.linalg.inv(sig*np.eye(i+1) + np.dot(M, np.dot(Qinv, M.T))), Y_training[0:i+1] - mu_av*np.ones(i+1).reshape(-1, 1))))

    pl.figure(1)
    pl.plot(X_test, field(X_test), '--r')
    pl.plot(X_test, mu_GF)
    pl.plot(X_training[0:i+1], Y_training[0:i+1], 'rx')
    pl.plot(s, mu_MRF, 'kx')
    pl.gca().fill_between(X_test.flat, (mu_GF-3*sq).flat, (mu_GF+3*sq).flat, color="#dddddd")
    pl.title('one draw GPR')
    pl.axis([-5, 5, -3, 3])
    pl.show()
    pl.close(1)
    print(i)
    time.sleep(0.5)












