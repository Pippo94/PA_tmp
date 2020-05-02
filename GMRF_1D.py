# GMRF 1D to GF comparision

import numpy as np
import matplotlib.pyplot as pl


def kernel(a, b):  # kernel for GF, note that GMRF is based on Matern kernel
    dist = 0.7
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * (1 / dist) * sqdist)


# Field to explore
field = lambda x: 2*np.sin(0.5*x)+0.2*np.sin(2.5*x)


N = 14       # number of training/measurement points.
n = 500         # number of test/Field points.
sig = 0.00005    # noise variance of measurements.

# Gaussian Field Regression
X_training = np.random.uniform(-5, 5, size=(N, 1))  # N random testpoints
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
Y_training = field(X_training) + sig*np.random.randn(N, 1)  # measurements

K = kernel(X_training, X_training)
Lk = np.linalg.cholesky(K+sig*np.eye(N))

Ks = kernel(X_training, X_test)
Kss = kernel(X_test, X_test)


mu_GF = np.dot(Ks.T, np.linalg.solve(Lk.T, np.linalg.solve(Lk, Y_training)))  # GF expectation

v = np.linalg.solve(Lk, Ks)  # for better calculation
Sigma_GF = Kss - np.dot(v.T, v)

sq = np.sqrt(np.diag(Sigma_GF)).reshape(-1, 1)


# GMRF
N_MRF = 30
s = np.linspace(-5, 5, N_MRF)  # GMRF grid points
p = X_training  # measurement points, same as GF
a = 2  # Precision Qii
Q = a*np.identity(N_MRF, int) - np.eye(N_MRF, k=1) - np.eye(N_MRF, k=-1)

# Neumann Boundary condition
Q[0, 0] = Q[0, 0]-0.99  # linkes Ende
Q[N_MRF-1, N_MRF-1] = Q[N_MRF-1, N_MRF-1]-0.99  # rechtes Ende

# Continous mapping
M = np.zeros((np.size(p), np.size(s)))
L = s[1]-s[0]
ind = np.searchsorted(s, p)

for i in range(len(p)):  # constructing M for mapping
    M[i, ind[i]-1] = np.linalg.norm(s[ind[i]-1] - p[i])/L
    M[i, ind[i]] = np.linalg.norm(s[ind[i]] - p[i])/L

Qinv = np.linalg.inv(Q)
mu_MRF = np.dot(Qinv, np.dot(M.T, np.dot(np.linalg.inv(sig*np.eye(N) + np.dot(M, np.dot(Qinv, M.T))), Y_training)))

# plot one draw GPR
pl.figure(1)
pl.plot(X_test, mu_GF)
pl.plot(X_training, field(X_training), 'ro')
pl.plot(s, mu_MRF, 'kx')
pl.plot(X_test, field(X_test), '--r')
pl.gca().fill_between(X_test.flat, (mu_GF-3*sq).flat, (mu_GF+3*sq).flat, color="#dddddd")
pl.title('GPR / GMRF')
pl.axis([-5, 5, -3, 3])
pl.savefig('gpr1.png', bbox_inches='tight')


pl.show()












