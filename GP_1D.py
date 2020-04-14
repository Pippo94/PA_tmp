import numpy as np
import matplotlib.pyplot as pl


def kernel(a, b):
    dist = 0.7
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * (1 / dist) * sqdist)


realf = lambda x: 2*np.sin(0.5*x)+0.2*np.sin(2.5*x)

N = 10         # number of training points.
n = 100         # number of test points.
s = 0.00005    # noise variance.

# GPR
X_training = np.random.uniform(-5, 5, size=(N, 1))  # N random testpoints
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
Y_training = realf(X_training) + s*np.random.randn(N, 1)  # without measurement noise

K = kernel(X_training, X_training)
Lk = np.linalg.cholesky(K+s*np.eye(N))

Ks = kernel(X_training, X_test)
Kss = kernel(X_test, X_test)

mu = np.dot(Ks.T, np.linalg.solve(Lk.T, np.linalg.solve(Lk, Y_training)))
v = np.linalg.solve(Lk, Ks)
E = Kss - np.dot(v.T, v)
sq = np.sqrt(np.diag(E)).reshape(-1, 1)

# plot real function
pl.figure(1)
pl.plot(X_test, realf(X_test))
pl.title('Original fct')
pl.axis([-5, 5, -3, 3])
pl.savefig('realfct.png', bbox_inches='tight')

# plot random
pl.figure(2)
pl.plot(X_test, (0.5*np.random.randn(n, 1)))
pl.title('Any Random numbers')
pl.axis([-5, 5, -3, 3])
pl.savefig('rndsmp.png', bbox_inches='tight')

# plot one draw GPR
pl.figure(3)
pl.plot(X_test, mu)
pl.plot(X_training,realf(X_training), 'rx')
pl.plot(X_test, realf(X_test), '--r')
pl.gca().fill_between(X_test.flat, (mu-3*sq).flat, (mu+3*sq).flat, color="#dddddd")
pl.title('one draw GPR')
pl.axis([-5, 5, -3, 3])
pl.savefig('gpr1.png', bbox_inches='tight')


pl.show()



