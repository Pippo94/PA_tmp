import matplotlib.pyplot as pl
import numpy as np
from GMRF import GMRF, GP

# Field to explore
field = lambda x: 2*np.sin(0.5*x)+0.2*np.sin(2.5*x)

xf = np.linspace(0, 10, 100)

# measurements
xm = np.linspace(0, 10, 5)  # locations
#xm = np.insert(xm, 2, xm[1]+0.1)

#xm = np.random.uniform(0, 10, 36)  # random locations

sig = 0.005  # for noise
ym = field(xm).reshape(-1, 1) + sig*np.random.randn(len(xm), 1)  # measurements with noise

# GMRF
Mfield = GMRF.Field1d(10, 12)

M = Mfield.mapping(xm)
Q = Mfield.Q

Qinv = np.linalg.inv(Q)
B = np.linalg.inv(sig*np.eye(len(xm)) + np.dot(M, np.dot(Qinv, M.T)))
mu_MRF = np.dot(Qinv, np.dot(M.T, np.dot(B, ym)))
SIG = Qinv - np.dot(Qinv, np.dot(M.T, np.dot(B, np.dot(M, Qinv))))
var = np.sqrt(np.diag(SIG))

# Gaussian regression
GP = GP.GP(sig)
GP.train(xm, ym)
mu_GF, sq = GP.predict(xf)   # GF expectation


# Field metric
metric_ym = field(xm).reshape(-1, 1) - np.dot(M, mu_MRF)
metric_grid = field(Mfield.xs).reshape(-1, 1) - mu_MRF.reshape(-1, 1)


fig, (ax1, ax2) = pl.subplots(1, 2)
fig.suptitle('GMRF and Gaussian Regression')
ax1.set_xlim([0, 10])
ax1.set_ylim([-6, 6])
ax1.plot(xf, field(xf), 'r--')
ax1.plot(xm, ym, 'bo')
ax1.plot(Mfield.xs, mu_MRF, 'kx')
ax1.fill_between(Mfield.xs.flat, (mu_MRF.reshape(-1, 1)-3*var.reshape(-1, 1)).flat, (mu_MRF.reshape(-1, 1)+3*var.reshape(-1, 1)).flat, color="#dddddd")
#ax1.plot(Mfield.xs, mu_MRF.reshape(-1, 1)+3*var.reshape(-1, 1))
#ax1.plot(Mfield.xs, mu_MRF.reshape(-1, 1)-3*var.reshape(-1, 1))

ax2.set_xlim([0, 10])
ax2.set_ylim([-6, 6])
ax2.plot(xf, field(xf), 'r--')
ax2.plot(xm, ym, 'bo')
ax2.plot(xf, mu_GF)
fig.gca().fill_between(xf.flat, (mu_GF-3*sq).flat, (mu_GF+3*sq).flat, color="#dddddd")

pl.figure(2)
pl.plot(range(len(xm)), metric_ym)

pl.figure(3)
pl.plot(range(len(Mfield.xs)), metric_grid)

pl.show()
