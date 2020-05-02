import matplotlib.pyplot as plt
import numpy as np

from Field import Field
from GMRF import GMRF
import numpy as np


field = Field.FieldEx3()
dim = np.array([20, 20])
res = np.array([45, 45])

GMRF = GMRF.Field2d(dim, res)

# measurements on grid
mx = 12
my = 12
sig = 0    # noise variance of measurements.

xp = np.linspace(0, dim[0], mx)
yp = np.linspace(0, dim[1], my)

# random measurement locations p
p = np.zeros((mx * my, 2))
for i in range(my):
    for j in range(mx):
        p[my*i+j, :] = np.array([xp[j], yp[i]])

y = np.zeros((mx * my, 1))
ym = np.zeros((mx * my, 1))

for i in range(mx * my):  # real field values
    y[i] = field.value(p[i, 0], p[i, 1])

ym = y + sig*np.random.randn(mx * my, 1)  # add measurement noise

# GMRF calculation
Qinv = np.linalg.inv(GMRF.Q)
M = GMRF.mapping(p)
print(M)
print(ym)



mu_MRF = np.dot(Qinv, np.dot(M.T, np.dot(np.linalg.inv(sig*np.eye(mx * my) + np.dot(M, np.dot(Qinv, M.T))), ym)))
print(np.dot(M, mu_MRF))
mu_matrix = mu_MRF.reshape(res[1], res[0])

# Field metric
metric = y.reshape(-1, 1) - np.dot(M, mu_MRF).reshape(-1, 1)

# Plot
plt.figure(1)
#plt.imshow(field_arr, cmap='viridis')
plt.imshow(field.field_arr, extent=(np.amin(0), np.amax(20), np.amin(20), np.amax(0)), aspect='auto')
plt.scatter(p[:, 0], p[:, 1])
plt.colorbar()

plt.figure(2)
plt.imshow(mu_matrix, extent=(np.amin(0), np.amax(20), np.amin(20), np.amax(0)), aspect='auto')
plt.scatter(p[:, 0], p[:, 1])
#plt.imshow(mu_matrix, cmap='viridis')
plt.colorbar()

plt.figure(3)
plt.plot(range(len(metric)), metric)

plt.show()
