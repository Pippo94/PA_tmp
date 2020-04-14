import matplotlib.pyplot as plt
import numpy as np


class Field:
    r = 3  # range factor

    def __init__(self, n, xy_min, xy_max):
        self.h = 1.5 * np.random.randn(n, 1)  # spot temperatures
        self.p = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))  # spot locations

    def v(self, x, y):
        v = 0
        for i in range(len(self.h)):
            v = v + self.h[i] * np.exp(-np.linalg.norm(self.p[i, :].reshape(-1, 1) - np.array([x, y]).reshape(-1, 1)) / self.r)
        return v


# field generation (heatmap)

ns = 13  # number of spots
xy_min = [0, 0]  # lower field boundaries
xy_max = [20, 20]  # upper field boundaries

x = np.linspace(xy_min[0], xy_max[0], 100)
y = np.linspace(xy_min[1], xy_max[1], 100)

hmap = Field(ns, xy_min, xy_max)
field_arr = np.zeros((100, 100))

for i in range(len(y)):
    for j in range(len(x)):
        field_arr[i, j] = hmap.v(x[j], y[i])

# measurements
m = 24  # number of measurements
sig = 0.00005    # noise variance of measurements.

p = np.random.uniform(low=xy_min, high=xy_max, size=(m, 2))  # random measurement locations p
ym = np.zeros((m, 1))

for i in range(m):  # measurement values with noise
    ym[i] = hmap.v(p[i, 0], p[i, 1]) + sig*np.random.randn()

# build Q matrix for GMRF
nx = 50
ny = 50

a = 4
Q = a * np.eye(nx * ny)
V = np.arange(nx * ny).reshape(ny, nx)  # GMRF vertices in array grid

for i in V.reshape(-1, 1):
    ind = np.argwhere(V == i)[0]

    if ind[0] - 1 >= 0:  # upper
        Q[i, V[ind[0] - 1, ind[1]]] = -1
    else:
        Q[i, i] = Q[i, i] - 1  # in case of boundary

    if ind[0] + 1 < ny:  # lower
        Q[i, V[ind[0] + 1, ind[1]]] = -1
    else:
        Q[i, i] = Q[i, i] - 1

    if ind[1] - 1 >= 0:  # left
        Q[i, V[ind[0], ind[1] - 1]] = -1
    else:
        Q[i, i] = Q[i, i] - 1

    if ind[1] + 1 < nx:  # right
        Q[i, V[ind[0], ind[1] + 1]] = -1
    else:
        Q[i, i] = Q[i, i] - 1

# build M-Matrix

xs = np.linspace(xy_min[0], xy_max[0], nx)
ys = np.linspace(xy_min[1], xy_max[1], ny)

ind_x = np.searchsorted(xs, p[:, 0])
ind_y = np.searchsorted(ys, p[:, 1])

M = np.zeros((m, nx*ny))
w = xs[1] - xs[0]
l = ys[1] - ys[0]

for i in range(m):
    M[i, V[ind_y[i] - 1, ind_x[i] - 1]] = np.linalg.norm(xs[ind_x[i] - 1] - p[i, 0]) * np.linalg.norm(
        ys[ind_y[i] - 1] - p[i, 1]) / (w * l)  # for phi s1
    M[i, V[ind_y[i] - 1, ind_x[i]]] = np.linalg.norm(xs[ind_x[i]] - p[i, 0]) * np.linalg.norm(
        ys[ind_y[i] - 1] - p[i, 1]) / (w * l)  # for phi s2
    M[i, V[ind_y[i], ind_x[i] - 1]] = np.linalg.norm(xs[ind_x[i] - 1] - p[i, 0]) * np.linalg.norm(
        ys[ind_y[i]] - p[i, 1]) / (w * l)  # for phi s3
    M[i, V[ind_y[i], ind_x[i]]] = np.linalg.norm(xs[ind_x[i]] - p[i, 0]) * np.linalg.norm(
        ys[ind_y[i]] - p[i, 1]) / (w * l)  # for phi s4

np.linalg.inv(Q)

# GMRF calculation
Qinv = np.linalg.inv(Q)
mu_MRF = np.dot(Qinv, np.dot(M.T, np.dot(np.linalg.inv(sig*np.eye(m) + np.dot(M, np.dot(Qinv, M.T))), ym)))

mu_matrix = mu_MRF.reshape(ny, nx)

plt.figure(1)
#plt.imshow(field_arr, cmap='viridis')
plt.imshow(field_arr, extent=(np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys)), aspect='auto')
plt.scatter(p[:, 0], p[:, 1])
plt.colorbar()
plt.figure(2)
plt.imshow(mu_matrix, extent=(np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys)), aspect='auto')
plt.scatter(p[:, 0], p[:, 1])
#plt.imshow(mu_matrix, cmap='viridis')
plt.colorbar()
plt.show()
