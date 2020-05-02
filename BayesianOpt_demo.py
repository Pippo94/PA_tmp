import matplotlib.pyplot as pl
import numpy as np
import time
from GMRF import GMRF, GP
from scipy.stats import norm

# Field to explore
field = lambda x: 2*np.sin(0.5*x)+0.2*np.sin(2.5*x)

xf = np.linspace(0, 10, 100)

# measurements
#xm = np.linspace(0, 10, 2)  # locations
#xm = np.insert(xm, 2, xm[1]+0.1)

xm1 = np.random.uniform(0, 10, 3)  # random locations
xm2 = xm1

sig = 0.05  # for noise
ym1 = field(xm1).reshape(-1, 1) + sig*np.random.randn(len(xm1), 1)  # measurements with noise
ym2 = ym1

gp1 = GP.GPR(sig)
gp1.train(xm1, ym1)

gp2 = GP.GPR(sig)
gp2.train(xm2, ym2)


for i in range(10):
    # do GPR
    mu1, s1 = gp1.predict(xf)
    mu2, s2 = gp1.predict(xf)

    # find bayesian opt
    opt1 = GP.BayOpt(mu1, s1, case='PI')
    opt2 = GP.BayOpt(mu2, s2)

    # plot
    fig, ([ax1, ax2], [ax3, ax4]) = pl.subplots(2, 2)
    fig.suptitle('Bayesian optimisation')

    # plot GPR PI
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-6, 6])
    ax1.plot(xf, field(xf), 'r--')
    ax1.plot(xm1, ym1, 'bo')
    ax1.plot(xf, mu1)
    ax1.fill_between(xf.flat, (mu1-3*s1).flat, (mu1+3*s1).flat, color="#dddddd")

    xm1 = np.append(xm1, xf[opt1.next])
    ym1 = np.append(ym1, field(xf[opt1.next]) + sig*np.random.randn())

    # plot GPR EI
    ax2.set_xlim([0, 10])
    ax2.set_ylim([-6, 6])
    ax2.plot(xf, field(xf), 'r--')
    ax2.plot(xm2, ym2, 'bo')
    ax2.plot(xf, mu2)
    ax2.fill_between(xf.flat, (mu2 - 3 * s2).flat, (mu2 + 3 * s2).flat, color="#dddddd")

    xm2 = np.append(xm2, xf[opt2.next])
    ym2 = np.append(ym2, field(xf[opt2.next]) + sig * np.random.randn())

    # plot acqu PI
    ax3.set_xlim([0, 10])
    ax3.set_ylim([-0.1, 1])
    ax3.plot(xf, opt1.acqu.reshape(-1, 1), 'g')
    ax3.plot(xf[opt1.next], opt1.acqu[opt1.next], 'gx')

    ax4.set_xlim([0, 10])
    ax4.set_ylim([-0.1, 1])
    ax4.plot(xf, opt2.acqu.reshape(-1, 1), 'g')
    ax4.plot(xf[opt2.next], opt2.acqu[opt2.next], 'gx')

    pl.show()
    pl.close()

    gp1.train(xm1.reshape(-1, 1), ym1.reshape(-1, 1))
    gp2.train(xm2.reshape(-1, 1), ym2.reshape(-1, 1))

    time.sleep(0.5)


