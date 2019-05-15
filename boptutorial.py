from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from sklearn import gaussian_process as gp
from scipy.stats import norm

import numpy as np

def g(x):
    return x*np.sin(x)


def acq(x, mu, sigma):
    return (min(g(x)) - mu)*norm.cdf(x, mu, sigma) + sigma*norm.pdf(x, mu, sigma)

#data = np.array([-8, -1, 2.5, 7.5]) 
x = np.linspace(-9,9,100)
data = np.random.uniform(-9,9,4)

gaussp = gp.GaussianProcessRegressor().fit(data.reshape(-1,1),g(data))
y_gp, sig = gaussp.predict(x.reshape(-1,1),return_std=True)
plt.plot(x,g(x), color='k', linestyle='--', alpha=.7)
plt.plot(data, g(data),'.',color='r')
plt.plot(x, y_gp, color='orange')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_gp - 1.96*sig,
         (y_gp + 1.96*sig)[::-1]]),
    alpha=.3, fc='orange', ec=None)
plt.plot(x, acq(x, y_gp, sig),color='purple')
plt.show()


