import matplotlib.pyplot as plt
import numpy as np
from math import exp
from numpy import linalg as LA
from sklearn import gaussian_process
from scipy import linalg as l
import pandas as pd


def f(x):
    return x*np.sin(x)

X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

y = f(X).ravel()

x = np.atleast_2d(np.linspace(0, 10, 50)).T
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, y)

y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)


plt.plot(X, y, 'r+')
plt.plot(x, y_pred, 'b')

#different kernel can be used...
def GaussianKernel(v1, v2, sigma,delta):
    if v2.shape[0] > v1.shape[0]:
        return GaussianKernel(v2, v1, sigma, delta).T

    v = np.zeros((v1.shape[0], v2.shape[0]))
    for i in range(0, v1.shape[0]):
        for j in range(i, v2.shape[0]):
            v[i][j] = sigma*2*exp(-LA.norm(v1[i]-v2[j], 2)**2/2)
            if i == j and delta != 0:
                v[i][j] +=  delta**2
            elif i!=j and j<v1.shape[0] and i<v2.shape[0]:
                v[j][i] = v[i][j]

    return v


kxx=GaussianKernel(X,X,2,.0)
ksx=GaussianKernel(x,X,2,.0)
lainv = LA.inv(kxx)
fs=np.dot(np.dot(ksx,lainv), y)
kss = GaussianKernel(x,x,2,0)
ss = kss - np.dot( np.dot(ksx,lainv),ksx.T)
print (np.linalg.eigvalsh(ss))
#cs = l.cholesky(ss, lower=True)
#z=np.ones(cs.shape[0])
#er = [x for x,y in np.dot(cs, z)]
#df = pd.DataFrame({'x': x, 'y':fs, 'yl': fs-2*er, 'yu': fs+2*er})
#plt.fill_between(df['x'], df['yl'], df['yu'])
plt.show()
