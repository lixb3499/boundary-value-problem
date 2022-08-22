import numpy as np
from matplotlib import pyplot as plt

def f_(x):
    return np.pi**2 *np.sin(np.pi*x)*np.cosh(np.sin(np.pi*x))

def u_(x):
    return np.sinh(np.sin(np.pi*x))

n=20
u=w=b=np.ones(n-1)

def unn(x):
    return x*(1-x)*(sum(u*np.sin(w*x+b)))

print(unn(0.5))
h=1/(n+1)
x=np.linspace(h, 1-h,n)
f=f_(x)


