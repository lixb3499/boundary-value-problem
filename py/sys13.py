from cProfile import label
from distutils.log import error
import numpy as np
from matplotlib import pyplot as plt

def f_(x):
    return np.pi**2 *np.sin(np.pi*x)*np.cosh(np.sin(np.pi*x))

def u_(x):
    return np.sinh(np.sin(np.pi*x))


fig, axes = plt.subplots(2, 3)
fig.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.30,hspace=0.35)
# fig.suptitle('solution')
axes_list1 = []
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes_list1.append(axes[i, j])
        
fig2, axes2 = plt.subplots(2, 3)
fig2.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.30,hspace=0.35)
# fig2.suptitle('error')
axes_list2 = []
for i in range(axes2.shape[0]):
    for j in range(axes2.shape[1]):
        axes_list2.append(axes2[i, j])

n_= [10,20,40,80,160,320]
for k in range(len(n_)):
    n=n_[k]
    h=1/(n+1)
    A=np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i == j:
                A[i-1,j-1] = 2/h**2 + np.pi**2 * (np.cos(np.pi*i*h))**2
            if abs(i-j) == 1:
                A[i-1,j-1] = -1/h**2
    
    x=np.linspace(h, 1-h,n)
    f=f_(x)
    g=np.linalg.solve(A,f)
    axes_list1[k].plot(x,g,label='differential solution')
    u=u_(x)
    axes_list1[k].plot(x,u,label='analytic solution')
    axes_list1[k].legend()
    axes_list1[k].set_title("n = %i"%n)
    e=np.log(abs(g-u))
    axes_list2[k].plot(x,e)
    axes_list2[k].set_title("n = %i"%n)
plt.show()
