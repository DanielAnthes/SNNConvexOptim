import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# define a problem to solve
def f(x, a, b, c):
    '''
    evaluates a quadratic polynomial at location x
    expects x,a,b to be column vectors and c to be a scalar constant
    '''
    return (x.T**2 @ a) + (x.T @ b) + c 


def fprime(x, a, b):
    '''
    evaluates the gradient of quadratic polynomial
    '''
    return ((2 * x) * a) + b


# define gradient descent procedure
def grad_descent(x_init, nsteps, eta, fprime):
    dim = x_init.shape[0]
    trajectory = np.zeros((nsteps + 1, dim))
    trajectory[0,:] = x_init.squeeze()
    x = x_init
    for i in range(nsteps):
        x -= eta * fprime(x)
        trajectory[i+1,:] = x.squeeze()
    return x, trajectory


# define a range for evaluation and plotting
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X1, X2 = np.meshgrid(x1, x2)

x = np.array([X1.flatten(), X2.flatten()])
a = np.array([[3.], [4.]])
b = np.array([[1.],[1.]])
c = 0

# evaluate function at each location
y = f(x, a, b, c)
Y = np.reshape(y, X1.shape)

# define some constraints


# plot surface
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis')
plt.show()


# solve unconstrained problem with gradient descent and plot
x_init = np.array([[10.],[9.]])

x_min, trajectory = grad_descent(x_init, 3000, 0.01, lambda x: fprime(x, a, b))

plt.figure()
plt.contourf(X1, X2, Y)
plt.plot(trajectory[:,0], trajectory[:,1], color='black', marker='.')
plt.show()

# add some constraints
