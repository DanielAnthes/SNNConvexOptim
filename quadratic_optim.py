import numpy as np
import matplotlib
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


def linear_ineq_constraint(x, y, m, b, less_eq=True):
    '''
    implement a linear inequality constraint, return True if constraint    
    is satisfied, False otherwise
    '''
    return  m * x + b <= y if less_eq else m * x + b >= y


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


def constrained_grad_descent(x_init, nsteps, eta, fprime, constraints, bounce_dirs):
    dim = x_init.shape[0]
    trajectory = np.zeros((nsteps + 1, dim))
    trajectory[0,:] = x_init.squeeze()
    x = x_init
    for i in range(nsteps):
        # first check for constraints
        constraintViolated = False
        
        for j, constraint in enumerate(constraints):
            if not constraint(*x.squeeze()):
                constraintViolated = True
                break
        
        # if constraint is violated bounce
        if constraintViolated:
            x += bounce_dirs[j][:, None]

        # else take regular optimization step
        else:
            if i < nsteps - 1:  # exclude last step to avoid 
                                # jumping into the constrained 
                                # area on the last step with no 
                                # chance to bounce back
                x -= eta * fprime(x)
        
        trajectory[i+1,:] = x.squeeze()
    
    return x, trajectory


###########################################
# set up constrained optimization problem #
###########################################

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

# add some constraints
# first constraint
m1 = 1
b1 = 3

constraint1 = [1 if linear_ineq_constraint(a, b, m1, b1) else 0 for a, b in x.T]
constraint1 = np.reshape(constraint1, X1.shape)

# second constraint
m2 = -1
b2 = 3

constraint2 = [1 if linear_ineq_constraint(a, b, m2, b2, less_eq=False) else 0 for a, b in x.T]
constraint2 = np.reshape(constraint2, X1.shape)

x_init = np.array([[-10.],[-5.]])

bounce_size = 0.03
constraints = [lambda x, y: linear_ineq_constraint(x, y, m1, b1), lambda x, y: linear_ineq_constraint(x, y, m2, b2, less_eq=False)]
bounce_dirs = np.array([[-1, 1], [-1, -1]]) * bounce_size

x_min, constrained_trajectory = constrained_grad_descent(x_init, 3000, 0.01, lambda x: fprime(x, a, b), constraints, bounce_dirs)
print(x_min, constrained_trajectory)
# plot
constraint_cmap = matplotlib.colors.ListedColormap([[186/255, 0, 0, 1], [1, 1, 1, 0]])  # transparent for 0, filled in for 1

plt.figure()
plt.contourf(X1, X2, Y, cmap='binary')
plt.plot(x1, m1 * x1 + b1, color='red')
plt.plot(x1, m2 * x1 + b2, color='red')
plt.contourf(X1, X2, constraint1, cmap=constraint_cmap, alpha=.3)
plt.contourf(X1, X2, constraint2, cmap=constraint_cmap, alpha=.3)
plt.plot(constrained_trajectory[:,0], constrained_trajectory[:,1], color='black', marker='.')
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.show()

# # define a range for evaluation and plotting
# x1 = np.linspace(-10, 10, 100)
# x2 = np.linspace(-10, 10, 100)

# X1, X2 = np.meshgrid(x1, x2)

# x = np.array([X1.flatten(), X2.flatten()])
# a = np.array([[3.], [4.]])
# b = np.array([[1.],[1.]])
# c = 0

# # evaluate function at each location
# y = f(x, a, b, c)
# Y = np.reshape(y, X1.shape)

# define some constraints


# plot surface
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X1, X2, Y, cmap='viridis')
# plt.show()

# # solve unconstrained problem with gradient descent and plot
# x_init = np.array([[10.],[9.]])

# x_min, trajectory = grad_descent(x_init, 3000, 0.01, lambda x: fprime(x, a, b))

# plt.figure()
# plt.contourf(X1, X2, Y)
# plt.plot(trajectory[:,0], trajectory[:,1], color='black', marker='.')
# plt.show()


