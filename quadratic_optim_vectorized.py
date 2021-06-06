import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_line_2D(f,g,t,x,y):
    return ( (f @ x) - (g[:,0][:,None] * y) - t) / g[:,1][:,None]

def evaluate_constraints(F,x,G,y,T):
    return (F @ x) - (G @ y) <= T

# define a problem to solve
def f(y, lamb, b):
    '''
    evaluates a quadratic polynomial at location x
    expects x,a,b to be column vectors and c to be a scalar constant
    '''
    return ((lamb / 2) * (y.T @ y)) + (y.T @ b)

def fprime(x, a, b):
    '''
    evaluates the gradient of quadratic polynomial
    '''
    return ((2 * x) * a) + b

def optim(lamb, y, b, iter, eta):
    trajectory = np.zeros((iter + 1, len(y)))
    trajectory[0] = y.squeeze()

    for i in range(iter):
        dy = (-lamb * y) - b
        print(dy)
        y += eta * dy
        trajectory[i+1] = y.squeeze()

    return trajectory, y

def constrained_optim(lamb, x, y, b, iter, eta, D, F, G, T):
    trajectory = np.zeros((iter + 1, len(y)))
    trajectory[0] = y.squeeze()
    
    for i in range(iter):
        s = evaluate_constraints(F, x, G, y, T)
        dy = (-lamb * y) - b + (s.T @ D).T
        assert dy.shape == (2,1), f"actual shape: {dy.shape}, {s.T.shape, D.shape}"
        y += eta * dy
        trajectory[i+1] = y.squeeze()

    return trajectory, y

y1 = np.linspace(-10, 10, 100)
y2 = np.linspace(-10, 10, 100)

Y1, Y2 = np.meshgrid(y1, y2)

Y = np.array([Y1.flatten(), Y2.flatten()])

# define parameters
F = np.array([[1,1],
              [3,-3], 
              [2,-1]])


G = np.array([[-3,2],
              [.1,1],
              [1,1]])

T = np.array( [[-3],
               [3],
               [5]])

n_constraints = F.shape[0]

x = np.array([[3],
              [1]])



# evaluate function at each location

z = np.zeros(Y.shape[1]).T

for i, y in enumerate(Y.T):
    z[i] = f(y[:,None], 1, np.ones((2,1)))

Z = np.reshape(z, Y1.shape)

################
###   INIT   ###
################

y = np.array([[-5.], [10.]])
trajectory, y_final = constrained_optim(1,x, y, np.ones((2,1)), 10000, 0.01, G, F, G, T)

################
### Plotting ###
################

y2 = plot_line_2D(F, G, T, x, Y[0])

feasible = np.empty((Y.shape[1], n_constraints)) 

for i, y in enumerate(Y.T):
    y = y[:,None]
    feasible[i] = evaluate_constraints(F,x,G,y,T).squeeze()

feasible_region = feasible.all(axis=1)
feasible_region = feasible_region.reshape(Y1.shape)
infeasible_region = ~ feasible_region

fig = plt.figure()
ax = fig.add_subplot()
ax.contourf(Y1,Y2,Z, cmap='binary')
ax.contourf(Y1, Y2, infeasible_region, cmap='binary', alpha=.3)
for i in range(n_constraints):
    ax.plot( Y[0], y2[i,:], label=f"constraint {str(i)}")

ax.plot(trajectory[:,0], trajectory[:,1], color='black')
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.legend()
plt.show()

