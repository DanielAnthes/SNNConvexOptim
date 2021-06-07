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
        y += eta * dy
        trajectory[i+1] = y.squeeze()

    return trajectory, y

def constrained_optim(lamb, x, y, b, iter, eta, D, F, G, T):
    trajectory = np.zeros((iter + 1, len(y)))
    trajectory[0] = y.squeeze()

    voltages = np.zeros((iter + 1, F.shape[0]))
    
    for i in range(iter):
        voltages[i] = ((F @ x) - (G @ y)).squeeze()
        s = ~ evaluate_constraints(F, x, G, y, T)  # record spike for violated constraint
        if s.any():
            dy = (s.T @ D).T
        else:
            dy = (-lamb * y) - b 
        # dy = (-lamb * y) - b + (s.T @ D).T
        assert dy.shape == (2,1), f"actual shape: {dy.shape}, {s.T.shape, D.shape}"
        y += eta * dy
        trajectory[i+1] = y.squeeze()
    
    voltages[-1] = ((F @ x) - (G @ y)).squeeze()
    return trajectory, voltages, y

def plot_neuron_voltages(voltages, thresholds, eta):
    n_neurons = len(thresholds)
    t = np.array(range(voltages.shape[0])) * eta

    for n in range(n_neurons):
        vs = voltages[:,n]
        plt.plot(t, vs, label=f"neuron {n}")
        plt.plot(t, [thresholds[n]] * len(t), label=f"threshold {n}")


def plot_raster(voltages, thresholds, eta):
    spikes = np.asarray(voltages > thresholds[None,:]).nonzero()
    plt.scatter(spikes[0] * eta, spikes[1])

################
###   INIT   ###
################

F = np.array([[1,1],  # forward weights
              [3,-3], 
              [2,-1]])

G = np.array([[1,2],  # recurrent weights
              [-.3,1.3],
              [1, .6]])

T = np.array( [[6],  # thresholds
               [7],
               [7]])

n_constraints = F.shape[0]

x = np.array([[5],  # input to the problem
              [3]])


y = np.array([[-5.], [10.]])  # initial position
b = np.ones((2,1))  # linear weight for problem
nsteps = 10000  # number of optimization steps
eta = 0.001  # learning rate
lamb = 1  # quadratic weight for problem

D = G / np.sqrt(G[:,0]**2 + G[:,1]**2)[:,None]  # normalize bounce vector to unit length
                                                # orthogonal to constraint boundaries
D *= 1000  # scale up D so that bounces become visible (has to counteract learning rate)

trajectory_constrained, voltages_constrained, y_optim_constrained = constrained_optim(lamb, x, y, b, nsteps, eta, D, F, G, T)

################
### Plotting ###
################

y1 = np.linspace(-10, 10, 100)
y2 = np.linspace(-10, 10, 100)

Y1, Y2 = np.meshgrid(y1, y2)

Y = np.array([Y1.flatten(), Y2.flatten()])

# evaluate function at each location
z = np.zeros(Y.shape[1]).T

for i, y in enumerate(Y.T):
    z[i] = f(y[:,None], 1, np.ones((2,1)))

Z = np.reshape(z, Y1.shape)

y2 = plot_line_2D(F, G, T, x, Y[0])  # constraint boundaries 

feasible = np.empty((Y.shape[1], n_constraints))  # feasible region

for i, y in enumerate(Y.T):
    y = y[:,None]
    feasible[i] = evaluate_constraints(F,x,G,y,T).squeeze()

feasible_region = feasible.all(axis=1)
feasible_region = feasible_region.reshape(Y1.shape)
infeasible_region = ~ feasible_region

fig = plt.figure()
ax = fig.add_subplot()
ax.contour(Y1,Y2,Z, cmap='binary', linestyles='dashed')
ax.contourf(Y1, Y2, infeasible_region, cmap='binary', alpha=.1)
for i in range(n_constraints):
    ax.plot( Y[0], y2[i,:], label=f"constraint {str(i)}")

ax.plot(trajectory_constrained[:,0], trajectory_constrained[:,1], color='red')
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.legend()

plt.figure()
plt.subplot(211)
plot_neuron_voltages(voltages_constrained, T.squeeze(), eta)
plt.legend()
plt.subplot(212)
plot_raster(voltages_constrained, T.squeeze(), eta)

plt.show()

