from util import plot_neuron_voltages, plot_raster, evaluate_constraints, plot_line_2D
import numpy as np
import matplotlib.pyplot as plt
import quadprog

def dx(x):
    '''
    we take the input to be constant
    '''
    return np.zeros(x.shape)

def dy(lamb, y, b, s, D):
    return -lamb * y - b + (s.T @ D).T

def dV(F, G, D, V, lamb, x, dx, s, b):
    return  -lamb * V + (F @ (lamb * x + dx)) - (G @ (s.T @ D).T) + (G @ b)

def dr(lamb, r, s):
    return -lamb * r + s

def SNN_trial(lamb, x, b, s, F, G, D, V, r, eta, nsteps):
    voltages = np.zeros((nsteps, len(V)))
    insta_fr = np.zeros((nsteps, len(V)))
    spikes = np.zeros((nsteps, len(V)))


    for i in range(nsteps):
        # check for spikes
        s = V >= T
        xprime = dx(x)
        vprime = dV(F, G, D, V, lamb, x, xprime, s, b)
        rprime = dr(lamb, r, s)

        V += eta * vprime
        r += eta * rprime

        voltages[i] = V.squeeze()
        insta_fr[i] = r.squeeze()
        spikes[i] = s.squeeze()

    return voltages, insta_fr, spikes
        

##################
### PARAMETERS ###
##################

F = np.array([[1,1],  # forward weights
              [3,-3], 
              [2,-1]])
G = np.array([[1,2],  # recurrent weights
              [-.3,1.3],
              [1, .6]])
T = np.array( [[6.],  # thresholds
               [7.],
               [7.]])
n_constraints = F.shape[0]
x = np.array([[5.],  # input to the problem
              [3.]])
b = np.ones((2,1))  # linear weight for problem
nsteps = 10000  # number of optimization steps
eta = 0.001  # learning rate
lamb = 1  # quadratic weight for problem
D = G / np.sqrt(G[:,0]**2 + G[:,1]**2)[:,None]  # normalize bounce vector to unit length
                                                # orthogonal to constraint boundaries
D *= 1000  # scale up D so that bounces become visible (has to counteract learning rate)
V = np.array([[3.],
              [3.],
              [3.]])
s = np.array([[False],
              [False],
              [False]])
r = np.zeros((3,1))

voltages, insta_fr, spikes = SNN_trial(lamb, x, b, s, F, G, D, V, r, eta, nsteps)

plt.figure()
plt.subplot(211)
plot_neuron_voltages(voltages, T, eta)
plt.legend()
plt.subplot(212)
plot_raster(voltages, T.squeeze(), eta)

# plot in 2D space

trajectory = (insta_fr @ D).T - (b / lamb)

ndim = 2
qp_G = np.eye(ndim) * lamb
q = b.squeeze()

qp_a = -q
qp_C = G.T
qp_b = -(T - (F @ x)).squeeze()
meq = 0

qp_y = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


# define a problem to solve
def f(y, lamb, b):
    '''
    evaluates a quadratic polynomial at location x
    expects x,a,b to be column vectors and c to be a scalar constant
    '''
    return ((lamb / 2) * (y.T @ y)) + (y.T @ b)


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

ax.scatter(trajectory[0,-10:], trajectory[1,-10:], color='red')
ax.scatter([qp_y[0]], [qp_y[1]], marker='*', color='black', s=200, label='solver y*')
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.xlabel('y1')
plt.ylabel('y2')
plt.legend()

plt.show()

