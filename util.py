import numpy as np
import matplotlib.pyplot as plt


def plot_neuron_voltages(voltages, thresholds, eta):
    n_neurons = len(thresholds)
    t = np.array(range(voltages.shape[0])) * eta
    for n in range(n_neurons):
        vs = voltages[:,n]
        plt.plot(t, vs, label=f"neuron {n}")
        plt.plot(t, [thresholds[n]] * len(t), label=f"threshold {n}")

def plot_readout(y, eta):
    n_steps, n_neurons = y.shape
    t = np.array(range(n_steps)) * eta
    for n in range(n_neurons):
        traj = y[:, n]
        plt.plot(t, traj, label=f"y{n+1}")

def plot_raster(voltages, thresholds, eta):
    spikes = np.asarray(voltages >= thresholds[None,:]).nonzero()
    plt.scatter(spikes[0] * eta, spikes[1])

def plot_line_2D(f,g,t,x,y):
    return ( (f @ x) - (g[:,0][:,None] * y) - t) / g[:,1][:,None]

def evaluate_constraints(F,x,G,y,T):
    return (F @ x) - (G @ y) <= T

def base2float(mantissa, exponent):
    return mantissa * 2 ** exponent

def getmant(real, exponent):
    return real / (2**exponent)
