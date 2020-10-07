"""
get behavior path
1. fit cubic spline to each 2 points, sample along curve, call this P^{seed}
2. run (nonlinear, nonparametric) optimizer on path
3. fit parametric path (cubic spiral) to the path (tracking path generation)
    - find end point (pf) which is the first point that has a time >= Tmax
    - sample several points of lateral offset to pf, call them pf'
    - for each pf', fit cubic spiral which ends in that position
4. fit parametric path (cubic spiral) to the path (tracking vel generation)
    - for each pf'...
    - sample vf' from [0, vmax] (why not have vf' be sampled at offset from vf?)
"""

from scipy.interpolate import CubicSpline
from behavioral import get_behav, State, layer_dist
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize
arr = np.array
w = 0 # 0=seed (), 1=straight line (no curve)

def get_seed(path, return_spline=False):
    x = np.cumsum(np.ones(len(path))*layer_dist) - layer_dist
    cs = CubicSpline(x, path)
    xs = np.arange(0, x[-1]+.1, .1)
    ys = cs(xs)
    seed = arr([xs, ys]).T
    if return_spline:
        return cs
    return seed


def get_optim(seed):
    def obj(offsets, verbose=False):
        p = seed + offsets.reshape(-1, 1) @ arr([0, 1]).reshape(1, -1)
        dist = np.linalg.norm(np.diff(p, axis=0), axis=1) # 0
        u = np.diff(p, axis=0) / dist.reshape(-1, 1)
        k = np.diff(u, axis=0) / np.min([dist[1:], dist[:-1]], axis=0).reshape(-1, 1)
        if verbose:
            return np.linalg.norm(k, axis=1)
        return w * sum(np.linalg.norm(k, axis=1)) + (1 - w) * sum(np.abs(offsets)) # fix the off by one error here
    offsets = np.zeros(shape=len(seed))
    sol = least_squares(obj, offsets, bounds=[-0.5-seed[:,1], 2.5-seed[:,1]])
    print(sol.fun)
    print(sol.success)
    print(sol.message)
    p_opt = seed + (sol.x.reshape(-1, 1) @ arr([0, 1]).reshape(1, -1))
    return p_opt

if __name__ == '__main__':
    state = State()
    state.load()
    path, vel, cost = get_behav(state)
    x = np.cumsum(np.ones(len(path))*layer_dist) - layer_dist
    seed = get_seed(path)
    p_opt = get_optim(seed)
    plt.scatter(x, path, label='waypoints')
    plt.plot(*seed.T, label='seed')
    plt.plot(*p_opt.T, label='p_opt')
    for i, (x, y) in enumerate(p_opt[1:-1]):
        plt.text(x, y, round(k[i], 1))
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.legend()
    plt.show()