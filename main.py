from behavioral import get_behav, layer_dist, State, behav_cost, safe
from motion import get_seed
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
import gym
from gym import spaces
arr = np.array
concat = np.concatenate


def project(state, path, vel):
    proposed = concat([path, vel])
    tol = 0.3
    lat_moves = product([-1, 0, 1], state.depth)
    vel_moves = product([-1, 0, 1], state.depth)
    paths = np.cumsum(lat_moves, axis=1) + state.pos
    vels = np.cumsum(vel_moves, axis=1) + state.vel
    Cfree = [concat([p, v])
            for p in paths for v in open_vels(state, path, vels)]
    idx = np.whichmin((Cfree-proposed)**2)
    freepath = Cfree[idx]
    dist = np.lingalg.norm(freepath-proposed)
    if np.max(np.abs(proposed - freepath)) > tol:
        return freepath, dist
    return proposed, 0

class Env(gym.Env):
    def __init__(self):
        depth = 3 
        self.action_space = spaces.Box(-1, 1, shape=(2, depth))
        self.state = State(width=4, depth=depth)
        # self.state.load()
        self.state.save()
        self.observation_space = spaces.Box(-1, 1, (len(self.state.as_space()),))
        self.history = []

    def reset(self):
        self.state = State(width=4, depth=3)

    def step(self, action):
        # travel 1 distance (layer) along planned trajectory
        path, vel = action
        (path, vel), dist = project(self.state, path, vel)
        rew = behav_cost(self.state, action)
        self.history.append((state, path, vel))
        self.state.pos = path[0]
        self.state.vel = vel[0]
        self.state.step(1)
        return (self.state, -rew-dist, False)

    def render(self, action):
        # p, v, _ = get_behav(self.state)
        # behav_cost(self.state, [p[1:], v[1:]]) 
        plot(self.state, action)


def plot(state, action):
    path, vel = action
    xs = np.cumsum(np.ones(len(path))*layer_dist) - layer_dist

    _, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(state.depth):
        for j in range(state.width):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1, alpha=state.speed_lim[i,j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), state.speed_lim[i,j].round())
            if state.static_obs[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))
    ax1.scatter(xs, path, label='behav')
    seed = get_seed(path)
    ax1.plot(*seed.T, label='seed', color='green')

    ax2.scatter(xs, vel)
    spline = get_seed(vel, True)
    xs = np.arange(0, xs[-1]+.1, .1)
    ax2.plot(xs, spline(xs), label='vel')
    ax2.plot(xs, spline(xs, 1), label='acc')
    ax2.plot(xs, spline(xs, 2), label='jerk')
    ax2.set_ylim(-3, 4)
    ax2.set_xlim(0, 3.5)
    ax2.legend()

    plt.show()

def plot_eps(history):
    states, paths, vels = history
    epspeed = np.vstack(state[0] + [state[-1] for state in states[1:]])
    epstatic = np.vstack(state[0] + [state[-1] for state in states[1:]])
    xs = [[1, 2, 3] + i for i in range(len(paths))]

    _, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1, alpha=epspeed[i,j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), epspeed[i,j].round())
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    for x, path in zip(xs, paths):
        ax1.scatter(x, path, label='behav')
        seed = get_seed(path)
        ax1.plot(*seed.T, label='seed', color='green')

        ax2.scatter(x, vel)
        spline = get_seed(vel, True)
        xs = np.arange(0, xs[-1]+.1, .1)
        ax2.plot(xs, spline(xs), color='blue', abel='vel')
        ax2.plot(xs, spline(xs, 1), color='green', label='acc')
        ax2.plot(xs, spline(xs, 2), color='red', label='jerk')
    ax2.legend()

    plt.show()

if __name__ == '__main__':
    env = Env()
    tr = 0
    done = False
    i = 0
    while not done:
        # [fr, fa, fj, fd, fk, fl, fc]
        # p, v, c = get_behav(env.state, weights=[0.5,1,1,1,1,4,1])
        p, v = env.action_space.sample()
        p = np.cumsum(p) + env.state.pos
        v = np.cumsum(v) + env.state.vel
        action = [p.astype(int), v]
        env.render(action)
        state, rew, done = env.step(action)
        tr += rew
        i += 1
        if i == 10:
            break
    print(tr)