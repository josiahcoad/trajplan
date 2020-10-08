from behavioral import get_behav, layer_dist, State, behav_cost, get_freespace, NoPathError
from motion import get_seed
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
import gym
from gym import spaces
from copy import deepcopy
arr = np.array
concat = np.concatenate

tol = 0.3 # max (abs) distance we allow the projection to be off

def project(state, action):
    # TODO: freespace not getting all freespace
    freespace = arr(get_freespace(state))
    if len(freespace) == 0:
        raise NoPathError(state)

    # print(freespace)
    idx = np.argmin(sum((freespace-action)**2))
    freetraj = freespace[idx]
    # print('prop:', action)
    # print('free:', freetraj)
    if np.max(np.abs(action - freetraj)) > tol:
        return [freetraj[0].astype(int), freetraj[1]]
    # print('chose well')
    return action


class Env(gym.Env):
    def __init__(self):
        depth=3
        width=4
        self.action_space = spaces.Box(-1, 1, shape=(2, depth))
        self.state = State(width=width, depth=depth)
        # self.state.load()
        self.state.save()
        self.observation_space = spaces.Box(-1, 1, (len(self.state.as_space()),))
        self.history = []

    def reset(self):
        self.state = State(width=width, depth=3)

    def step(self, action):
        # travel 1 distance (layer) along planned trajectory
        self.history.append((deepcopy(self.state), deepcopy(action)))
        rew = behav_cost(self.state, action)
        path, vel = action
        self.state.pos = path[0]
        self.state.vel = vel[0]
        self.state.step(1)
        return (self.state, -rew, False)

    def render(self, action):
        plot(self.state, action)


def plot(state, action):
    path = [state.pos] + list(action[0])
    vel = [state.vel] + list(action[1])

    xs = np.cumsum(np.ones(len(path))*layer_dist) - layer_dist

    _, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(state.depth):
        for j in range(state.width):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1, alpha=state.speed_lim[i,j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), state.speed_lim[i,j].round())
            if state.static_obs[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))
    ax1.scatter(xs, path, label='behav')
    seed = get_seed(xs, path)
    ax1.plot(*seed.T, label='seed', color='green')

    ax2.scatter(xs, vel)
    spline = get_seed(xs, vel, True)
    xs = np.arange(0, xs[-1]+.1, .1)
    ax2.plot(xs, spline(xs), label='vel')
    ax2.plot(xs, spline(xs, 1), label='acc')
    ax2.plot(xs, spline(xs, 2), label='jerk')
    ax2.set_ylim(-3, 4)
    ax2.set_xlim(0, 3.5)
    ax2.legend()

    plt.show()


def plot_eps(history):
    states = [h[0] for h in history]
    actions = [h[1] for h in history]
    statics = [s.static_obs for s in states]
    speeds = [s.speed_lim for s in states]
    epspeed = np.vstack((speeds[0], [speed[-1] for speed in speeds[1:]]))
    epstatic = np.vstack([statics[0], [static[-1] for static in statics[1:]]])
    _, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1, alpha=epspeed[i,j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), epspeed[i,j].round())
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    xs = [np.arange(states[0].depth + 1) + i for i in range(len(actions))]
    for x, state, (path, vel) in zip(xs, states, actions):
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        ax1.scatter(x, path, label='behav', color='purple')
        seed = get_seed(x, path)
        ax1.plot(*seed.T, label='seed', color='green')

        ax2.scatter(x, vel, color='purple')
        spline = get_seed(x, vel, True)
        xs = np.linspace(x[0], x[-1], 100)
        ax2.plot(xs, spline(xs), color='blue', label='vel')
        ax2.plot(xs, spline(xs, 1), color='green', label='acc')
        ax2.plot(xs, spline(xs, 2), color='red', label='jerk')
    ax2.legend()

    plt.show()


def action_adjust(state, dp, dv):
    p = np.cumsum(dp.round()) + env.state.pos
    v = np.cumsum(dv) + env.state.vel
    action = arr([p.astype(int), v])
    return project(state, action)


if __name__ == '__main__':
    env = Env()
    tr = 0
    done = False
    i = 0
    while not done:
        # [fr, fa, fj, fd, fk, fl, fc]
        action = get_behav(env.state, weights=[1,1,1,1,1,1,1])
        # dp, dv = env.action_space.sample()
        # action = action_adjust(env.state, dp, dv)
        # env.render(action)
        state, rew, done = env.step(action)
        tr += rew
        i += 1
        if i == 10:
            break
    plot_eps(env.history)
    print(tr)