from behavioral import get_behav, behav_cost, get_freespace, NoPathError
from state import State
from constants import LAYER_DIST
from motion import get_spline
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import gym
from gym import spaces
from copy import deepcopy

arr = np.array
concat = np.concatenate


def project(state, proposal):
    # TOL must be below 0.5, else we'd round to a different cell
    TOL = .5  # max (abs) distance we allow the projection to be off
    freespace = arr(get_freespace(state))
    if len(freespace) == 0:
        raise NoPathError(state)

    idx = np.argmin(((freespace-proposal)**2).sum(1).sum(1))
    freetraj = freespace[idx]
    if np.max(np.abs(proposal - freetraj)) > TOL:
        # our proposal was too far from the freespace.
        residual = ((freetraj-proposal)**2).sum()
        return freetraj, residual
    return proposal, 0


def postprocess_action(state, action):
    dp, dv = np.split(action, 2)
    p = np.cumsum(dp) + state.pos
    v = np.cumsum(dv) + state.vel
    action = arr([p, v])
    return project(state, action)


def plot(state, action):
    path = [state.pos] + list(action[0])
    vel = [state.vel] + list(action[1])

    xs = np.cumsum(np.ones(len(path))*LAYER_DIST) - LAYER_DIST

    _, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(state.depth):
        for j in range(state.width):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=state.speed_lim[i, j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), state.speed_lim[i, j].round())
            if state.static_obs[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))
    ax1.scatter(xs, path, label='behav')
    seed = get_spline(xs, path)
    ax1.plot(*seed.T, label='seed', color='green')

    ax2.scatter(xs, vel)
    spline = get_spline(xs, vel, True)
    xs = np.arange(0, xs[-1]+.1, .1)
    ax2.plot(xs, spline(xs), label='vel')
    ax2.plot(xs, spline(xs, 1), label='acc')
    ax2.plot(xs, spline(xs, 2), label='jerk')
    ax2.set_ylim(-3, 4)
    ax2.set_xlim(0, 3.5)
    ax2.legend()

    plt.show()


class Env(gym.Env):
    depth = 3
    width = 3

    def __init__(self, save_history=False, weights=None, max_steps=None):
        super().__init__()
        self.save_history = save_history
        self.weights = weights  # used for cost function calculation
        self.max_steps = max_steps  # stop early if reached this
        self.history = []
        self.reset()
        self.action_space = spaces.Box(-1, 1, shape=(2*self.depth,))
        self.observation_space = spaces.Box(0, 5, self.state.obs.shape)

    def reset(self):
        self.state = State(width=self.width, depth=self.depth)
        self.stepn = 0
        return self.state.obs

    def step(self, action):
        try:
            action, residual = postprocess_action(self.state, action)
        except NoPathError:
            return self.state.obs, 0, True, {}
        # travel 1 distance (layer) along planned trajectory
        if self.save_history:
            self.history.append((deepcopy(self.state), deepcopy(action)))
        bcost, parts = behav_cost(self.state, action, self.weights, return_parts=True)
        reward = (25-bcost)/10 - residual # normalization of cost based on apriori knowledge
        path, vel = action
        self.state.pos = path[0]
        self.state.vel = vel[0]
        self.state.step(1)
        self.stepn += 1
        done = self.stepn >= self.max_steps if self.max_steps else False
        return self.state.obs, reward, done, parts

    def render(self, action):
        action = postprocess_action(self.state, action)
        plot(self.state, action)


if __name__ == '__main__':
    from stable_baselines.common.env_checker import check_env
    check_env(Env())
