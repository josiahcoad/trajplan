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
    def __init__(self, depth, width, move_dist, plan_dist,
                 save_history=False, weights=None, max_steps=None,
                 obstacle_pct=0.2):
        super().__init__()
        assert depth >= plan_dist and plan_dist >= move_dist
        self.depth = depth
        self.width = width
        self.move_dist = move_dist
        self.plan_dist = plan_dist
        self.save_history = save_history
        self.obstacle_pct = obstacle_pct
        self.weights = weights  # used for cost function calculation
        self.max_steps = max_steps  # stop early if reached this
        self.history = []
        self.reset()
        self.action_space = spaces.Box(-1, 1, shape=(2*self.plan_dist,))
        self.observation_space = spaces.Box(0, 5, self.state.obs.shape)

    def reset(self, history=None):
        """history is a list of states that represent a previous episode
        (the pos/vel will need to be overwritten)"""
        self.stepn = 0
        if history is not None:
            self.epload = deepcopy(history)
            self.state = self.epload[self.stepn]
        else:
            self.epload = None
            self.state = State(width=self.width, depth=self.depth, obstacle_pct=self.obstacle_pct)

        return self.state.obs

    def step(self, action):
        planning_space = self.state.truncated(self.plan_dist)
        # try to project action to safe space (if Error raised, means there is blockage)
        try:
            action, residual = postprocess_action(planning_space, action)
        except NoPathError:
            if self.save_history:
                self.history.append((planning_space, None))
            return self.state.obs, 0, True, {}
        # save history
        if self.save_history:
            self.history.append((planning_space, deepcopy(action)))
        # get reward for action
        bcost, parts = behav_cost(planning_space, action, self.weights, return_parts=True)
        parts['residual'] = round(residual, 2)
        reward = (25 - (bcost + self.weights.get('residual', 1) * residual)) / 10 # normalization of cost based on apriori knowledge
        path, vel = action
        # update the state
        self.stepn += 1
        if self.epload is not None:
            self.state = self.epload[self.stepn]
        else:
            # travel `move_dist` distance (num layers) along planned trajectory
            self.state.step(self.move_dist)
        self.state.pos = path[self.move_dist-1]
        self.state.vel = vel[self.move_dist-1]
        done = self.stepn >= self.max_steps if self.max_steps else False
        return self.state.obs, reward, done, parts

    def render(self, action):
        action = postprocess_action(self.state, action)
        plot(self.state, action)


if __name__ == '__main__':
    from stable_baselines.common.env_checker import check_env
    check_env(Env(3,3,3,3,3))
