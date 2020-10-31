# pylint: disable=attribute-defined-outside-init
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib.patches import Rectangle

from behavioral import NoPathError, behav_cost, get_behav, get_freespace
from constant import LAYER_DIST
from state import State

arr = np.array
concat = np.concatenate


def project(state, proposal):
    freespace = arr(get_freespace(state))
    if len(freespace) == 0:
        raise NoPathError(state)

    idx = np.argmin(((freespace-proposal)**2).sum(1).sum(1))
    freetraj = freespace[idx]
    if np.max(np.abs(proposal - freetraj)) > 0.5:  # 0.5 is rounding cutoff
        # our proposal was too far from the freespace.
        residual = ((freetraj-proposal)**2).sum()
        return freetraj, residual
    return proposal, 0


def action_to_traj(state, action):
    # pylint: disable=unbalanced-tuple-unpacking
    dp, dv = np.split(action, 2)
    p = np.cumsum(dp) + state.pos
    v = np.cumsum(dv) + state.vel
    traj = arr([p, v])
    return traj


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
    ax2.scatter(xs, vel)
    ax2.set_ylim(-3, 4)
    ax2.set_xlim(0, 3.5)
    ax2.legend()

    plt.show()


def unsafe(state, path):
    # path should be absolute path not including the current position
    # return true if planned path is in occupied space
    path = path.round().astype(int)
    if any(path < 0) or any(path >= state.width):
        return True
    return any(state.static_obs[i, j] for i, j in enumerate(path))


class Env(gym.Env):
    def __init__(self, depth, width, move_dist, plan_dist,
                 save_history=False, weights=None, max_steps=None,
                 obstacle_pct=0.2, penalize_needed_lane_change=False):
        super().__init__()
        assert depth >= plan_dist >= move_dist
        self.depth = depth
        self.width = width
        self.move_dist = move_dist
        self.plan_dist = plan_dist
        self.save_history = save_history
        self.obstacle_pct = obstacle_pct
        self.penalize_needed_lane_change = penalize_needed_lane_change
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
            self.state = State(width=self.width, depth=self.depth,
                               obstacle_pct=self.obstacle_pct, assure_open_path=True)

        return self.state.obs

    def step(self, action):
        # convert action to absolute path
        traj = action_to_traj(self.state, action)
        # save history
        if self.save_history:
            self.history.append((deepcopy(self.state), deepcopy(traj)))
        if unsafe(self.state, traj[0]):
            done, info = True, {}
            # check if it was avoidable....
            wall = len(get_freespace(self.state)) == 0
            reward = self.weights.get('success', 10) if wall \
                else self.weights.get('fail', -10)
            info['wall'] = wall
        else:
            # get reward for action TODO: BUG FIX!!!!
            done = False
            bcost, info = behav_cost(
                self.state, traj, self.weights, return_parts=True)
            if not self.penalize_needed_lane_change:
                rtraj = get_behav(self.state,
                                  self.weights, absolute=True)
                _, rinfo = behav_cost(self.state,
                                      rtraj, self.weights, return_parts=True)
                bcost -= rinfo['fl'] * self.weights.get('fl', 1)
            reward = self.weights.get('step_bonus', 0) - bcost
        # update the state
        path, vel = traj
        self.stepn += 1
        if self.epload is not None:
            self.state = self.epload[self.stepn]
            self.state.pos = path[self.move_dist-1]
            self.state.vel = vel[self.move_dist-1]
        else:
            self.state.pos = path[self.move_dist-1]
            self.state.vel = vel[self.move_dist-1]
            # travel `move_dist` distance (num layers) along planned trajectory
            self.state.step(self.move_dist)
            # set a wall in the environment
            if self.max_steps and self.stepn == self.max_steps:
                self.state.static_obs[self.depth-1, :] = 1
        # save state after done since we will need to have it if we load eps
        if done and self.save_history:
            self.history.append((deepcopy(self.state), deepcopy(traj)))
        return self.state.obs, reward, done, info

    def render(self, action):
        traj = action_to_traj(self.state, action)
        plot(self.state, traj)


if __name__ == '__main__':
    from stable_baselines.common.env_checker import check_env
    check_env(Env(3, 3, 3, 3, 3))
