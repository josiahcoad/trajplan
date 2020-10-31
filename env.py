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


def action_to_traj(state, action):
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


def unsafe(state, path):
    # path should be absolute path not including the current position
    # return true if planned path is in occupied space
    path = path.round().astype(int)
    if any(path < 0) or any(path >= state.depth):
        return True
    return any(state.static_obs[i, j] for i, j in enumerate(path))


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
            # get reward for action
            done = False
            bcost, info = behav_cost(
                self.state, traj, self.weights, return_parts=True)
            # normalization of cost based on apriori knowledge
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
            if self.max_steps and self.stepn >= self.max_steps - 1:
                self.state.static_obs[self.depth-1, :] = 1

        return self.state.obs, reward, done, info

    def render(self, action):
        traj = action_to_traj(self.state, action)
        plot(self.state, traj)


if __name__ == '__main__':
    from stable_baselines.common.env_checker import check_env
    check_env(Env(3, 3, 3, 3, 3))
