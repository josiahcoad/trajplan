from behavioral import get_behav, behav_cost, get_freespace, NoPathError
from state import State
from constants import LAYER_DIST, SEED
from motion import get_spline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
import gym
from gym import spaces
from copy import deepcopy
from stable_baselines import SAC
from stable_baselines.common.cmd_util import make_vec_env
from env import Env

arr = np.array
concat = np.concatenate
np.random.seed(SEED)


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
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=epspeed[i, j]/10, color='red', zorder=-1))
            ax1.text((i+.5), (j-.5), epspeed[i, j].round())
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    xs = [np.arange(states[0].depth + 1) + i for i in range(len(actions))]
    for i, (x, state, (path, vel)) in enumerate(zip(xs, states, actions)):
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        ax1.scatter(x[0], path[0], color='purple')
        spline = get_spline(x, path, True)
        xs = np.linspace(x[0], x[1], num=20)
        ax1.plot(xs, spline(xs), color='green')

        ax2.scatter(x[0], vel[0], color='purple')
        spline = get_spline(x, vel, True)
        xs = np.linspace(x[0], x[1], num=20)
        ax2.plot(xs, spline(xs), color='blue', label='vel' if i == 0 else None)
        ax2.plot(xs, spline(xs, 1), color='green',
                 label='acc' if i == 0 else None)
        ax2.plot(xs, spline(xs, 2), color='red',
                 label='jrk' if i == 0 else None)
    ax1.set_xlim(-1, 13)
    ax2.set_xlim(-1, 13)
    ax2.legend()

    plt.show()

def train():
    env = make_vec_env(Env, n_envs=1, seed=SEED)
    agent = SAC('MlpPolicy', env, verbose=1, seed=SEED)
    agent.learn(100_000)
    agent.save('SAC')

def test(method, render_step=False):
    if method == 'rl':
        agent = SAC.load('SAC')
    env = Env(save_history=True)
    done = False
    obs = env.reset()
    tr = 0
    i = 0
    while not done:
        if method == 'rule':
            # [fr, fa, fj, fd, fk, fl, fc]
            action = get_behav(env.state, weights=[.1, 1, 1, 1, 1, 10, 1])
        elif method == 'random':
            action = env.action_space.sample()
        elif method == 'rl':
            action, _ = agent.predict(obs)
        if render_step:
            env.render(action)
        obs, rew, done, _ = env.step(action)
        tr += rew
        i += 1
        if i == 10:
            break
    if env.save_history:
        plot_eps(env.history)
    print(tr)

if __name__ == '__main__':
    test('rl')
