import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from behavioral import get_behav, NoPathError
from state import State
from constants import SEED
from motion import get_spline
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from stable_baselines import SAC, PPO2
from stable_baselines.common.cmd_util import make_vec_env
import time
from env import Env
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.callbacks import EvalCallback
import pandas as pd

arr = np.array
concat = np.concatenate
# np.random.seed(SEED)


def plot_eps(env):
    states = [h[0] for h in env.history]
    actions = [h[1] for h in env.history[:-1]] # the last action is None because we detected end of road
    nlayers = env.step_layers
    statics = [s.static_obs for s in states]
    speeds = [s.speed_lim for s in states]
    epspeed = np.vstack([speeds[0], *[speed[-nlayers:] for speed in speeds[1:]]])
    epstatic = np.vstack([statics[0], *[static[-nlayers:] for static in statics[1:]]])
    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2twin = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=epspeed[i, j]/10, color='red', zorder=-1))
            # ax1.text((i+.5), (j-.5), epspeed[i, j].round().astype(int))
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    xseed = np.arange(states[0].depth + 1)
    xs = [xseed + i * nlayers for i in range(len(actions))]
    p_bc = 0 # path (starting) boundary condition (first derivative)
    v_bc = 0 # velocity ^^
    path_len = 0
    for i, (x, state, (path, vel)) in enumerate(zip(xs, states, actions)):
        # plot path
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        ax1.scatter(x[:nlayers], path[:nlayers], color='purple')
        spline = get_spline(x, path, p_bc, True)
        p_bc = spline([x[nlayers]], 1)[0] # eval the first deriv at the 'nlayers' point
        xs = np.linspace(x[0], x[nlayers], num=20)
        ys = spline(xs)
        ax1.plot(xs, ys, color='green')

        # plot heading and steer
        dy = np.diff(ys)
        dx = np.diff(xs)
        head = np.rad2deg(np.arctan2(dy, dx)) + [0]
        deltas = [0] + np.sqrt(dx**2 + dy**2)
        dists = np.cumsum(deltas) + path_len
        ax2.plot(dists, head, color='blue')
        steers = np.diff(head) / 30
        ax2twin.plot(dists[1:], steers, color='green')
        path_len = dists[-1]
        # TODO: curve = np.diff(steers)

        # plot velocity profile
        ax3.scatter(x[:nlayers], vel[:nlayers], color='purple')
        ax3.scatter(x[:nlayers]+1, epspeed[i:nlayers+i][int(round(path[1]))],
                color='orange', label='refvel' if i == 0 else None)
        vspline = get_spline(x, vel, v_bc, True)
        v_bc = vspline([x[nlayers]], 1)[0]
        xs = np.linspace(x[0], x[nlayers], num=20)
        ax3.plot(xs, vspline(xs), color='blue',
                label='vel' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 1), color='green',
                label='acc' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 2), color='red',
                label='jrk' if i == 0 else None)
    
    # plot last point
    ax1.scatter(x[nlayers], path[nlayers], color='purple')
    
    # set plotting options 
    color = 'tab:blue'
    ax2.set_xlabel('path dist (s)')
    ax2.set_ylabel('heading (deg)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-75, 75)
    color = 'tab:green'
    ax2twin.set_ylabel('turn', color=color) 
    ax2twin.tick_params(axis='y', labelcolor=color)
    ax2twin.set_ylim(-1, 1)

    ax3.legend()
    ax3.set_xlim(*ax1.get_xlim())

    plt.show()


def save_episode(env):
    history = env.history
    states = arr([s[0] for s in history])
    np.save('history.npy', states)


def train(agent=None):
    weights = {'fr': 0.3}
    eval_callback = EvalCallback(Env(weights=weights), best_model_save_path='logs/models',
                             log_path='logs', eval_freq=1_000,
                             deterministic=True, render=False)

    vecenv = make_vec_env(lambda: Env(weights=weights), 32, monitor_dir='logs/training')
    if agent:
        agent.set_env(vecenv)
    else:
        hparams = dict(n_steps=64, nminibatches=32, gamma=0.95,
                       learning_rate=2e-5, ent_coef=0.01,
                       cliprange=0.2, noptepochs=25, lam=0.99)
        agent = PPO2('MlpPolicy', vecenv, verbose=True, **hparams)
    agent.learn(200_000, callback=eval_callback)
    agent.save('PPO')


def test(agent=None, random=False, render_step=False, eps_plot=True):
    weights = {'fr': 0.3}
    # history = np.load('history.npy', allow_pickle=True)
    method = 'random' if random else ('rl' if agent else 'rule')
    env = Env(save_history=eps_plot, max_steps=100, weights=weights)
    obs = env.reset()
    done = False
    cost_parts = []
    tr = 0
    i = 0
    while not done:
        if method == 'rule':
            try:
                action = get_behav(env.state, weights)
            except NoPathError:
                action = np.ones(6) # should trigger a NoPathError in env.step
        elif method == 'random':
            action = env.action_space.sample()
        elif method == 'rl':
            action, _ = agent.predict(obs)
        if render_step:
            env.render(action)
        obs, rew, done, parts = env.step(action)
        cost_parts.append(parts)
        tr += rew
        i += 1
    if eps_plot:
        agg_map = {'fr': 'mean', 'fa': 'max', 'fj': 'max', 'fd': 'mean',
                'fk': 'max', 'fl': 'sum', 'fc': 'max'}
        print(pd.DataFrame(cost_parts).agg(agg_map).round(1))
        plot_eps(env)
        save_episode(env)
    return tr, i


if __name__ == '__main__':
    agent = PPO2.load('best_model')
    # train() 
    np.random.seed(int(time.time()))
    start = time.time()
    scores = []
    for _ in range(1):
        score, eplen = test(agent=None, eps_plot=True)
        print('eplen:', eplen)
        print('score:', int(round(score)))
        scores.append(score / eplen)
    print('----------')
    print(time.time() - start)
    scores = arr(scores)[arr(scores) != 0]
    print(np.mean(scores))
    # plt.plot(scores)
    # plt.show()
