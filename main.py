import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from behavioral import get_behav, NoPathError
from state import State
from constants import SEED
from motion import get_spline
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from stable_baselines import SAC
from stable_baselines.common.cmd_util import make_vec_env
import time
from env import Env
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.callbacks import EvalCallback
import pandas as pd

arr = np.array
concat = np.concatenate
# np.random.seed(SEED)


def plot_eps(history):
    states = [h[0] for h in history]
    actions = [h[1] for h in history]
    statics = [s.static_obs for s in states]
    speeds = [s.speed_lim for s in states]
    epspeed = np.vstack((speeds[0], [speed[-1] for speed in speeds[1:]]))
    epstatic = np.vstack([statics[0], [static[-1] for static in statics[1:]]])
    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2twin = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=epspeed[i, j]/10, color='red', zorder=-1))
            # ax1.text((i+.5), (j-.5), epspeed[i, j].round().astype(int))
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    xs = [np.arange(states[0].depth + 1) + i for i in range(len(actions))]
    p_bc = 0 # path (starting) boundary condition (first derivative)
    v_bc = 0 # velocity ^^
    path_len = 0
    for i, (x, state, (path, vel)) in enumerate(zip(xs, states, actions)):
        # plot path
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        ax1.scatter(x[0], path[0], color='purple')
        spline = get_spline(x, path, p_bc, True)
        p_bc = spline(x[1:2], 1)[0]
        xs = np.linspace(x[0], x[1], num=20)
        ys = spline(xs)
        ax1.plot(xs, ys, color='green')

        # plot heading and steer
        dy = np.diff(ys)
        dx = np.diff(xs)
        head = np.rad2deg(np.arctan2(dy, dx)) + [0]
        deltas = [0] + np.sqrt(dx**2 + dy**2)
        dists = np.cumsum(deltas) + path_len
        ax2.plot(dists, head, color='blue')
        steers = np.diff(head) / 10
        ax2twin.plot(dists[1:], steers, color='green')
        path_len = dists[-1]

        # plot velocity profile
        ax3.scatter(x[0], vel[0], color='purple')
        ax3.scatter(x[0]+1, epspeed[i][int(round(path[1]))], color='orange',
                    label='refvel' if i == 0 else None)
        vspline = get_spline(x, vel, v_bc, True)
        v_bc = vspline(x[1:2], 1)[0]
        xs = np.linspace(x[0], x[1], num=20)
        ax3.plot(xs, vspline(xs), color='blue',
                label='vel' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 1), color='green',
                 label='acc' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 2), color='red',
                 label='jrk' if i == 0 else None)

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


def train(agent=None):
    weights = {'fr': 0.3}
    eval_callback = EvalCallback(Env(weights=weights), best_model_save_path='logs/models',
                             log_path='logs', eval_freq=20000,
                             deterministic=True, render=False)
    env = Monitor(Env(weights=weights), 'logs/training')
    if agent:
        agent.set_env(env)
    else:
        agent = SAC('MlpPolicy', env, verbose=True)
    agent.learn(200_000, callback=eval_callback)
    agent.save('SAC')


def test(agent=None, random=False, render_step=False, eps_plot=True):
    weights = {'fr': 0.3}
    method = 'random' if random else ('rl' if agent else 'rule')
    env = Env(save_history=eps_plot, max_steps=100, weights=weights)
    obs = env.reset()
    done = False
    blocked = False
    cost_parts = []
    tr = 0
    i = 0
    while not done:
        if method == 'rule':
            # [fr, fa, fj, fd, fk, fl, fc]
            try:
                action = get_behav(env.state, weights)
            except NoPathError:
                blocked = True
        elif method == 'random':
            action = env.action_space.sample()
        elif method == 'rl':
            action, _ = agent.predict(obs)
        if render_step:
            env.render(action)
        if blocked:
            obs, rew, done = env.state.obs, 0, True
        else:
            obs, rew, done, parts = env.step(action)
            cost_parts.append(parts)
        tr += rew
        i += 1
    if env.history:
        agg_map = {'fr': 'mean', 'fa': 'max', 'fj': 'max', 'fd': 'mean',
                'fk': 'max', 'fl': 'sum', 'fc': 'max'}
        print(pd.DataFrame(cost_parts).agg(agg_map).round(1))
        plot_eps(env.history)
    return tr, i


if __name__ == '__main__':
    agent = SAC.load('SAC_best')
    # train(agent)  
    np.random.seed(int(time.time()))
    start = time.time()
    scores = []
    for _ in range(2):
        score, eplen = test(agent=agent, eps_plot=True)
        print('eplen:', eplen)
        print('score:', int(round(score)))
        scores.append(score / eplen)
    print('----------')
    print(time.time() - start)
    scores = arr(scores)[arr(scores) != 0]
    print(np.mean(scores))
    # plt.plot(scores)
    # plt.show()
