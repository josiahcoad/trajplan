from utils import plot_eps, save_episode
import pandas as pd
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench.monitor import Monitor
from env import Env
import time
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import SAC, PPO2
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from motion import get_spline
from constants import SEED
from state import State
from behavioral import get_behav, get_freespace, NoPathError
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
arr = np.array
AGG_MAP = {'fr': 'mean', 'fa': 'max', 'fj': 'max', 'fd': 'mean',
           'fk': 'max', 'fl': 'sum', 'fc': 'max', 'ft': 'mean'}


def train(name, agent=None):
    weights = {'fr': 0.3, 'fl': 1, 'fk': 1,
               'ft': 1, 'collision': -10, 'bias': 10}
    depth, width, move_dist, plan_dist = 3, 3, 3, 3
    max_steps, obstacle_pct = 1, 0.5

    def mkenv(): return Env(depth, width, move_dist, plan_dist,
                            max_steps=max_steps, weights=weights,
                            obstacle_pct=obstacle_pct)

    eval_callback = EvalCallback(mkenv(),
                                 best_model_save_path=f'logs/{name}',
                                 log_path='logs', eval_freq=1_000,
                                 deterministic=True, render=False)

    vecenv = make_vec_env(mkenv, 32, monitor_dir=f'logs/{name}/training')
    if agent:
        agent.set_env(vecenv)
    else:
        hparams = dict(n_steps=64, nminibatches=64, gamma=0.90,
                       learning_rate=2e-5, ent_coef=0.01,
                       cliprange=0.4, noptepochs=25, lam=0.99)
        agent = PPO2('MlpPolicy', vecenv, verbose=True, **hparams)
    agent.learn(1_000_000, callback=eval_callback)
    agent.save(f'logs/{name}/final')
    vecenv.close()
    return agent


def test(agent=None, render_step=False, eps_plot=True, eps_file=None):
    method = 'random' if agent == 'random' else ('rl' if agent else 'rule')
    weights = {'fr': 0.3, 'fl': 1, 'fk': 1,
               'ft': 1, 'success': 10, 'fail': 10, 'step_bonus': 10}
    depth, width, move_dist, plan_dist = 3, 3, 3, 3
    max_steps, obstacle_pct = 1, 0.5

    env = Env(depth, width, move_dist, plan_dist, save_history=eps_plot,
              max_steps=max_steps, weights=weights, obstacle_pct=obstacle_pct)

    eps_load = None if eps_file is None else \
        np.load(eps_file, allow_pickle=True)
    obs = env.reset(eps_load)
    done = False
    infos = []
    tr = 0
    i = 0
    while not done:
        if method == 'rule':
            try:
                action = get_behav(env.state.truncated(plan_dist), weights)
            except NoPathError:
                action = np.zeros(6)
        elif method == 'random':
            action = env.action_space.sample()
        elif method == 'rl':
            action, _ = agent.predict(obs, deterministic=True)
        if render_step:
            env.render(action)
        obs, rew, done, info = env.step(action)
        if not done:  # info is empty when done
            infos.append(info)
        tr += rew
        i += 1
    if eps_plot:
        if infos:
            print(pd.DataFrame(infos).agg(AGG_MAP).round(1))
        plot_eps(env)
        save_episode(env)
    # we failed if we are done and there was not a wall
    fail = not info['wall']
    return tr, i, fail


def demo(agent, n, eps_file=None):
    viz = n == 1
    start = time.time()
    scores = []
    fails = 0
    for _ in range(n):
        score, eplen, fail = test(agent, eps_plot=viz, eps_file=eps_file)
        fails += int(fail)
        if viz:
            print('eplen:', eplen)
            print('score:', int(round(score)))
        scores.append(score / eplen)
    print('----------')
    print('Time:', round(time.time() - start, 2))
    print('Success:', 1 - fails / n)
    scores = arr(scores)[arr(scores) != 0]
    print('Mean score:', np.mean(scores).round(2))
    if not viz:
        plt.plot(scores)
        plt.show()


if __name__ == '__main__':
    agent = PPO2.load('best_model')
    demo(None, 1)  # 'history.npy'
