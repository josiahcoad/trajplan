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
from utils import plot_eps, save_episode
arr = np.array

def train(agent=None):
    weights = {'fr': 0.3, 'fl': 20, 'fk': 20, 'ft': 20, 'collision': -100, 'bias': 100}
    depth, width, move_dist, plan_dist = 3, 3, 3, 3
    mkenv = lambda: Env(depth, width, move_dist, plan_dist,
                        max_steps=20, weights=weights,
                        obstacle_pct=0.1)

    eval_callback = EvalCallback(mkenv(),
                             best_model_save_path='logs/models',
                             log_path='logs', eval_freq=1_000,
                             deterministic=True, render=False)

    vecenv = make_vec_env(mkenv, 32, monitor_dir='logs/training')
    if agent:
        agent.set_env(vecenv)
    else:
        hparams = dict(n_steps=64, nminibatches=64, gamma=0.90,
                       learning_rate=2e-5, ent_coef=0.01,
                       cliprange=0.4, noptepochs=25, lam=0.99)
        agent = PPO2('MlpPolicy', vecenv, verbose=True, **hparams)
    agent.learn(1_000_000, callback=eval_callback)
    agent.save('logs/models/final')
    vecenv.close()
    return agent


def test(agent=None, render_step=False, eps_plot=True, eps_file=None):
    method = 'random' if agent == 'random' else ('rl' if agent else 'rule')
    weights = {'fr': 0.3, 'fl': 20, 'fk': 20, 'ft': 10}
    depth, width, move_dist, plan_dist = 3, 3, 3, 3
    env= Env(depth, width, move_dist, plan_dist, save_history=eps_plot,
             max_steps=20, weights=weights, obstacle_pct=0.1)

    eps_load = None if eps_file is None else np.load(eps_file, allow_pickle=True)
    obs = env.reset(eps_load)
    done = False
    cost_parts = []
    tr = 0
    i = 0
    while not done:
        if method == 'rule':
            try:
                action = get_behav(env.state.truncated(plan_dist), weights)
            except NoPathError:
                action = np.ones(6) # should trigger a NoPathError in env.step
        elif method == 'random':
            action = env.action_space.sample()
        elif method == 'rl':
            action, _ = agent.predict(obs, deterministic=True)
        if render_step:
            env.render(action)
        obs, rew, done, parts = env.step(action)
        if not done: # parts is empty when done
            cost_parts.append(parts)
        tr += rew
        i += 1
    if eps_plot:
        agg_map = {'fr': 'mean', 'fa': 'max', 'fj': 'max', 'fd': 'mean',
                   'fk': 'max', 'fl': 'sum', 'fc': 'max', 'ft': 'mean'}
        if cost_parts:
            print(pd.DataFrame(cost_parts).agg(agg_map).round(1))
        plot_eps(env)
        save_episode(env)
    return tr, i


def demo(agent, n, eps_file=None):
    viz = n == 1
    start = time.time()
    scores = []
    for _ in range(n):
        score, eplen = test(agent, eps_plot=viz, eps_file=eps_file)
        if viz:
          print('eplen:', eplen)
          print('score:', int(round(score)))
        scores.append(score / eplen)
    print('----------')
    print('Time:', round(time.time() - start, 2))
    scores = arr(scores)[arr(scores) != 0]
    print('Mean score:', np.mean(scores).round(2))
    if not viz:
        plt.plot(scores)
        plt.show()

if __name__ == '__main__':
    _agent = PPO2.load('logs/models/best_model')
    demo(None, 1) # 'history.npy'