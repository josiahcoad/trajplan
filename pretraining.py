import tensorflow as tf

import numpy as np
from stable_baselines import SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from behavioral import NoPathError, get_behav
from env import Env
from state import State

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# generate expert trajectory
env_depth, env_width, nlayers = 3, 3, 2


def expert(obs):
    try:
        state = State(env_depth, env_width).load_obs(obs)
        return get_behav(state, weights={'fr': 0.3})
    except NoPathError:
        return np.zeros(env_depth*2)


def main():
    generate_expert_traj(expert, 'expert', Env(3, 3, 3, 3), n_episodes=100)

    # pretrain model
    dataset = ExpertDataset(expert_path='expert.npz')
    model = SAC('MlpPolicy', Env(3, 3, 3, 3), verbose=1)
    model.pretrain(dataset, n_epochs=5000)
    model.save('pretrained_sac')

    # Test the pre-trained model
    env = model.get_env()
    obs = env.reset()

    reward_sum = 0
    i = 0
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        i += 1
        if done:
            print(reward_sum, i, reward_sum / i)
            reward_sum = 0
            i = 0
            obs = env.reset()

    env.close()
