from stable_baselines import PPO2
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from behavioral import get_behav
from main import Env

def generate_expert():
    env = Env()

    def expert(obs):
        return get_behav(obs, weights=[.1,1,1,1,1,10,1])

    generate_expert_traj(expert, 'expert', env, n_episodes=10)


# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                        traj_limitation=1, batch_size=128)

model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=1000)

# As an option, you can train the RL agent
# model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render()
        if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

env.close()