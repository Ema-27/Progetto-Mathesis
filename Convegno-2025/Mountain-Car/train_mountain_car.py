import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("MountainCar-v0")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()
env.close()
