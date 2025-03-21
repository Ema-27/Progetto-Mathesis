from stable_baselines3 import DQN
import gymnasium as gym

# Creiamo l'ambiente
env = gym.make("Acrobot-v1")

# Creiamo il modello DQN
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=50000, batch_size=64)

# Alleniamo il modello
model.learn(total_timesteps=50000)

# Salviamo il modello
model.save("acrobot_dqn")
