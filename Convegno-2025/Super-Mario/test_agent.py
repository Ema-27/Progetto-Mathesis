import gymnasium as gym
import time
from stable_baselines3 import DQN

# Creiamo l'ambiente con il render_mode corretto
env = gym.make("Acrobot-v1", render_mode="human")

# Carichiamo il modello addestrato
model = DQN.load("acrobot_dqn")

# Resettiamo l'ambiente
obs, info = env.reset()
done = False

# Testiamo l'agente
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(0.05)  # Rallenta l'esecuzione per visualizzare meglio il comportamento

env.close()
