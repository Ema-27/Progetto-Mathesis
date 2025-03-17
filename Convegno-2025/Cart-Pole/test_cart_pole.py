import gymnasium as gym
from stable_baselines3 import PPO

# Carica l'ambiente e il modello addestrato
env = gym.make("CartPole-v1", render_mode="human")  # Modalità visibile
model = PPO.load("cartpole_ppo")

obs, _ = env.reset()
done = False

# Esegue il gioco con l'agente addestrato
while not done:
    action, _states = model.predict(obs)  # L'agente sceglie l'azione
    obs, reward, done, _, _ = env.step(action)

env.close()
