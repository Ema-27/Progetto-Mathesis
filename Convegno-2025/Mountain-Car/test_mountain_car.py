import gymnasium as gym
from stable_baselines3 import DQN

# Creazione dell'ambiente con visualizzazione
env = gym.make("MountainCar-v0", render_mode="human")

# Addestramento del modello
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Reset dell'ambiente prima della visualizzazione
obs, _ = env.reset()

# Esecuzione dell'agente addestrato con rendering
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()  # Visualizza il gioco a schermo
    if done or truncated:
        obs, _ = env.reset()

env.close()
