import gym_super_mario_bros
from stable_baselines3 import DQN
from nes_py.wrappers import JoypadSpace
import gym

# Creazione dell'ambiente con rendering
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")  # Usa la versione v3
env = JoypadSpace(env, [["right"], ["right", "A"]])  # Comandi semplici

# Caricamento del modello addestrato
model = DQN.load("super_mario_dqn")

# Reset dell'ambiente prima della simulazione
obs = env.reset()

# Esecuzione dell'agente addestrato con rendering
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()  # Visualizza il gioco a schermo
    if done:
        obs = env.reset()

# Chiusura dell'ambiente dopo il testing
env.close()
