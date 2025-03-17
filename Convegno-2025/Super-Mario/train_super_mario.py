import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from nes_py.wrappers import JoypadSpace
import gym

# Creazione dell'ambiente senza rendering
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")  # Usa la versione v3
env = JoypadSpace(env, [["right"], ["right", "A"]])  # Comandi semplici

# Vettorializzazione dell'ambiente per Stable-Baselines3
env = DummyVecEnv([lambda: env])

# Creazione e addestramento del modello
model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Salvataggio del modello
model.save("super_mario_dqn")

# Chiusura dell'ambiente
env.close()

print("Training completato e modello salvato come 'super_mario_dqn.zip'")
