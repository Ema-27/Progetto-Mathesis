import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Creazione dell'ambiente CartPole
env = gym.make("CartPole-v1")

# Vettorializzazione dell'ambiente per Stable-Baselines3
env = DummyVecEnv([lambda: env])

# Creazione del modello PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,   # Learning rate standard per PPO
    gamma=0.99,             # Sconto per le ricompense future
    n_steps=2048,           # Passi prima di aggiornare la policy
    ent_coef=0.01,          # Favorisce l'esplorazione
    batch_size=64,          # Batch size ottimale
    n_epochs=10,            # Numero di iterazioni per ogni batch
    clip_range=0.2          # Range di clipping per PPO
)

# Addestramento del modello
model.learn(total_timesteps=20000)

# Salvataggio del modello
model.save("cartpole_ppo")

# Chiusura dell'ambiente
env.close()
