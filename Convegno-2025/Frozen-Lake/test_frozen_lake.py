import gymnasium as gym
import numpy as np

# Caricamento della Q-table
q_table = np.load("frozenlake_qtable.npy")

# Creazione dell'ambiente con rendering
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

# Reset dell'ambiente
obs, _ = env.reset()

# Testa l'agente su 10 partite
for episode in range(10):
    print(f"Partita {episode + 1}")
    done = False
    while not done:
        action = np.argmax(q_table[obs])  # Usa la Q-table per scegliere l'azione
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        if done or truncated:
            print(f"Risultato partita {episode + 1}: {'VITTORIA' if reward > 0 else 'SCONFITTA'}")
            obs, _ = env.reset()

env.close()
