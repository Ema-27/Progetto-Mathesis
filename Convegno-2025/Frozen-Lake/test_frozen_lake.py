import gymnasium as gym
import numpy as np

# Caricamento della Q-table
q_table = np.load("frozenlake_qtable.npy")

# Creazione dell'ambiente con rendering
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

# Testa l'agente su 10 partite
vittorie = 0
for episode in range(10):
    print(f"Partita {episode + 1}")
    obs, _ = env.reset()  # Reset dell'ambiente all'inizio di ogni episodio
    done = False
    while not done:
        action = np.argmax(q_table[obs])  # Usa la Q-table per scegliere l'azione
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        if done or truncated:
            print(f"Risultato partita {episode + 1}: {'VITTORIA' if reward > 0 else 'SCONFITTA'}")
            if reward > 0:
                vittorie += 1
            break  # Uscire dal ciclo while per passare alla partita successiva

# Calcolare la percentuale di vittorie
percentuale = (vittorie / 10) * 100
print(f"La percentuale di Vittorie è del: {percentuale}% su 10 partite")

env.close()
