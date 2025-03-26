import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Imposta il backend corretto per PyCharm
import matplotlib.pyplot as plt

# Creazione dell'ambiente
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inizializzazione della Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Parametri di apprendimento ottimizzati
alpha = 0.3        # Velocità di apprendimento più alta
gamma = 0.95       # Valore leggermente ridotto per stabilizzare l'apprendimento
epsilon = 1.0      # Esplorazione massima all'inizio
min_epsilon = 0.05 # Evita che l'esplorazione diventi zero
epsilon_decay = 0.9995  # Decadimento più lento dell'esplorazione
num_episodes = 100000  # Aumentiamo gli episodi per migliorare l'apprendimento

# Liste per tracciare i progressi
rewards_per_episode = []
q_table_mean = []

# Training per num_episodes
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Politica epsilon-greedy: esplorazione vs sfruttamento
        if np.random.rand() > epsilon:
            action = np.argmax(q_table[state])  # Sfrutta la Q-table
        else:
            action = env.action_space.sample()  # Esplora nuove azioni

        next_state, reward, done, truncated, _ = env.step(action)

        # Aggiornamento della Q-table con la formula Q-learning
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward  # Conta le vittorie

    # Memorizza il numero di vittorie e il valore medio della Q-table
    rewards_per_episode.append(total_reward)
    q_table_mean.append(np.mean(q_table))

    # Riduzione graduale di epsilon (senza scendere sotto min_epsilon)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Salvataggio della Q-table
np.save("frozenlake_qtable.npy", q_table)

print("✅ Training completato! Q-table salvata in 'frozenlake_qtable.npy'")

# **📊 Grafico delle vittorie per episodio**
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Media mobile per rendere il grafico più leggibile
rolling_avg = np.convolve(rewards_per_episode, np.ones(500)/500, mode='valid')
plt.plot(rolling_avg, label="Media su 500 episodi", color="blue")
plt.xlabel("Episodi")
plt.ylabel("Vittorie su 500 episodi")
plt.title("Progressione delle vittorie")
plt.legend()

# **📊 Grafico dell'evoluzione della Q-table**
plt.subplot(1, 2, 2)
plt.plot(q_table_mean, label="Valore medio della Q-table", color="red")
plt.xlabel("Episodi")
plt.ylabel("Valore medio della Q-table")
plt.title("Evoluzione della Q-table")
plt.legend()

print("Q-table finale:")
print(q_table)

plt.show()



