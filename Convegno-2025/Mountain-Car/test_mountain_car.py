import gymnasium as gym
from stable_baselines3 import DQN

def test():
    # Creiamo l'ambiente con la modalità di rendering
    env = gym.make("MountainCar-v0", render_mode="human")

    # Carica il modello salvato
    model = DQN.load("mountaincar_dqn_model")

    # Testa il modello su 10 partite
    num_episodes = 1
    for episode in range(num_episodes):
        print(f"Partita {episode + 1}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
        print(f"Risultato partita {episode + 1}: {'VITTORIA' if total_reward > 0 else 'SCONFITTA'}")

    env.close()

if __name__ == "__main__":
    test()
