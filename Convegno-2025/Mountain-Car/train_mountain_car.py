import gymnasium as gym
from stable_baselines3 import DQN

def train():
    # Creiamo l'ambiente con la modalità di rendering
    env = gym.make("MountainCar-v0", render_mode="human")

    # Creiamo il modello DQN
    model = DQN("MlpPolicy", env, verbose=1)

    # Inizia il training
    print("🔄 Inizio training...")
    model.learn(total_timesteps=50000)

    # Salva il modello addestrato
    model.save("mountaincar_dqn_model")
    print("✅ Training completato. Modello salvato come 'mountaincar_dqn_model'")

    # Chiudi l'ambiente
    env.close()

if __name__ == "__main__":
    train()
