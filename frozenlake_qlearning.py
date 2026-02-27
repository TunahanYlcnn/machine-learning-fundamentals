# 🔹 Açıklama: FrozenLake-v1 ortamında Q-learning algoritmasıyla ajan eğitimi. Gym ve NumPy 2.0 sürümleriyle uyumludur.
# 🔹 Gerekli pip paketleri: pip install gym numpy tqdm matplotlib

import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- NumPy 2.0 uyumluluğu ---
if not hasattr(np, "bool8"):
    np.bool8 = bool

# --- Gym reset ve step uyum fonksiyonları ---
def get_state_from_reset(reset_return):
    """gym.reset() bazı sürümlerde sadece state döner, bazı sürümlerde (state, info)."""
    if isinstance(reset_return, (tuple, list)):
        return reset_return[0]
    return reset_return

def step_env(env, action):
    """env.step() farklı Gym sürümlerinde 4 veya 5 eleman dönebilir."""
    ret = env.step(action)
    if len(ret) == 4:
        new_state, reward, done, info = ret
        return new_state, reward, done, info
    elif len(ret) == 5:
        new_state, reward, terminated, truncated, info = ret
        done = terminated or truncated
        return new_state, reward, done, info
    else:
        raise ValueError("Beklenmeyen step dönüşü formatı!")

# --- Ortam oluşturma ---
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
state = get_state_from_reset(environment.reset())

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("İlk Q-Tablosu:")
print(qtable)

# --- Eğitim parametreleri ---
episodes = 1000
alpha = 0.5
gamma = 0.9
outcomes = []

# --- Eğitim döngüsü ---
for _ in tqdm(range(episodes)):
    state = get_state_from_reset(environment.reset())
    done = False
    success = False
    
    while not done:
        # Eylem seçimi: Q-table değerlerine göre veya rastgele
        if np.max(qtable[state]) > 0:
            action = int(np.argmax(qtable[state]))
        else:
            action = environment.action_space.sample()
        
        new_state, reward, done, info = step_env(environment, action)
        
        # Q-Tablosu güncelle
        qtable[state, action] = qtable[state, action] + alpha * (
            reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
        )
        
        state = new_state
        
        if reward == 1:
            success = True

    outcomes.append(success)

print("\nEğitim Sonrası Q-Tablosu:")
print(qtable)

# --- Başarı grafiği ---
plt.figure(figsize=(8, 3))
plt.plot(np.cumsum(outcomes), color='green')
plt.title("FrozenLake Q-Learning Eğitim Başarı Grafiği")
plt.xlabel("Episode")
plt.ylabel("Kümülatif Başarı")
plt.grid(True)
plt.show()

# --- Test döngüsü ---
episodes = 100
nb_success = 0

for _ in tqdm(range(episodes)):
    state = get_state_from_reset(environment.reset())
    done = False
    
    while not done:
        action = int(np.argmax(qtable[state]))
        new_state, reward, done, info = step_env(environment, action)
        state = new_state
        nb_success += reward

print(f"\n✅ Başarı Oranı: %{100 * nb_success / episodes:.2f}")
