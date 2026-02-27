# 🔹 Açıklama: Taxi-v3 ortamında Q-learning eğitimi. Gym ve NumPy 2.0 sürümlerine uyumlu hale getirilmiştir.
# 🔹 Gerekli pip paketleri: pip install gym numpy tqdm

import gym
import numpy as np
import random
from tqdm import tqdm

if not hasattr(np, "bool8"):
    np.bool8 = bool  # gym bazı yerlerde np.bool8 kullanıyor

def get_state_from_reset(reset_return):
    """gym.reset() bazı sürümlerde sadece state (obs) döner,
    bazı sürümlerde (state, info) döner."""
    if isinstance(reset_return, (tuple, list)):
        return reset_return[0]
    return reset_return

def step_env(env, action):
    """env.step() farklı gym sürümlerinde 4-tuple veya 5-tuple dönebilir."""
    ret = env.step(action)
    if len(ret) == 4:
        next_state, reward, done, info = ret
        return next_state, reward, done, info
    elif len(ret) == 5:
        next_state, reward, terminated, truncated, info = ret
        done = terminated or truncated
        return next_state, reward, done, info
    else:
        next_state = ret[0]
        reward = ret[1] if len(ret) > 1 else 0
        done = bool(ret[2]) if len(ret) > 2 else False
        info = ret[-1] if len(ret) > 0 else {}
        return next_state, reward, done, info

# Ortam oluşturma
env = gym.make("Taxi-v3", render_mode="ansi")
state = get_state_from_reset(env.reset())

# Render (bazı sürümler liste döndürebiliyor)
try:
    rendered = env.render()
    if isinstance(rendered, list) and len(rendered) > 0:
        print(rendered[0])
    else:
        print(rendered)
except Exception:
    pass

"""
Hareket kodları (Taxi-v3):
0: güney
1: kuzey
2: doğu
3: batı
4: yolcuyu almak
5: yolcuyu bırak
"""

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1  # öğrenme oranı
gamma = 0.6  # iskonto oranı
epsilon = 0.1  # keşif oranı

# Eğitim döngüsü
for i in tqdm(range(1, 100001)):
    state = get_state_from_reset(env.reset())
    done = False
    
    while not done:
        # %10 keşif, %90 sömürü
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))
    
        next_state, reward, done, info = step_env(env, action)
        
        # Q-table güncelleme
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state
        
print("Training finished ✅")

# Test bölümü
total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(episodes)):
    state = get_state_from_reset(env.reset())
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = int(np.argmax(q_table[state]))
        next_state, reward, done, info = step_env(env, action)
                
        state = next_state
        
        if reward == -10:
            penalties += 1
            
        epochs += 1
    
    total_epoch += epochs
    total_penalties += penalties
    
print(f"Result after {episodes} episodes")
print("Average timesteps per episode:", total_epoch / episodes)
print("Average penalties per episode:", total_penalties / episodes)
