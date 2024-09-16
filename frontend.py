import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# Başlık
st.title("Pekiştirmeli Öğrenme: Rastgele Hareket ve Eğitimli Hareket")

# Açıklama
st.write("""
Bu uygulama, ajanın başlangıçta rastgele hareket ettiği ve Pekiştirmeli Öğrenmenin bir yöntemi olan Value Iteration yöntemi ile eğitildikten sonra
optimal hareketler yaptığı bir simülasyon gösterir. İlk olarak rastgele hareketleri izleyin, ardından 
'eğitime başla' butonuna basarak ajanın öğrenmiş halini görün.
""")

# Parametreler
grid_size = st.slider("Ortam boyutunu seçin (NxN)", 5, 10, 5)
gamma = st.slider("İndirim Faktörü (Gamma)", 0.01, 1.0, 0.9)
theta = st.slider("Eşik Değeri (Theta)", 0.001, 0.1, 0.01)
walls_percentage = st.slider("Engellerin yüzdesi (%)", 0, 50, 20)

# Hedef pozisyon
goal_position = [grid_size-1, grid_size-1]

# Duvarlar ekle
def create_walls(grid_size, walls_percentage):
    num_walls = int((grid_size * grid_size) * (walls_percentage / 100))
    walls = []
    while len(walls) < num_walls:
        wall_position = [np.random.randint(0, grid_size), np.random.randint(0, grid_size)]
        if wall_position != [0, 0] and wall_position != goal_position and wall_position not in walls:
            walls.append(wall_position)
    return walls

walls = create_walls(grid_size, walls_percentage)

# Eylemler ve yönler
actions = ["Yukarı", "Aşağı", "Sola", "Sağa"]

# Hareket fonksiyonu
def move_agent(action, position):
    new_position = position.copy()
    if action == 0 and position[0] > 0:  # Yukarı
        new_position[0] -= 1
    elif action == 1 and position[0] < grid_size - 1:  # Aşağı
        new_position[0] += 1
    elif action == 2 and position[1] > 0:  # Sola
        new_position[1] -= 1
    elif action == 3 and position[1] < grid_size - 1:  # Sağa
        new_position[1] += 1
    return new_position if new_position not in walls else position

# Ödül fonksiyonu
def get_reward(position):
    if position == goal_position:
        return 10  # Hedefe ulaşma ödülü
    else:
        return -1  # Her adımda küçük ceza

# Rastgele hareket
def random_movement():
    agent_position = [0, 0]
    steps = 0
    placeholder = st.empty()

    while agent_position != goal_position:
        action = np.random.choice(4)  # Rastgele aksiyon seç
        agent_position = move_agent(action, agent_position)
        steps += 1
        
        # Hareketi görselleştirme
        visualize_grid(agent_position, walls, goal_position, placeholder)
        time.sleep(0.3)
    
    st.success("Ajan rastgele hareketler ile hedefe ulaştı!")

# Value Iteration Fonksiyonu
def value_iteration(grid_size, gamma, theta, walls):
    values = np.zeros((grid_size, grid_size))  # Durum değerleri
    policy = np.zeros((grid_size, grid_size), dtype=int)  # Aksiyon politikaları (her durumda en iyi aksiyon)

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if [i, j] == goal_position or [i, j] in walls:
                    continue  # Hedefte veya duvarda aksiyon yok

                v = values[i, j]
                action_values = []

                # 4 farklı eylemi dene ve değer hesapla
                for action in range(4):
                    new_position = move_agent(action, [i, j])
                    reward = get_reward(new_position)
                    action_values.append(reward + gamma * values[new_position[0], new_position[1]])

                # En iyi aksiyonu seç ve değeri güncelle
                best_action_value = max(action_values)
                values[i, j] = best_action_value
                policy[i, j] = np.argmax(action_values)

                delta = max(delta, abs(v - values[i, j]))

        # Eşik değeri altında değişiklik olduğunda dur
        if delta < theta:
            break

    return values, policy

# Ortamı görselleştirme
def visualize_grid(agent_position, walls, goal_position, placeholder):
    grid = np.zeros((grid_size, grid_size))
    for wall in walls:
        grid[wall[0], wall[1]] = -1  # Duvarlar
    grid[goal_position[0], goal_position[1]] = 0.5  # Hedef
    grid[agent_position[0], agent_position[1]] = 1  # Ajan
    
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.grid(True)
    
    placeholder.pyplot(fig)

# Ajanın hareketini gerçekleştirme (Policy'ye göre)
def agent_movement(policy):
    agent_position = [0, 0]
    steps = 0
    placeholder = st.empty()

    while agent_position != goal_position:
        action = policy[agent_position[0], agent_position[1]]
        agent_position = move_agent(action, agent_position)
        steps += 1
        
        # Hareketi görselleştirme
        visualize_grid(agent_position, walls, goal_position, placeholder)
        time.sleep(0.2)

        # Adımları yazdır    
    st.success("Ajan eğitimli hareketlerle hedefe ulaştı!")

haraket_turu = st.radio("Hareket Türünü Seçin", ("Rastgele", "Eğitimli"))

if haraket_turu == "Rastgele":
    if st.button("Rastgele Hareket Başlat"):
        st.write("Ajan rastgele hareket ediyor...")
        random_movement()

elif haraket_turu == "Eğitimli":
    if st.button("Eğitime Başla"):
        st.write("Dinamik Programlama ile Ajan Öğreniyor...")
        
        # Value Iteration çalıştır
        values, policy = value_iteration(grid_size, gamma, theta, walls)
        st.write("Değer Tablosu (Value Table):")
        st.write(values)
        
        st.write("Politika (En iyi aksiyonlar):")
        st.write(policy)

        # Ajanı hareket ettirme
        agent_movement(policy)

# Rastgele hareket butonu
# if st.button("Rastgele Hareket Başlat"):
#     st.write("Ajan rastgele hareket ediyor...")
#     random_movement()

# # Value Iteration ve Ajanın Hareketi
# if st.button("Eğitime Başla"):
#     st.write("Dinamik Programlama ile Ajan Öğreniyor...")
    
#     # Value Iteration çalıştır
#     values, policy = value_iteration(grid_size, gamma, theta, walls)
#     st.write("Değer Tablosu (Value Table):")
#     st.write(values)
    
#     st.write("Politika (En iyi aksiyonlar):")
#     st.write(policy)

#     # Ajanı hareket ettirme
#     agent_movement(policy)
