import pygame
import numpy as np
import random
import sys
import os
import logging
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns  # 新增


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 游戏常量
WIDTH, HEIGHT = 1000, 800
GRID_SIZE = 40
GRID_WIDTH = (WIDTH - 200) // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'YELLOW': (255, 255, 0),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BROWN': (139, 69, 19),
    'GRAY': (128, 128, 128),
    'DARK_GRAY': (50, 50, 50)
}

# 初始化Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Pac-Man Q-Learning")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
bold_font = pygame.font.Font(None, 32)

# 加载图片素材
try:
    player_img = pygame.image.load("player.webp")
    enemy_img = pygame.image.load("enemy.webp")
    gold_img = pygame.image.load("gold.webp")
    logging.info("图片加载成功！")
except Exception as e:
    logging.error(f"图片加载失败: {e}")
    player_img = pygame.Surface((GRID_SIZE, GRID_SIZE))
    player_img.fill(COLORS['YELLOW'])
    enemy_img = pygame.Surface((GRID_SIZE, GRID_SIZE))
    enemy_img.fill(COLORS['RED'])
    gold_img = pygame.Surface((GRID_SIZE, GRID_SIZE))
    gold_img.fill(COLORS['GREEN'])

# 调整图片大小
player_img = pygame.transform.scale(player_img, (GRID_SIZE, GRID_SIZE))
enemy_img = pygame.transform.scale(enemy_img, (GRID_SIZE, GRID_SIZE))
gold_img = pygame.transform.scale(gold_img, (GRID_SIZE, GRID_SIZE))

# 强化学习核心参数
alpha = 0.7  # 初始学习率
gamma = 0.97  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.9995
epsilon_min = 0.02  # 基础最小值
adaptive_epsilon_min = 0.02  # 动态最小值
min_alpha = 0.1
batch_size = 256
target_update = 75
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
success_threshold = 0.25
boost_factor = 1.5
FPS = 30

# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_prio)

    def sample(self, batch_size, alpha=0.7, beta=0.6):
        probs = np.array(self.priorities) ** alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

replay_buffer = PrioritizedReplayBuffer(10000)

# 双Q网络架构
class DoubleQTable:
    def __init__(self, state_dims, action_size):
        self.main = np.random.uniform(-0.1, 0.1, state_dims + (action_size,))
        self.target = np.copy(self.main)

    def update_target(self):
        self.target = np.copy(self.main)

# 信息面板类
class InfoPanel:
    def __init__(self):
        self.width = 200
        self.height = HEIGHT
        self.x = WIDTH - self.width
        self.y = 0

    def draw(self, screen, episode, score, epsilon, steps, avg_q, success_count):
        pygame.draw.rect(screen, COLORS['DARK_GRAY'], (self.x, self.y, self.width, self.height))
        title = bold_font.render("Game Info", True, COLORS['WHITE'])
        screen.blit(title, (self.x + 10, 20))

        y_offset = 60
        info_items = [
            ("Episode", episode),
            ("Score", f"{score:.1f}"),
            ("Epsilon", f"{epsilon:.4f}"),
            ("Steps", steps),
            ("Avg QΔ", f"{avg_q:.2f}"),
            ("Success", sum(success_count))
        ]
        for label, value in info_items:
            text = font.render(f"{label}: {value}", True, COLORS['WHITE'])
            screen.blit(text, (self.x + 10, y_offset))
            y_offset += 35

        logs_title = bold_font.render("Recent Logs", True, COLORS['WHITE'])
        screen.blit(logs_title, (self.x + 10, HEIGHT - 160))

        log_y = HEIGHT - 120
        for msg in log_messages:
            log_text = font.render(msg, True, COLORS['WHITE'])
            screen.blit(log_text, (self.x + 10, log_y))
            log_y += 20

# 迷宫生成
def generate_maze(width, height):
    maze = []
    for y in range(height):
        row = []
        for x in range(width):
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                row.append('#')
            else:
                row.append('.' if random.random() > 0.1 else '#')
        maze.append(''.join(row))
    return maze

maze = generate_maze(GRID_WIDTH, GRID_HEIGHT)
obstacles = [(x, y) for y, row in enumerate(maze) for x, char in enumerate(row) if char == '#']

def is_obstacle(x, y):
    return (x, y) in obstacles

# 强化版Pacman类
class Pacman:
    def __init__(self):
        self.reset()
        self.path_history = deque(maxlen=5)
        self.visited_grids = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # 新增访问统计矩阵

    def reset(self):
        self.x, self.y = GRID_WIDTH // 2, GRID_HEIGHT // 2
        self.visited = set()
        self.visited.add((self.x, self.y))

    def move(self, action):
        new_x, new_y = self.x, self.y
        if action == 'UP' and not is_obstacle(self.x, self.y - 1):
            new_y -= 1
        elif action == 'DOWN' and not is_obstacle(self.x, self.y + 1):
            new_y += 1
        elif action == 'LEFT' and not is_obstacle(self.x - 1, self.y):
            new_x -= 1
        elif action == 'RIGHT' and not is_obstacle(self.x + 1, self.y):
            new_x += 1

        self.x, self.y = new_x, new_y
        self.visited.add((new_x, new_y))
        self.path_history.append((new_x, new_y))
        self.visited_grids[self.x][self.y] += 1  # 更新访问次数

    def get_state(self):
        px, py = self.x, self.y
        return (
            px % 3,
            py % 3,
            int(any(e.x == px for e in enemies)),
            int(any(e.y == py for e in enemies)),
            min(4, len(food.positions) // 2),
            int(px > GRID_WIDTH // 2),
            sum(1 for e in enemies if abs(e.x - px) < 3),
            min(4, sum(1 for p in food.positions if abs(p[0] - px) < 4))  # 修正括号
        )

# 食物类
class Food:
    def __init__(self):
        self.positions = []
        self.reset()

    def reset(self):
        self.positions = [(x, y) for y, row in enumerate(maze) for x, char in enumerate(row) if char == '.']
        random.shuffle(self.positions)
        self.positions = self.positions[:10]

    def collect(self, x, y):
        if (x, y) in self.positions:
            self.positions.remove((x, y))
            return True
        return False

# 敌人类
class Enemy:
    def __init__(self):
        self.reset()

    def reset(self):
        while True:
            self.x = random.randint(1, GRID_WIDTH - 2)
            self.y = random.randint(1, GRID_HEIGHT - 2)
            if not is_obstacle(self.x, self.y):
                break

    def move_towards(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        if abs(dx) > abs(dy):
            new_x = self.x + (1 if dx > 0 else -1)
            if not is_obstacle(new_x, self.y):
                self.x = new_x
        else:
            new_y = self.y + (1 if dy > 0 else -1)
            if not is_obstacle(self.x, new_y):
                self.y = new_y

    def smart_move(self, pacman):
        if random.random() < 0.7:
            self.move_towards(pacman.x, pacman.y)
        else:
            self.move_random()

    def move_random(self):
        action = random.choice(actions)
        new_x, new_y = self.x, self.y
        if action == 'UP' and not is_obstacle(self.x, self.y - 1):
            new_y -= 1
        elif action == 'DOWN' and not is_obstacle(self.x, self.y + 1):
            new_y += 1
        elif action == 'LEFT' and not is_obstacle(self.x - 1, self.y):
            new_x -= 1
        elif action == 'RIGHT' and not is_obstacle(self.x + 1, self.y):
            new_x += 1
        self.x, self.y = new_x, new_y

# 增强奖励函数
def calculate_reward(pacman, food, enemies):
    reward = 2  # 基础移动奖励
    px, py = pacman.x, pacman.y
    remaining = len(food.positions)

    if food.collect(px, py):
        base = 300 + 50 * (10 - remaining)
        if remaining < 5: base *= 3
        reward += base

    enemy_dist = min((abs(e.x - px) + abs(e.y - py) for e in enemies)) if enemies else 15
    gold_dist = min((abs(p[0] - px) + abs(p[1] - py) for p in food.positions)) if food.positions else 15

    reward += 50 * np.exp(-gold_dist / 3)
    reward -= 40 * np.exp(-enemy_dist / 1.5)

    visit_ratio = len(pacman.visited) / 200
    if (px, py) not in pacman.visited:
        reward += 35 * (1 - visit_ratio) ** 2
    else:
        reward -= max(1, 3 * visit_ratio)

    if len(pacman.path_history) >= 3:
        if pacman.path_history[-1] != pacman.path_history[-3]:
            reward += 10

    if enemy_dist < 6:
        reward -= 60 * (6 - enemy_dist) ** 1.5

    if not food.positions and (px, py) == (GRID_WIDTH // 2, GRID_HEIGHT // 2):
        reward += 800

    return reward

# 初始化游戏对象
pacman = Pacman()
food = Food()
enemies = [Enemy() for _ in range(2)]
info_panel = InfoPanel()

# 状态维度
state_dims = (3, 3, 2, 2, 5, 2, 3, 5)
q_table = DoubleQTable(state_dims, len(actions))

# 训练参数
max_episodes = 1000
max_steps = 1000
episode_rewards = []
episode_avg_q = []
success_count = []
epsilon_history = []
log_messages = deque(maxlen=5)

# 优化训练循环
for episode in range(1, max_episodes + 1):
    pacman.reset()
    food.reset()
    for e in enemies: e.reset()
    total_reward = 0
    steps = 0
    episode_q = []
    success = 0

    while steps < max_steps:
        state = pacman.get_state()

        # ε-贪婪策略
        if random.random() < epsilon:
            action = random.choice(actions)
            action_idx = actions.index(action)
        else:
            action_idx = np.argmax(q_table.main[state])
            action = actions[action_idx]

        # 执行动作
        pacman.move(action)

        # 敌人移动
        for e in enemies: e.smart_move(pacman)

        # 计算奖励
        reward = calculate_reward(pacman, food, enemies)
        total_reward += reward

        # 获取新状态
        next_state = pacman.get_state()
        done = any(pacman.x == e.x and pacman.y == e.y for e in enemies)
        if not food.positions and not done:
            success = 1
            done = True

        # 存储经验
        td_error = abs(reward + gamma * np.max(q_table.target[next_state]) - q_table.main[state][action_idx])
        replay_buffer.add((state, action_idx, reward, next_state, done, td_error))

        # 经验回放
        # 经验回放
        if len(replay_buffer.buffer) >= batch_size:
            batch, indices, weights = replay_buffer.sample(batch_size)
            new_priorities = []

            # 使用 enumerate 获取索引 i 和样本 exp
            for i, exp in enumerate(batch):
                s, a_idx, r, ns, d, _ = exp

                # 状态裁剪（确保维度正确）
                s = tuple(
                    max(0, min(int(x), dim_size - 1))
                    for x, dim_size in zip(s, state_dims))

                ns = tuple(
                    max(0, min(int(x), dim_size - 1))
                    for x, dim_size in zip(ns, state_dims))


                # 计算目标值和TD误差
                target = r + (1 - d) * gamma * np.max(q_table.target[ns])
                delta = abs(target - q_table.main[s][a_idx])

                # 使用 weights[i] 替代 indices.index(exp)
                q_table.main[s][a_idx] += alpha * (target - q_table.main[s][a_idx]) * weights[i]

                new_priorities.append(delta + 1e-5)
                episode_q.append(delta)

                # 更新优先级
                for idx, prio in zip(indices, new_priorities):
                    replay_buffer.priorities[idx] = prio

        # 同步目标网络
        if steps % target_update == 0:
            q_table.update_target()

        # 渲染
        screen.fill(COLORS['GRAY'])
        for y, row in enumerate(maze):
            for x, char in enumerate(row):
                if char == '#':
                    pygame.draw.rect(screen, COLORS['BROWN'], (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for pos in food.positions:
            screen.blit(gold_img, (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE))
        screen.blit(player_img, (pacman.x * GRID_SIZE, pacman.y * GRID_SIZE))
        for e in enemies:
            screen.blit(enemy_img, (e.x * GRID_SIZE, e.y * GRID_SIZE))

        avg_q = np.mean(episode_q) if episode_q else 0
        info_panel.draw(screen, episode, total_reward, epsilon, steps, avg_q, success_count)
        pygame.display.flip()
        clock.tick(FPS)

        if done:
            break
        steps += 1

    # 动态参数调整
    alpha = max(min_alpha, 0.5 * (1 + np.cos(episode / 500 * np.pi)) * 0.7)
    if len(success_count) >= 100 and np.mean(success_count[-100:]) > 0.3:
        alpha *= 0.995

    # ε策略调整
    epsilon = max(adaptive_epsilon_min, epsilon * epsilon_decay)
    if episode % 100 == 0 and len(success_count) >= 100:
        recent_success = np.mean(success_count[-100:])
        if recent_success < success_threshold:
            epsilon = min(0.3, epsilon * boost_factor)
            adaptive_epsilon_min = min(0.05, adaptive_epsilon_min * 1.1)
        else:
            adaptive_epsilon_min = max(0.005, adaptive_epsilon_min * 0.95)

    if episode % 400 == 0:
        epsilon = min(0.4, max(adaptive_epsilon_min, epsilon * 1.3))

    # 记录数据
    success_count.append(success)
    episode_rewards.append(total_reward)
    episode_avg_q.append(np.mean(episode_q) if episode_q else 0)
    epsilon_history.append(epsilon)

    # 日志记录
    log_msg = f"Ep {episode} | Reward: {total_reward:.1f} | Avg QΔ: {episode_avg_q[-1]:.2f} | ε: {epsilon:.4f} | Success: {success}"
    logging.info(log_msg)
    log_messages.append(log_msg)

    # 定期保存模型
    if episode % 100 == 0:
        np.save(f"q_table_ep{episode}.npy", q_table.main)

# 可视化增强
def plot_training():
    plt.figure(figsize=(18, 12))
    plt.style.use('seaborn-darkgrid')  # 全局样式

    # 1. 奖励趋势图
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='#1f77b4', label='Raw')
    reward_ma = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    plt.plot(reward_ma, color='#ff7f0e', linewidth=2, label='MA100')
    plt.title("Reward Trend", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.4)

    # 2. Q值动态图
    plt.subplot(2, 2, 2)
    plt.plot(episode_avg_q, color='#2ca02c', linewidth=2)
    plt.title("Q-Value Dynamics", fontsize=14)
    plt.grid(alpha=0.4)

    # 3. 探索率演化图
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_history, color='#d62728')
    plt.yscale('log')
    plt.title("Exploration Rate (ε)", fontsize=14)
    plt.grid(alpha=0.4)

    # 4. 迷宫探索热力图
    plt.subplot(2, 2, 4)
    log_visited = np.log1p(pacman.visited_grids.T)  # 对数变换
    sns.heatmap(
        log_visited,
        cmap="YlGnBu",
        annot=False,
        cbar_kws={'label': 'log(Visits + 1)'},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f"Maze Exploration Heatmap (Episodes={max_episodes})", fontsize=14)
    plt.xticks(
        np.arange(0, GRID_WIDTH, 5),
        labels=np.arange(0, GRID_WIDTH, 5) * GRID_SIZE,
        rotation=45
    )
    plt.yticks(
        np.arange(0, GRID_HEIGHT, 5),
        labels=np.arange(0, GRID_HEIGHT, 5) * GRID_SIZE
    )
    plt.xlabel("X Coordinate (pixels)", fontsize=12)
    plt.ylabel("Y Coordinate (pixels)", fontsize=12)

    # 统一保存
    plt.tight_layout()
    plt.savefig('training_report_v2.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training()

# 保持窗口
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    clock.tick(5)

pygame.quit()