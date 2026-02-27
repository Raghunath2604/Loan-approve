"""
Multi-Agent DQN Comparison — CartPole-v1
Compare: Full DQN | No Target Network | No Replay Buffer
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import keras
from keras import layers
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPISODES = 50
MAX_STEPS = 200
BATCH_SIZE = 32
MEMORY_SIZE = 2000
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99


def build_model(state_size, action_size):
    model = keras.Sequential([
        keras.Input(shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(LR), loss='mse')
    return model


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions, dtype=int),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, use_replay=True, use_target=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_replay = use_replay
        self.use_target = use_target
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.online = build_model(state_size, action_size)
        self.target = build_model(state_size, action_size)
        if use_target:
            self.target.set_weights(self.online.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.online(np.array([state]), training=False).numpy()
        return int(np.argmax(q[0]))

    def train_step(self, state, action, reward, next_state, done):
        if self.use_replay:
            self.memory.push((state, action, reward, next_state, done))
            if len(self.memory) < BATCH_SIZE:
                return 0.0
            states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        else:
            states = np.array([state])
            actions = np.array([action], dtype=int)
            rewards = np.array([reward])
            next_states = np.array([next_state])
            dones = np.array([done])

        target_q = self.online(states, training=False).numpy()
        net = self.target if self.use_target else self.online
        next_q = net(next_states, training=False).numpy()

        for i in range(len(states)):
            t = rewards[i]
            if not dones[i]:
                t += GAMMA * np.max(next_q[i])
            target_q[i][actions[i]] = t

        h = self.online.fit(states, target_q, epochs=1, verbose=0)
        return h.history['loss'][0]

    def update_target(self):
        if self.use_target:
            self.target.set_weights(self.online.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def train_agent(name, use_replay, use_target):
    env = gym.make("CartPole-v1")
    s_size = env.observation_space.shape[0]  # type: ignore
    a_size = env.action_space.n  # type: ignore
    agent = DQNAgent(s_size, a_size, use_replay, use_target)

    scores, losses = [], []

    for ep in range(EPISODES):
        state, _ = env.reset()
        total = 0.0
        ep_loss = []

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.train_step(state, action, float(reward), next_state, done)
            if loss:
                ep_loss.append(loss)

            state = next_state
            total += float(reward)
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % 5 == 0:
            agent.update_target()

        scores.append(total)
        losses.append(np.mean(ep_loss) if ep_loss else 0)
        print(f"  {name} | Ep {ep+1}/{EPISODES} | Score: {total:.0f} | Eps: {agent.epsilon:.3f}")

    env.close()
    return scores, losses


# --- Run All Agents ---
configs = {
    "Full DQN":  (True, True),
    "No Target": (True, False),
    "No Replay": (False, False),
}

results = {}
for name, (replay, target) in configs.items():
    print(f"\n=== Training: {name} ===")
    scores, losses = train_agent(name, replay, target)
    results[name] = {"scores": scores, "losses": losses}
# --- Compute accuracy (% of max possible score = 200) ---
MAX_SCORE = 200.0
for name in results:
    results[name]["accuracy"] = [(s / MAX_SCORE) * 100 for s in results[name]["scores"]]

# --- Individual Accuracy Graph for each algorithm ---
colors = {"Full DQN": "#2196F3", "No Target": "#FF9800", "No Replay": "#F44336"}

for name in results:
    fig, ax = plt.subplots(figsize=(10, 5))
    acc = results[name]["accuracy"]
    avg_acc = np.convolve(acc, np.ones(10)/10, mode='valid')

    ax.plot(acc, alpha=0.3, color=colors[name], label="Per Episode")
    ax.plot(range(9, len(acc)), avg_acc, linewidth=2, color=colors[name], label="Moving Avg (10)")
    ax.axhline(y=np.mean(acc), linestyle='--', color='gray', label=f"Mean: {np.mean(acc):.1f}%")

    ax.set_title(f"{name} — Accuracy (% of Max Score)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fname = f"{name.replace(' ', '_').lower()}_accuracy.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")

# --- Combined comparison plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for name in results:
    axes[0].plot(results[name]["scores"], label=name)
axes[0].set_title("Episode Rewards")
axes[0].legend()

for name in results:
    acc = results[name]["accuracy"]
    avg = np.convolve(acc, np.ones(10)/10, mode='valid')
    axes[1].plot(avg, label=name)
axes[1].set_title("Accuracy Moving Avg (10)")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

for name in results:
    axes[2].plot(results[name]["losses"], label=name)
axes[2].set_title("Training Loss")
axes[2].legend()

plt.tight_layout()
plt.savefig("dqn_comparison.png", dpi=100)
plt.close()

# --- All 3 Algorithms Accuracy in ONE Graph ---
fig, ax = plt.subplots(figsize=(12, 6))

for name in results:
    acc = results[name]["accuracy"]
    avg_acc = np.convolve(acc, np.ones(10)/10, mode='valid')
    ax.plot(acc, alpha=0.2, color=colors[name])
    ax.plot(range(9, len(acc)), avg_acc, linewidth=2.5, color=colors[name], label=f"{name} (Mean: {np.mean(acc):.1f}%)")

ax.set_title("All 3 DQN Algorithms — Accuracy Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("all_algorithms_accuracy.png", dpi=150)
plt.close()
print("Saved: all_algorithms_accuracy.png")

print("\nDone! All plots saved.")