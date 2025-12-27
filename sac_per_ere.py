import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


RENDER = False
WINDOW = 12
ALPHA = 0.1
GAMMA = 0.99
LOG_STD_MIN = -20
LOG_STD_MAX = 2
BATCH_SIZE = 256
NUM_EPISODES = 100
UPDATES_PER_STEP = 1
LEARNING_STARTS = 10_000
MAX_EPISODE_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ere_indices(buffer_len, step, eta=0.999, batch_size=256):
    ck = int(buffer_len * (eta ** step))
    ck = max(ck, batch_size)
    return np.arange(buffer_len - ck, buffer_len)


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.6, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.storage = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.next_index = 0


    def __len__(self):
        return len(self.storage)


    def push(self, state, action, reward, new_state, done):
        if len(self.storage) < self.capacity:
            self.storage.append((state, action, reward, new_state, done))
        else:
            self.storage[self.next_index] = (state, action, reward, new_state, done)

        # max priority for new transitions
        if len(self.storage) > 1:
            self.priorities[self.next_index] = self.priorities.max()
        else:
            self.priorities[self.next_index] = 1.0

        # index wraparound if storage is full
        self.next_index = (self.next_index + 1) % self.capacity


    def sample(self, batch_size, indices=None):
        if indices is None:
            indices = np.arange(len(self.storage))

        probs = self.priorities[indices] ** self.alpha
        probs /= probs.sum()

        sampled_indices = np.random.choice(indices, batch_size, p=probs)
        batch = [self.storage[i] for i in sampled_indices]

        weights = (len(indices) * probs[np.searchsorted(indices, sampled_indices)]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, new_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, new_states, dones, sampled_indices, weights


    def update_priorities(self, indices, td_errors):
        # td = temporal difference
        for i, td in zip(indices, td_errors):
            self.priorities[i] = abs(td) + self.epsilon


class QNetwork(nn.Module):
    def __init__(self, state_dim=27, action_dim=8, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class Agent(nn.Module):
    def __init__(self, state_dim=27, action_dim=8, hidden_dim=256, tau=0.005):
        # output layer is size 16 for the mean and std of each of the 8 actions,
        # so that we can draw from a distribution, in this case normal

        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        ).to(device)

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.action_dim = action_dim
        self.tau = 0.005


    def get_policy_mean_stds(self, state):
        mean_std = self.policy(state)
        means = mean_std[:, :self.action_dim]
        log_stds = mean_std[:, self.action_dim:]
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = torch.exp(log_stds)
        return means, stds, log_stds


    def sample_action_and_log_prob(self, states):
        means, stds, _ = self.get_policy_mean_stds(states)
        dist = Normal(means, stds)

        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=1, keepdim=True)

        return actions, log_probs


    def select_action(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            actions, _ = self.sample_action_and_log_prob(state)
        return actions.squeeze(0)


    def update_policy(self, states):
        states = torch.from_numpy(states).float().to(device)

        actions, log_probs = self.sample_action_and_log_prob(states)

        q_min = torch.min(
            self.q1(states, actions),
            self.q2(states, actions),
        )

        policy_loss = (ALPHA * log_probs - q_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()


    def soft_update(self, source, target):
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)


    def update_q(self, batch, weights):
        states, actions, rewards, new_states, dones = batch

        states      = torch.from_numpy(states).float().to(device)
        actions     = torch.from_numpy(actions).float().to(device)
        rewards     = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        new_states  = torch.from_numpy(new_states).float().to(device)
        dones       = torch.from_numpy(dones).float().unsqueeze(1).to(device)
        weights     = torch.from_numpy(weights).float().unsqueeze(1).to(device)

        with torch.no_grad():
            new_actions, log_probs = self.sample_action_and_log_prob(new_states)

            q_min = torch.min(
                self.q1_target(new_states, new_actions),
                self.q2_target(new_states, new_actions),
            )

            targets = rewards + GAMMA * (1 - dones) * (q_min - ALPHA * log_probs)

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        td_error1 = q1_pred - targets
        td_error2 = q2_pred - targets
        q1_loss = (weights * td_error1.pow(2)).mean()
        q2_loss = (weights * td_error2.pow(2)).mean()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        td_errors = (td_error1.abs() + td_error2.abs()) / 2
        return td_errors.detach().cpu().numpy().squeeze()


def moving_average(x, window):
    return [np.mean(x[i:i+window]) for i in range(len(x)-window+1)]


if RENDER:
    env = gym.make("Ant-v5", render_mode="human", healthy_z_range=(0.3, 3))
else:
    env = gym.make("Ant-v5", healthy_z_range=(0.3, 3))

buffer = ReplayBuffer(capacity=1_000_000)
agent = Agent(state_dim=105, action_dim=8)

episode_rewards = []
ere_step = 0
global_step = 0
state, _ = env.reset()

for i in range(NUM_EPISODES):
    print(f"Episode {i}")

    step_in_episode = 0
    episode_reward = 0.0
    done = False
    while not done:
        if RENDER:
            env.render()

        step_in_episode += 1
        global_step += 1

        state_inputs = torch.from_numpy(state).float().to(device)
        action = agent.select_action(state_inputs)

        new_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        episode_reward += float(reward)
        done = terminated or truncated

        buffer.push(
            state,
            action.cpu().numpy(),
            reward,
            new_state,
            done
        )

        state = new_state

        if len(buffer) > BATCH_SIZE and global_step > LEARNING_STARTS:
            for _ in range(UPDATES_PER_STEP):
                indices = ere_indices(len(buffer), ere_step, batch_size=BATCH_SIZE)

                (states, actions, rewards, new_states, dones,
                sampled_indices, weights) = buffer.sample(BATCH_SIZE, indices)

                td_errors = agent.update_q(
                    (states, actions, rewards, new_states, dones),
                    weights,
                )

                agent.update_policy(states)
                buffer.update_priorities(sampled_indices, td_errors)
                ere_step += 1

        if step_in_episode >= MAX_EPISODE_STEPS:
            done = True

        if done:
            episode_rewards.append(episode_reward)
            state, _ = env.reset()

env.close()

plt.figure(figsize=(15, 4))
plt.plot(episode_rewards, label="Episode Rewards", alpha=0.5)
plt.plot(
    range(WINDOW - 1, len(episode_rewards)),
    moving_average(episode_rewards, WINDOW),
    label="12-Episode Average",
    linewidth=2,
)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()
plt.show()
