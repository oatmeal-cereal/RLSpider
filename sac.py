from collections import deque
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


ALPHA = 0.2
GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# may or may not be used
class EnvWrapper(gym.Env):
    def __init__(self, render_mode="human"):
        self.env = gym.make("Ant-v5", render_mode=render_mode)


class ReplayBuffer:
    def __init__(self, capacity):
        self.storage = deque(maxlen=capacity)


    def __len__(self):
        return len(self.storage)


    def push(self, state, action, reward, new_state, done):
        self.storage.append((
            np.array(state[:27], dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array(new_state[:27], dtype=np.float32),
            np.array([done], dtype=np.float32),
        ))


    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, new_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, new_states, dones


class QNetwork(nn.Module):
    def __init__(self, state_dim=27, action_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
        )


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class Agent(nn.Module):
    def __init__(self, state_dim=27, action_dim=8):
        # output layer is size 16 for the mean and std of each of the 8 actions,
        # so that we can draw from a distribution, in this case normal

        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 21),
            nn.ReLU(),
            nn.Linear(21, 16),
            nn.ReLU(),
        )

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)


    def get_policy_mean_stds(self, state):
        mean_std = self.policy(state)
        mean_std = torch.Tensor.detach(mean_std).numpy()
        means = mean_std[:8]
        # because we are learning the logs of the stds rather than the
        # actual stds, we exponentiate them first
        stds = np.exp(mean_std[8:])

        return means, stds


    def select_policy_action(self, state):
        means, stds = self.get_policy_mean_stds(state)
        action = np.random.normal(means, stds)
        return action


    def update_policy(self):
        pass


    def calculate_target_values(self, samples):
        # the samples are the random samples from before, (s,a,r,s',d)
        target_values = []

        for s in samples:
            _, _, reward, next_state, done = s

            if done:
                target_values.append(reward)
                continue

            next_state = torch.from_numpy(next_state)
            means, stds = self.get_policy_mean_stds(next_state)
            next_action = np.random.normal(means, stds)

            sa_values = np.concatenate([next_state, next_action], dtype=np.float32)
            sa_values = torch.from_numpy(sa_values)

            q1_value = self.q1(next_state, next_action)
            q2_value = self.q2(next_state, next_action)
            q_min = min(q1_value, q2_value)

            # print(0.5 * np.log(2 * np.pi * np.square(stds)))

            action_entropy = np.sum(
                np.add((0.5 * np.log(2 * np.pi * np.square(stds))), 0.5)
            )

            target = reward + (GAMMA * q_min - (ALPHA * action_entropy))

            print(reward, q1_value, q2_value, action_entropy, target)

            target_values.append(target)

        return np.array(target_values)


    def update_q_functions(self, targets, predicted):
        # this is MSE then backpropagation on the q-networks, qfunction1 and qfunction2
        pass


env = gym.make("Ant-v5", healthy_z_range=(0.3, 3))
# env = gym.make("Ant-v5", render_mode="human", healthy_z_range=(0.3, 3))

buffer = ReplayBuffer(capacity=1_000_000)
agent = Agent()

num_episodes = 20
state, _ = env.reset()

for i in range(num_episodes):
    done = False
    print(f"Episode {i}")

    while not done:
        # env.render()

        state_inputs = torch.from_numpy(state[:27]).float()
        action = agent.select_policy_action(state_inputs)
        # action = env.action_space.sample()  # random action

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, new_state, done)
        state = new_state

        if done:
            state, _ = env.reset()

target_values = agent.calculate_target_values(buffer.sample(batch_size=200))

env.close()
