from collections import deque
import random
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.clip_grad as clip_grad
from torch.distributions import Normal


ALPHA = 0.2
GAMMA = 0.99
POLYAK = 0.9

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
        self.storage.append((state, action, reward, new_state, done))


    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, new_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, new_states, dones


class QNetwork(nn.Module):
    def __init__(self, state_dim=27, action_dim=8, hidden_dim=18):
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
    def __init__(self, state_dim=27, action_dim=8):
        # output layer is size 16 for the mean and std of each of the 8 actions,
        # so that we can draw from a distribution, in this case normal

        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 21),
            nn.ReLU(),
            nn.Linear(21, action_dim * 2),
            nn.Softmax()
        )

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters())
        self.q2_optimizer = optim.Adam(self.q2.parameters())
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        #separate target q-networks, initialised with the same parameters, but will eventually be different
        self.qtarget1 = copy.deepcopy(self.q1).to(device)
        self.qtarget2 = copy.deepcopy(self.q2).to(device)


    def get_policy_mean_stds(self, state):
        mean_std = self.policy(state)
        means = mean_std[:, :8]
        log_stds = mean_std[:, 8:]
        stds = torch.exp(log_stds)
        return means, stds, log_stds


    def select_policy_action(self, state):
        with torch.no_grad():
            means, stds, _ = self.get_policy_mean_stds(state.unsqueeze(0))
            #print(means[0][0], stds[0][0])
            action = Normal(means, stds).sample()
            action = torch.tanh(action)
        return action.squeeze()


    def update_policy(self, states, rewards, next_states):
        states = torch.from_numpy(states).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        means, stds, _ = self.get_policy_mean_stds(states)

        dist = Normal(means, stds)
        sampled_actions = dist.rsample()
        log_probs = dist.log_prob(sampled_actions).sum(dim=1, keepdim=True)

        q1_val = self.qtarget1(states, sampled_actions)
        q2_val = self.qtarget2(states, sampled_actions)
        q_min = torch.min(q1_val, q2_val)
        
        #treating the q-network as a value network by setting action values to 0
        v1_s_val = self.q1(states, torch.from_numpy(np.zeros((len(states), 8))).float())
        v2_s_val = self.q2(states, torch.from_numpy(np.zeros((len(states), 8))).float())
        
        v1_s_next_val = self.q1(next_states, torch.from_numpy(np.zeros((len(states), 8))).float())
        v2_s_next_val = self.q2(next_states, torch.from_numpy(np.zeros((len(states), 8))).float())
        
        v_s_val_min = torch.min(v1_s_val, v2_s_val)
        v_s_next_val_min = torch.min(v1_s_next_val, v2_s_next_val)
        
        advantage = torch.add(rewards, torch.subtract(torch.multiply(v_s_next_val_min, GAMMA), v_s_val_min))

        #policy_loss = (ALPHA * log_probs - q_min).mean()
        policy_loss = (((ALPHA - advantage) * log_probs) - q_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()


    def update_q_functions(self, samples):
        states, actions, rewards, new_states, dones = samples

        states      = torch.from_numpy(states).float().to(device)
        actions     = torch.from_numpy(actions).float().to(device)
        rewards     = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        new_states  = torch.from_numpy(new_states).float().to(device)
        dones       = torch.from_numpy(dones).float().unsqueeze(1).to(device)

        with torch.no_grad():
            means, stds, _ = self.get_policy_mean_stds(new_states)
            dist = Normal(means, stds)
            new_actions = torch.tanh(dist.rsample())
            log_probs = dist.log_prob(new_actions).sum(dim=1, keepdim=True)

            q1_new = self.q1(new_states, new_actions)
            q2_new = self.q2(new_states, new_actions)
            q_min = torch.min(q1_new, q2_new)

            targets = rewards + GAMMA * (1 - dones) * (q_min - ALPHA * log_probs)
            
            #update target q network parameters using reparameterization trick
            for target_param, param in zip(self.qtarget1.parameters(), self.q1.parameters()):
                target_param.data.copy_((POLYAK * target_param) - ((1 - POLYAK) * param))

        q1_pred = self.q1(states, actions)
        q1_loss = nn.functional.mse_loss(q1_pred, targets)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        q2_pred = self.q2(states, actions)
        q2_loss = nn.functional.mse_loss(q2_pred, targets)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return q1_loss.item(), q2_loss.item()


#env = gym.make("Ant-v5", healthy_z_range=(0.3, 3))
env = gym.make("Ant-v5", render_mode="human", healthy_z_range=(0.3, 3))

buffer = ReplayBuffer(capacity=1_000_000)
agent = Agent()

batch_size = 256
num_episodes = 20
state, _ = env.reset()

for i in range(num_episodes):
    print(f"Episode {i}")

    done = False
    while not done:
        # env.render()

        state_inputs = torch.from_numpy(state[:27]).float().to(device)
        action = agent.select_policy_action(state_inputs)

        new_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = terminated or truncated

        buffer.push(state[:27], action, reward, new_state[:27], done)
        state = new_state

        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            agent.update_q_functions(batch)
            agent.update_policy(batch[0], batch[2], batch[3])

        if done:
            state, _ = env.reset()

env.close()
