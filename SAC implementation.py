import gymnasium as gym
import torch
from torch import nn
import numpy as np
import copy
import math

reg_alpha = 0.2
gamma = 0.9

#i may or may not use this
class EnvWrapper(gym.Env):
    def __init__(self, render_mode='human'):
        self.env = gym.make("Ant-v5", render_mode=render_mode)

#env = gym.make("Ant-v5", render_mode="human", healthy_z_range=(0.3, 3))
env = gym.make("Ant-v5", healthy_z_range=(0.3, 3))

class ReplayBuffer():
    def __init__(self):
        self.storage = []
    
    def add_sample(self, sample):
        self.storage.append(sample)
        
    def get_random_samples(self):
        all_samples = copy.copy(self.storage)
        np.random.shuffle(all_samples)
        return all_samples[:200]
           
#take state-action pairs and predict the reward
class AgentFunctions():
    def __init__(self):
        #output layer is size 16 for the mean and std of each of the 8 actions, so that we can draw from a distribution, in this case normal
        self.policy = nn.Sequential(nn.Linear(27, 21), nn.Tanh(), nn.Linear(21, 16))
        self.qfunction1 = nn.Sequential(nn.Linear(35, 18), nn.Tanh(), nn.Linear(18, 1), nn.Tanh())
        self.qfunction2 = nn.Sequential(nn.Linear(35, 18), nn.Tanh(), nn.Linear(18, 1), nn.Tanh())
        self.targetqfunction1 = copy.deepcopy(self.qfunction1)
        self.targetqfunction2 = copy.deepcopy(self.qfunction2)
        
    def get_policy_mean_stds(self, state):
        action_mean_std = self.policy(state)
        action_mean_std = torch.Tensor.detach(action_mean_std).numpy()
        action_means = action_mean_std[:8]
        #because we are learning the logs of the stds rather than the actual stds, we exponentiate them first
        action_stds = np.exp(action_mean_std[8:])
        
        return action_means, action_stds
        
    def select_policy_action(self, state):
        means, stds = self.get_policy_mean_stds(state)
        
        action = np.random.normal(means, stds, 8)
        
        return action
    
    def update_policy(self):
        pass
        
    def calculate_target_values(self, samples):
        #the samples are the random samples from before, (s,a,r,s',d)
        target_values = []
        for s in samples:
            reward = s[2]
            next_state = s[3]
            done = s[4]
            
            if done:
                target_values.append(reward)
            else:
                next_state = torch.from_numpy(next_state)
                means, stds = self.get_policy_mean_stds(next_state)
                next_action = np.random.normal(means, stds, 8)
                sa_values = np.concatenate([next_state, next_action], dtype=np.float32)
                sa_values = torch.from_numpy(sa_values)
                q1_value = torch.Tensor.detach(self.targetqfunction1(sa_values)).numpy()[0]
                q2_value = torch.Tensor.detach(self.targetqfunction2(sa_values)).numpy()[0]
                
                #print(0.5 * np.log(2 * np.pi * np.square(stds)))
                
                action_entropy = np.sum(np.add((0.5 * np.log(2 * np.pi * np.square(stds))), 0.5))
                
                target = reward + (gamma * min(q1_value, q2_value) + (reg_alpha * action_entropy))
                
                print(reward, q1_value, q2_value, action_entropy, target)
                
                target_values.append(target)
                
        return target_values
    
    def update_q_functions(self, targets, predicted):
        #this is MSE then backpropagation on the q-networks, qfunction1 and qfunction2
        pass
        
replay_buffer = ReplayBuffer()
agent = AgentFunctions()

no_episodes = 20
state = env.reset()[0]

for i in range(no_episodes):
    terminated = False
    print(i)
    while not terminated:
        # Render the environment
        #env.render()

        # Take a random action
        #action = env.action_space.sample()
        state_inputs = np.float32(state[:27])
        state_inputs = torch.from_numpy(state_inputs)
        action = agent.select_policy_action(state_inputs)
        
        new_state, reward, terminated, truncated, info = env.step(action)
        
        replay_buffer.add_sample((state_inputs, action, reward, np.float32(new_state[:27]), terminated))

        state = new_state

        if terminated:
            # Reset the environment if the episode is done
            state = env.reset()[0]
            i += 1
            
random_samples = replay_buffer.get_random_samples()
target_values = agent.calculate_target_values(random_samples)

# Close the environment
env.reset()
env.close()