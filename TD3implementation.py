import gymnasium as gym
import torch
import matplotlib.pyplot as plt
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('torch cuda build', torch.version.cuda)
import torch.nn as neural_network
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from collections import deque

EPISODES_NUMBER = 300
MAXIMUM_STEPS = 1000
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
POLICY_DELAY = 2
NOISE_STD = 0.1
TARGET_NOISE_STD = 0.2
TARGET_NOISE_CLIP = 0.5
BUFFER_SIZE = 1000000
TENSOR_TYPE = torch.float32 #float64 works as default but is slower
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # safe speedup on recent PyTorch
#FALL_PENALTY = -10.0  
#MINIMUM_TORSO_Z = 0.275

# ---- Training setup ----
def get_tensor(x, device=DEVICE):
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=TENSOR_TYPE)


class ReplayBuffer:
    def __init__(self, maximum_size):
        #double-ended queue holds most recent states for buffer
        self.buffer = deque(maxlen=maximum_size)

    def store(self, state, action, step_reward, next_state, is_done):
        self.buffer.append((state, action, step_reward, next_state, is_done))
    
    def sample_batch(self, batch_size):
        # randomly select from memory
        batch = random.sample(self.buffer, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        is_dones = []

        for transition in batch:
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            is_dones.append(transition[4])

        return (
            get_tensor(states),
            get_tensor(actions),
            get_tensor(rewards).unsqueeze(1),
            get_tensor(next_states),
            get_tensor(is_dones).unsqueeze(1)
        )


    def buffer_size(self):
        return len(self.buffer)

class Actor(neural_network.Module):
    def __init__(self, observation_space_size, action_space_size, max_act_size):
        super().__init__()
        self.max_act_size = max_act_size
        # network struct: observation -> 256 -> 256 -> action
        self.nn = neural_network.Sequential(
            neural_network.Linear(observation_space_size, 256), neural_network.ReLU(),
            neural_network.Linear(256, 256), neural_network.ReLU(),
            neural_network.Linear(256, action_space_size), neural_network.Tanh() # keep abolute value at 1 or less
        ) 

    def forward(self, observation):
        return self.max_act_size * self.nn(observation)
    
class Critic(neural_network.Module):
    def __init__(self, observation_space_size, action_space_size):
        super().__init__()
        # network struct: observation (befor action) and action -> 256 -> 256 -> 1 (q-value)
        self.nn = neural_network.Sequential(
            neural_network.Linear(observation_space_size + action_space_size, 256), neural_network.ReLU(),
            neural_network.Linear(256, 256), neural_network.ReLU(),
            neural_network.Linear(256, 1)
        )

    def forward(self, observation, action):
        nn_input = torch.cat([observation, action], dim=1)
        return self.nn(nn_input)

def action_from_actor(actor, state_tensor):
    action = actor(state_tensor).detach().cpu().numpy()[0]
    return action

def select_action(actor, state, noise_std):
    state_tensor = get_tensor(state).unsqueeze(0)
    action = action_from_actor(actor, state_tensor)
    
    action = action + np.random.normal(0, noise_std, size=action.shape) # add random noise to every element of action for exploration
    return np.clip(action, -actor.max_act_size, actor.max_act_size) # keep action in accepted limits of environment

# move target towards source
def soft_update(target, source, tau):
    target_params = list(target.parameters())
    source_params = list(source.parameters())

    for i in range(len(target_params)):
        target_params[i].data = tau * source_params[i].data + (1 - tau) * target_params[i].data

def get_target_q(reward, next_state, is_done, actor_target, critic_target1, critic_target2, gamma, noise_std, noise_clip):
    with torch.no_grad(): #don't store gradients for efficiency
        next_action = actor_target(next_state)
        
        random_noise = torch.randn_like(next_action) * noise_std
        noise = torch.clamp(random_noise, -noise_clip, noise_clip)

        next_action = torch.clamp(next_action + noise, -actor_target.max_act_size, actor_target.max_act_size)

        critic1_q = critic_target1(next_state, next_action)
        critic2_q = critic_target2(next_state, next_action)
        min_q = torch.min(critic1_q, critic2_q)

        return reward + gamma * (1 - is_done) * min_q

    
def update_critics(critic1, critic2, critic1_target, critic2_target, actor_target, optimiser1, optimiser2, batch):
    
    states, actions, rewards, next_states, dones = batch

    target_q = get_target_q(rewards, next_states, dones, actor_target, critic1_target, critic2_target, GAMMA, TARGET_NOISE_STD, TARGET_NOISE_CLIP)

    # compare q values with MSE
    q_val_1 = critic1(states, actions)
    q_val_2 = critic2(states, actions)

    mse = neural_network.MSELoss()
    mse1 = mse(q_val_1, target_q)
    mse2 = mse(q_val_2, target_q)

    optimiser1.zero_grad()
    optimiser2.zero_grad()
    mse1.backward()
    mse2.backward()
    
    optimiser1.step()
    optimiser2.step()

def update_actor(actor, critic1, optimiser, batch):
    states = batch[0]
    loss = -critic1(states, actor(states)).mean()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# ---- Environment -----
env = gym.make("Ant-v5", render_mode=None)
observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.shape[0]
max_act_size = env.action_space.high[0]

actor = Actor(observation_space_size, action_space_size, max_act_size).to(DEVICE)
actor_target = Actor(observation_space_size, action_space_size, max_act_size).to(DEVICE)
critic1 = Critic(observation_space_size, action_space_size).to(DEVICE)
critic2 = Critic(observation_space_size, action_space_size).to(DEVICE)
critic1_target = Critic(observation_space_size, action_space_size).to(DEVICE)
critic2_target = Critic(observation_space_size, action_space_size).to(DEVICE)

actor_target.load_state_dict(actor.state_dict())
critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())

actor_optimiser = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
critic1_optimiser = optim.Adam(critic1.parameters(), lr=CRITIC_LEARNING_RATE)
critic2_optiniser = optim.Adam(critic2.parameters(), lr=CRITIC_LEARNING_RATE)

replay_buffer = ReplayBuffer(BUFFER_SIZE)
total_step_count = 0

# ---- run once before training, with rendering ----
def run_rendered(actor):
    rendered_env = gym.make("Ant-v5", render_mode="human")
    state = rendered_env.reset()[0]

    for _ in range(MAXIMUM_STEPS):
        # actors picks action
        state_tensor = get_tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = action_from_actor(actor, state_tensor)

        # take action
        next_state, reward, is_terminating, forcibly_ended, _ = rendered_env.step(action)
        state = next_state

        rendered_env.render()

        if is_terminating or forcibly_ended:
            state = rendered_env.reset()[0]

    rendered_env.close()

print("Showing render of untrained model...")
run_rendered(actor)

# ---- training ----

start_time = datetime.now()
print(f"Training start time: {start_time} | device: {DEVICE}")
episode_rewards_over_time = []

for episode in range(EPISODES_NUMBER):
    state = env.reset()[0]
    episode_reward = 0

    for step in range(MAXIMUM_STEPS):
        # pick and take action, get next state and info
        action = select_action(actor, state, NOISE_STD)
        next_state, reward, is_terminating, forcibly_ended, _ = env.step(action)

        # note that the floor seemingly has a height of -1
        
        #torso_height = next_state[0]  # Ant's torso height in observation
#        if torso_height < MIN_TORSO_HEIGHT:
 #           reward += FALL_PENALTY
  #          is_done = True
   #     else:
        is_done = is_terminating or forcibly_ended

        # fill buffer and update values
        replay_buffer.store(state, action, reward, next_state, is_done)
        state = next_state
        episode_reward += reward
        total_step_count += 1

        # once the buffer is big enough, learn from it
        if replay_buffer.buffer_size() > BATCH_SIZE:
            batch = replay_buffer.sample_batch(BATCH_SIZE)

            update_critics(critic1, critic2, critic1_target, critic2_target, actor_target,critic1_optimiser, critic2_optiniser, batch)

            if total_step_count % POLICY_DELAY == 0:
                update_actor(actor, critic1, actor_optimiser, batch)
                soft_update(actor_target, actor, TAU)
                soft_update(critic1_target, critic1, TAU)
                soft_update(critic2_target, critic2, TAU)

        if is_done:
            break

    episode_rewards_over_time.append(episode_reward)
    print(f"Episode Number: {episode + 1} | Reward: {episode_reward}")

end_time = datetime.now()
print("Training start time: " + str(start_time))
print("Training end time: " + str(end_time))
print("Total training time: " + str(end_time - start_time))

plt.figure(figsize=(12,6))
plt.plot(episode_rewards_over_time, label='Episode Reward')
window = 12
if len(episode_rewards_over_time) >= window:
    smoothed = np.convolve(episode_rewards_over_time, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(episode_rewards_over_time)), smoothed, label=f"{window}-ep moving avg")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.legend()
plt.show()

#---- View After Training (With Rendering) ----

entry = ""
while entry != "e":
    entry = input("enter \"e\" to exit, or anything else to view the trained model\n")
    if entry != "e":
        run_rendered(actor)

env.reset()
env.close()














