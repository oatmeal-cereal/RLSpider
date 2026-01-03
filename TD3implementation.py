import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import torch.nn as neural_network
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from collections import deque
import multiprocessing as mp
import time

TENSOR_TYPE = torch.float32 #float64 works as default but is slower
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # safe speedup on recent PyTorch

EPISODES_NUMBER = 2000
MAXIMUM_STEPS = 500
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
POLICY_DELAY = 2
NOISE_STD = 0.1
TARGET_NOISE_STD = 0.2
TARGET_NOISE_CLIP = 0.5
BUFFER_SIZE = 1000000

NUM_ENVS = 8
LEARNING_STARTS = 10000
UPDATES_PER_STEP = 1
VEC_MODE = "async"  # "async" or "sync"
TRAIN_EVERY = 4          # do training once every N vector-steps
UPDATES_PER_TRAIN = 1    # number of critic updates per train event
#FALL_PENALTY = -10.0
#MINIMUM_TORSO_Z = 0.275

# ---- Training setup ----
def get_tensor(x, device=DEVICE):
    if isinstance(x, np.ndarray) and x.dtype == np.float32:
        arr = x
    else:
        arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=TENSOR_TYPE)

class ReplayBuffer:
    def __init__(self, maximum_size, obs_dim, act_dim):
        self.max_size = int(maximum_size)
        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store_batch(self, states, actions, rewards, next_states, dones):
        n = states.shape[0]
        idxs = (np.arange(n) + self.ptr) % self.max_size

        self.obs[idxs] = states
        self.acts[idxs] = actions
        self.rews[idxs, 0] = rewards
        self.next_obs[idxs] = next_states
        self.dones[idxs, 0] = dones.astype(np.float32)

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)
    
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            get_tensor(self.obs[idxs]),
            get_tensor(self.acts[idxs]),
            get_tensor(self.rews[idxs]),
            get_tensor(self.next_obs[idxs]),
            get_tensor(self.dones[idxs]),
        )

    def buffer_size(self):
        return self.size

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
    with torch.no_grad():
        action = actor(state_tensor).cpu().numpy()[0]
    return action

def select_action_batch(actor, states_np, noise_std):
    with torch.no_grad():
        states_t = get_tensor(states_np)
        actions = actor(states_t).cpu().numpy()

    if noise_std > 0:
        actions = actions + np.random.normal(0, noise_std, size=actions.shape)

    return np.clip(actions, -actor.max_act_size, actor.max_act_size)

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

mse_loss = neural_network.MSELoss()

def update_critics(critic1, critic2, critic1_target, critic2_target, actor_target, optimiser1, optimiser2, batch):
    
    states, actions, rewards, next_states, dones = batch

    target_q = get_target_q(
        rewards, next_states, dones,
        actor_target, critic1_target, critic2_target,
        GAMMA, TARGET_NOISE_STD, TARGET_NOISE_CLIP
    )

    q_val_1 = critic1(states, actions)
    q_val_2 = critic2(states, actions)

    loss1 = mse_loss(q_val_1, target_q)
    loss2 = mse_loss(q_val_2, target_q)
    loss = loss1 + loss2

    optimiser1.zero_grad(set_to_none=True)
    optimiser2.zero_grad(set_to_none=True)
    loss.backward()
    
    optimiser1.step()
    optimiser2.step()

def update_actor(actor, critic1, optimiser, batch):
    states = batch[0]
    loss = -critic1(states, actor(states)).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# ---- Environment -----
def make_env():
    def thunk():
        return gym.make("Ant-v5", render_mode=None)
    return thunk
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

def main():
    print('torch', torch.__version__)
    print('cuda available', torch.cuda.is_available())
    print('torch cuda build', torch.version.cuda)

    if VEC_MODE == "async":
        vec_env = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    else:
        vec_env = gym.vector.SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

    observation_space_size = vec_env.single_observation_space.shape[0]
    action_space_size = vec_env.single_action_space.shape[0]
    max_act_size = float(vec_env.single_action_space.high[0])

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
    critic2_optimiser = optim.Adam(critic2.parameters(), lr=CRITIC_LEARNING_RATE)

    replay_buffer = ReplayBuffer(BUFFER_SIZE, observation_space_size, action_space_size)
    total_step_count = 0


    print("Showing render of untrained model...")
    # run_rendered(actor)

    # ---- training ----

    start_time = datetime.now()
    print(f"Training start time: {start_time} | device: {DEVICE} | num_envs: {NUM_ENVS}")
    episode_rewards_over_time = []
    obs, _ = vec_env.reset()
    last_t = time.perf_counter()
    last_steps = 0

    global_step_vec = 0  # counts vector-steps (calls to vec_env.step)

    for episode in range(EPISODES_NUMBER):
        ep_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

        for step in range(MAXIMUM_STEPS):
            actions = select_action_batch(actor, obs, NOISE_STD)
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

            dones = np.logical_or(terminated, truncated)
            replay_buffer.store_batch(obs, actions, rewards, next_obs, dones)

            obs = next_obs
            ep_rewards += rewards
            total_step_count += NUM_ENVS
            global_step_vec += 1

            # throughput log unchanged
            now = time.perf_counter()
            if now - last_t >= 5.0:
                steps_done = total_step_count - last_steps
                sps = steps_done / (now - last_t)
                print(f"throughput: {sps:.0f} env-steps/s | mode={VEC_MODE} | num_envs={NUM_ENVS} | buffer={replay_buffer.buffer_size()}")
                last_t = now
                last_steps = total_step_count

            # train only every TRAIN_EVERY vector steps 
            can_train = replay_buffer.buffer_size() >= max(LEARNING_STARTS, BATCH_SIZE)
            if can_train and (global_step_vec % TRAIN_EVERY == 0):
                for _ in range(UPDATES_PER_TRAIN):
                    batch = replay_buffer.sample_batch(BATCH_SIZE)
                    update_critics(
                        critic1, critic2,
                        critic1_target, critic2_target,
                        actor_target,
                        critic1_optimiser, critic2_optimiser,
                        batch
                    )

                    if (global_step_vec % POLICY_DELAY) == 0:
                        update_actor(actor, critic1, actor_optimiser, batch)
                        soft_update(actor_target, actor, TAU)
                        soft_update(critic1_target, critic1, TAU)
                        soft_update(critic2_target, critic2, TAU)

        mean_reward = float(ep_rewards.mean())
        episode_rewards_over_time.append(mean_reward)
        print(f"Episode Number: {episode + 1} | Mean reward over {NUM_ENVS} envs: {mean_reward:.1f}")

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
        entry = input("enter \"e\" to exit, p to view the graph, or anything else to view the trained model\n")
        if entry == "p":
            plt.show()
        if entry != "e":
            run_rendered(actor)

    vec_env.close()

if __name__ == "__main__":
    mp.freeze_support() 
    main()












