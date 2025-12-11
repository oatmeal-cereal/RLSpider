import gymnasium as gym

env = gym.make("Ant-v5", render_mode="human")
env.reset()

for _ in range(1000):
    # Render the environment
    env.render()

    # Take a random action
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)

    if done:
        # Reset the environment if the episode is done
        observation = env.reset()

# Close the environment
env.reset()
env.close()