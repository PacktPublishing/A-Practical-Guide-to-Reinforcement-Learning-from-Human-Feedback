import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import logging

# Set up the environment
env = gym.make('MountainCar-v0',render_mode="human")
# env = gym.make('MountainCar-v0')
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the Q-network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_net = QNetwork(input_dim, output_dim)
target_net = QNetwork(input_dim, output_dim)
target_net.load_state_dict(q_net.state_dict())  # Initialize target network with same weights
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# Modularized functions

def log_training_info(episode, action, reward, state, next_state, done, trunc):
    logging.info(f"Episode: {episode}, Action: {action}, Reward: {reward}, "
                 f"State: {state}, Next State: {next_state}, Done: {done}, Trunc: {trunc}")

def initialize_environment(env):
    state, _ = env.reset()
    total_reward = 0
    done = False
    return state, total_reward, done

def select_action(q_net, state, epsilon, env):
    if random.random() > epsilon:
        with torch.no_grad():
            action = q_net(state).max(1)[1].item()
    else:
        action = env.action_space.sample()
    return action

def store_experience(memory, state, action, reward, next_state, done):
    memory.push((state, action, reward, next_state, done))

def update_q_network(q_net, target_net, memory, optimizer, loss_fn, gamma, batch_size):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    
    # Convert to numpy arrays to avoid slow tensor creation from lists
    batch_state_np = np.array(batch_state, dtype=np.float32)
    batch_action_np = np.array(batch_action, dtype=np.int64)
    batch_reward_np = np.array(batch_reward, dtype=np.float32)
    batch_next_state_np = np.array(batch_next_state, dtype=np.float32)
    batch_done_np = np.array(batch_done, dtype=np.float32)
    
    # Convert numpy arrays to tensors
    batch_state_tensor = torch.from_numpy(batch_state_np)
    batch_action_tensor = torch.from_numpy(batch_action_np).unsqueeze(1)
    batch_reward_tensor = torch.from_numpy(batch_reward_np)
    batch_next_state_tensor = torch.from_numpy(batch_next_state_np)
    batch_done_tensor = torch.from_numpy(batch_done_np)
    

    current_q_values = q_net(batch_state_tensor).gather(1, batch_action_tensor)
    max_next_q_values = target_net(batch_next_state_tensor).max(1)[0].detach()
    expected_q_values = batch_reward_tensor + (gamma * max_next_q_values * (1 - batch_done_tensor))

    loss = loss_fn(current_q_values.squeeze(), expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop for DQN
def update_target_network(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())

def decay_epsilon(epsilon, epsilon_end, epsilon_start, episode, epsilon_decay):
    return max(epsilon_end, epsilon_start - episode / epsilon_decay)

def train_dqn(q_net, target_net, env, optimizer, loss_fn, num_episodes=1000, gamma=0.99, 
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500, batch_size=32, target_update_freq=10):
    memory = ReplayMemory(capacity=10000)
    epsilon = epsilon_start
    rewards = []

    for episode in range(num_episodes):
        state, total_reward, done = initialize_environment(env)

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = select_action(q_net, state_tensor, epsilon, env)
            next_state, reward, done, trunc, info = env.step(action)
            done = done or trunc
            log_training_info(episode, action, reward, state, next_state, done, trunc)

            store_experience(memory, state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            update_q_network(q_net, target_net, memory, optimizer, loss_fn, gamma, batch_size)

        # Update the target network every 'target_update_freq' episodes
        if episode % target_update_freq == 0:
            update_target_network(q_net, target_net)

        epsilon = decay_epsilon(epsilon, epsilon_end, epsilon_start, episode, epsilon_decay)
        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}')

    return rewards

# Training loop for REINFORCE

# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

# Plotting function
def plot_rewards(rewards):
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()

# # Training the model
# rewards = train_dqn(q_net, env, optimizer, loss_fn)

# # Saving the model
# save_model(q_net, 'q_network.pth')

# # Plotting training rewards
# plot_rewards(rewards)

# Inference
def inference(env, model, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].item()
            next_state, reward, done, trunc, info = env.step(action)
            done = done or trunc
            total_reward += reward
            state = next_state
        print(f'Episode {episode + 1}, Total Reward: {total_reward}')
    env.close()

# Load and run inference
# load_model(q_net, 'q_network.pth')
# inference(env, q_net)

if __name__ == "__main__":
    print(args.train)
    if args.train:
        rewards = train_dqn(q_net, target_net, env, optimizer, loss_fn)
        # # Saving the model
        save_model(q_net, 'q_network_tgt_test.pth')

        # # Plotting training rewards
        plot_rewards(rewards)
    else:
        load_model(q_net, 'q_network_tgt_test.pth')
        inference(env,q_net, episodes=10)