import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gridworld import GridWorld
from reward_models import RewardModel

def simulate_human_reward(state, next_state, goal_position, obstacle_positions):
    next_state_array = np.array(next_state)
    goal_position_array = np.array(goal_position)
    progress_reward = -np.linalg.norm(next_state_array - goal_position_array)
    safety_reward = -5 if next_state in obstacle_positions else 0
    goal_efficiency_reward = 10 if np.array_equal(next_state_array, goal_position_array) else -1
    return np.array([progress_reward, safety_reward, goal_efficiency_reward])

def generate_human_feedback(grid_size, actions, goal_position, obstacle_positions, num_samples):
    # actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    data = []
    for _ in range(num_samples):
        state = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
        action = actions[np.random.choice(len(actions))]
        next_state = (state[0] + action[0], state[1] + action[1])
        rewards = simulate_human_reward(state, next_state, goal_position, obstacle_positions)
        data.append((state, next_state, rewards))
    return data

def train_reward_model(data, state_size=2, hidden_size=64,output_size = 2, num_epochs=100, batch_size=64, learning_rate=0.001):
    reward_model = RewardModel(state_size, hidden_size, output_size)
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        np.random.shuffle(data)
        batch_losses = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            states = torch.tensor([item[0] for item in batch], dtype=torch.float32)
            next_states = torch.tensor([item[1] for item in batch], dtype=torch.float32)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)

            optimizer.zero_grad()
            predictions = reward_model(states, next_states)
            loss = criterion(predictions, rewards)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return reward_model

if __name__ == "__main__":
    grid_size = (6, 6)
    goal_position = (5, 5)
    obstacle_positions = [(2, 3), (3, 1), (4, 3)]
    gridworld = GridWorld(grid_size, obstacle_positions)
    actions = gridworld.actions
    # Generate human feedback and train the reward model
    human_feedback_data = generate_human_feedback(grid_size, actions, goal_position, obstacle_positions, num_samples=500)
    reward_model = train_reward_model(human_feedback_data, state_size=2, hidden_size=64,output_size=3, num_epochs=5000)

    # Save the trained model
    torch.save(reward_model.state_dict(), "reward_model.pth")
    print("Reward model saved to reward_model.pth")