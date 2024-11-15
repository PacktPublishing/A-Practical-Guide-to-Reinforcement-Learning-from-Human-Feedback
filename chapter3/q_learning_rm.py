import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gridworld import GridWorld
from reward_models import RewardModel

# Subclass with Neural Network-based Reward Model
class GridWorldWithNNReward(GridWorld):
    def __init__(self, grid_size, obstacle_positions, reward_model_path):
        super().__init__(grid_size, obstacle_positions)
        self.reward_model = self.load_reward_model(reward_model_path)
        self.reward_model.eval()

    def load_reward_model(self, path):
        model = RewardModel(state_size=2, hidden_size=64, output_size=3)
        model.load_state_dict(torch.load(path))
        return model

    def q_learning_with_nn_reward(self, num_episodes=1000, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, policy_checkpoints=None):
        rewards = []
        steps_per_episode = []
        policy_progress = []
        if policy_checkpoints is None:
            policy_checkpoints = [num_episodes // 1000, num_episodes // 100, num_episodes // 50, num_episodes - 1]

        for episode in range(1, num_episodes+1):
            state = self.start_position
            total_reward = 0
            steps = 0
            visited_states = [state]
            while state!= self.goal_position:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.num_actions)
                else:
                    action = np.argmax(self.Q_values[self.state_to_index(state)])

                next_state = self.get_next_state(state, self.actions[action])
                
                tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                tensor_next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                rewards_nn = self.reward_model(tensor_state, tensor_next_state)
                progress_reward, safety_reward, goal_efficiency_reward = rewards_nn.squeeze(0).detach().numpy()
                reward = progress_reward +  safety_reward + goal_efficiency_reward
                reward = np.clip(reward, -5, 10)  # Clip reward

                target = reward + discount_factor * np.max(self.Q_values[self.state_to_index(next_state)])
                self.Q_values[self.state_to_index(state), action] += learning_rate * (target - self.Q_values[self.state_to_index(state), action])

                total_reward += reward
                state = next_state
                steps += 1
                visited_states.append(state)

            rewards.append(total_reward)
            steps_per_episode.append(steps)

            if episode in policy_checkpoints:
                policy_progress.append((self.get_optimal_policy(), visited_states, episode))

            if episode % 20 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")

        return rewards, steps_per_episode, policy_progress



if __name__ == "__main__":
    grid_size = (6, 6)
    goal_position = (5, 5)
    obstacle_positions = [(2, 3), (3, 1), (4, 3)]
    reward_model_path = "reward_model.pth"

    # Initialize GridWorld with NN-based reward model
    grid_world_nn = GridWorldWithNNReward(grid_size, obstacle_positions, reward_model_path)

    # Run Q-learning with neural network-based reward model
    rewards, steps_per_episode, policy_progress = grid_world_nn.q_learning_with_nn_reward(num_episodes=500, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)

    # Plot the learning curve, steps per episode, and policy progress
    grid_world_nn.plot_rewards(rewards)
    grid_world_nn.plot_steps_per_episode(steps_per_episode)
    grid_world_nn.plot_policy_progress(policy_progress)