import numpy as np
import matplotlib.pyplot as plt
from .plotting_utilities import (
    plot_rewards,
    plot_steps_per_episode,
    plot_policy_progress,
)

class GridWorld:
    def __init__(self, grid_size, obstacle_positions):
        self.grid_size = grid_size
        self.start_position = (2, 2)
        self.goal_position = (5,5) #(grid_size[0] - 1, grid_size[1] - 1)
        self.obstacle_positions = obstacle_positions

        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        self.action_symbols = ['→', '←', '↓', '↑']  # Action symbols for visualization
        self.num_actions = len(self.actions)
        self.num_states = np.prod(self.grid_size)
        self.Q_values = np.zeros((self.num_states, self.num_actions))

    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])

    def is_valid_state(self, state):
        return 0 <= state[0] < self.grid_size[0] and 0 <= state[1] < self.grid_size[1] \
               and state not in self.obstacle_positions

    def get_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if self.is_valid_state(next_state):
            return next_state
        return state

    def q_learning(self, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, policy_checkpoints=None):
        rewards = []
        steps_per_episode = []
        policy_progress = []  # Store optimal policy at different checkpoints
        if policy_checkpoints is None:
            policy_checkpoints = [num_episodes // 1000, num_episodes // 100, num_episodes // 50, num_episodes - 1]

        for episode in range(1, num_episodes+1):
            state = self.start_position
            total_reward = 0
            steps = 0
            visited_states = [state]
            while state != self.goal_position:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.num_actions)
                else:
                    action = np.argmax(self.Q_values[self.state_to_index(state)])

                next_state = self.get_next_state(state, self.actions[action])
                reward = -1
                if next_state == self.goal_position:
                    reward = 10
                elif next_state in self.obstacle_positions:
                    reward = -5

                target = reward + discount_factor * np.max(self.Q_values[self.state_to_index(next_state)])
                self.Q_values[self.state_to_index(state), action] += learning_rate * (target - self.Q_values[self.state_to_index(state), action])

                total_reward += reward
                state = next_state
                steps += 1
                visited_states.append(state)

            rewards.append(total_reward)
            steps_per_episode.append(steps)

            # Check if the current episode is one of the checkpoints for policy evaluation
            if episode in policy_checkpoints:
                policy_progress.append((self.get_optimal_policy(), visited_states, episode))

            if episode % 20 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")

        return rewards, steps_per_episode, policy_progress

    def get_optimal_policy(self):
        optimal_policy = np.zeros(self.grid_size, dtype=int)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                if state == self.goal_position:
                    optimal_policy[i, j] = -1  # Indicate that the goal state has no action (or it's not applicable)
                elif state in self.obstacle_positions:
                    optimal_policy[i, j] = -1  # Indicate that an obstacle state has no action (or it's not applicable)
                else:
                    valid_actions = []
                    for action in range(self.num_actions):
                        next_state = self.get_next_state(state, self.actions[action])
                        if self.is_valid_state(next_state):
                            valid_actions.append(action)
                    if valid_actions:
                        # Choose the best action among valid_actions and store the original action index
                        q_vals = [self.Q_values[self.state_to_index(state), a] for a in valid_actions]
                        best_idx = int(np.argmax(q_vals))
                        optimal_policy[i, j] = valid_actions[best_idx]
                    else:
                        optimal_policy[i, j] = -1  # No valid action
        return optimal_policy

    def plot_rewards(self, rewards):
        plot_rewards(rewards)

    def plot_steps_per_episode(self, steps_per_episode):
        plot_steps_per_episode(steps_per_episode)

    def plot_policy_progress(self, policy_progress):
        plot_policy_progress(policy_progress, self.grid_size, self.action_symbols, self.start_position, self.goal_position, self.obstacle_positions)


if __name__ == "__main__":
    # Adjust hyperparameters for experimentation
    obstacle_positions = [(2, 3), (3, 1), (4, 3)]
    grid_world = GridWorld(grid_size=(6, 6), obstacle_positions=obstacle_positions)
    rewards, steps_per_episode, policy_progress = grid_world.q_learning(num_episodes=1000, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)

    # Plot the learning curve, steps per episode, and policy progress
    grid_world.plot_rewards(rewards)
    grid_world.plot_steps_per_episode(steps_per_episode)
    grid_world.plot_policy_progress(policy_progress)

