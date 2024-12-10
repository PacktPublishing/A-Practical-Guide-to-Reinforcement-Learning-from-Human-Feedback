import numpy as np
import matplotlib.pyplot as plt

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
            # Define rewards
            reward = 0
            if next_state == self.goal_position:
                reward = 10  # Positive reward for reaching the goal
            elif next_state in self.obstacle_positions:
                reward = -5  # Negative reward for hitting obstacles
            else:
                reward = -1  # Negative reward for each step

            return next_state, reward
        return state, -1  # Penalty for attempting an invalid action

    def q_learning(self, num_episodes=200, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, policy_checkpoints=None):
        rewards = []
        steps_per_episode = []
        policy_progress = []  # Store optimal policy at different checkpoints
        if policy_checkpoints is None:
            policy_checkpoints = [num_episodes // 300, num_episodes // 30, num_episodes // 10, num_episodes//2]

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

                next_state, reward = self.get_next_state(state, self.actions[action])

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

            if episode % 5 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")

        return rewards, steps_per_episode, policy_progress

    # def q_learning_transfer(self, pre_learned_Q_values):
    #     # Initialize Q-values with pre-learned values
    #     self.Q_values = pre_learned_Q_values

    #     # Perform Q-learning in the new environment
    #     rewards_transfer, steps_per_episode, policy_progress = self.q_learning()
    #     return rewards_transfer, steps_per_episode, policy_progress
    
    def q_learning_transfer(self, pre_learned_Q_values,epsilon=0.01):
        # Initialize Q-values with pre-learned values
        self.Q_values[:pre_learned_Q_values.shape[0], :pre_learned_Q_values.shape[1]] = pre_learned_Q_values
        print(f"Q learning with Transferred Q-values")

        # Perform Q-learning in the new environment
        rewards_transfer, steps_per_episode, policy_progress = self.q_learning(epsilon=0.01)
        return rewards_transfer, steps_per_episode, policy_progress

    def q_learning_scratch(self):
        # Perform Q-learning from scratch in the new environment
        #  Reset Q-values to zeros (or some other initial values)
        self.Q_values = np.zeros((self.num_states, self.num_actions))  # Or initialize with other initial values if needed
        print(f"Q learning from Scratch")

        rewards_scratch, steps_per_episode, policy_progress = self.q_learning()
        return rewards_scratch, steps_per_episode, policy_progress

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
                        next_state, _ = self.get_next_state(state, self.actions[action])
                        if self.is_valid_state(next_state):
                            valid_actions.append(action)
                    if valid_actions:
                        optimal_policy[i, j] = np.argmax([self.Q_values[self.state_to_index(state), a] for a in valid_actions])
                    else:
                        optimal_policy[i, j] = -1  # No valid action
        return optimal_policy

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Q-learning Performance')
        plt.show()

    def plot_steps_per_episode(self, steps_per_episode):
        plt.plot(steps_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')
        plt.show()

    def plot_policy_progress(self, policy_progress):
        num_checkpoints = len(policy_progress)
        fig, axes = plt.subplots(1, num_checkpoints, figsize=(6*num_checkpoints, 6))

        for i, (policy, visited_states, episode) in enumerate(policy_progress):
            ax = axes[i]
            ax.set_title(f'Policy at Episode {episode}')

            for y in range(policy.shape[0]):
                for x in range(policy.shape[1]):
                    state = (y, x)
                    if state == self.goal_position:
                        ax.text(x, self.grid_size[0] - y - 1, 'G', va='center', ha='center', fontsize=20)
                        ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgreen'))
                    elif state in self.obstacle_positions:
                        ax.text(x, self.grid_size[0] - y - 1, 'X', va='center', ha='center', fontsize=20)
                        ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgray'))
                    elif state == self.start_position:
                        ax.text(x, self.grid_size[0] - y - 1, 'S', va='center', ha='center', fontsize=20, color='red')
                        ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightblue'))
                    else:
                        action = policy[y, x]
                        ax.text(x, self.grid_size[0] - y - 1, self.action_symbols[action], va='center', ha='center', fontsize=20)

            # Plot the path taken
            for j in range(len(visited_states) - 1):
                x1, y1 = visited_states[j][1], self.grid_size[0] - visited_states[j][0] - 1
                x2, y2 = visited_states[j + 1][1], self.grid_size[0] - visited_states[j + 1][0] - 1
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=3)

            ax.set_xticks(np.arange(-0.5, policy.shape[1], 1))
            ax.set_yticks(np.arange(-0.5, policy.shape[0], 1))
            ax.grid(color='black', linestyle='-', linewidth=1)

            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    def plot_rewards_comparison(self, rewards_transfer, rewards_scratch):
        # Calculate convergence speed
        convergence_speed_transfer = next((i for i, reward in enumerate(rewards_transfer) if reward >= 0.9 * max(rewards_transfer)), len(rewards_transfer))
        convergence_speed_scratch = next((i for i, reward in enumerate(rewards_scratch) if reward >= 0.9 * max(rewards_scratch)), len(rewards_scratch))

        # Calculate total reward
        total_reward_transfer = sum(rewards_transfer)
        total_reward_scratch = sum(rewards_scratch)

        # Smoothed curves using moving averages
        window_size = 10
        rewards_transfer_smoothed = np.convolve(rewards_transfer, np.ones(window_size)/window_size, mode='valid')
        rewards_scratch_smoothed = np.convolve(rewards_scratch, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size//2, len(rewards_transfer_smoothed) + window_size//2), rewards_transfer_smoothed, label='Transfer Learning', linestyle='--', color='blue', alpha=0.7)
        plt.plot(np.arange(window_size//2, len(rewards_scratch_smoothed) + window_size//2), rewards_scratch_smoothed, label='Training from Scratch', linestyle='--', color='orange', alpha=0.7)
        
        # Highlight transfer learning epochs
        transfer_epochs = [episode for _, _, episode in policy_progress]
        for epoch in transfer_epochs:
            plt.axvline(x=epoch, color='gray', linestyle='--', linewidth=0.8)

        # Add convergence speed and total reward to plot
        plt.text(0.5, 0.6, f"Convergence Speed (Transfer): {convergence_speed_transfer}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.55, f"Total Reward (Transfer): {total_reward_transfer}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.5, f"Convergence Speed (Scratch): {convergence_speed_scratch}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.45, f"Total Reward (Scratch): {total_reward_scratch}", transform=plt.gca().transAxes)

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Q-learning Performance Comparison')
        plt.legend()
        plt.show()

    def plot_comparison_metrics(self, rewards_transfer, rewards_scratch, steps_transfer, steps_scratch):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot total reward comparison
        ax1 = axes[0]
        ax1.plot(rewards_transfer, label='Transfer Learning')
        ax1.plot(rewards_scratch, label='Training from Scratch')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Total Reward Comparison')
        ax1.legend()

        # Plot steps per episode comparison
        ax2 = axes[1]
        ax2.plot(steps_transfer, label='Transfer Learning')
        ax2.plot(steps_scratch, label='Training from Scratch')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps per Episode')
        ax2.set_title('Steps per Episode Comparison')
        ax2.legend()

        plt.tight_layout()
        plt.show()




## Transfer Learning
# # Pre-learned policy and Q-values from a 6x6 grid world
# pre_learned_policy_6x6 = np.array([[1, 2, 1, 1, 1, 2],
#                                    [0, 2, 0, 0, 0, 2],
#                                    [0, 2, -1, -1, -1, 2],
#                                    [0, 2, 0, 0, 0, 2],
#                                    [0, 0, 0, 0, 0, 0],
#                                    [0, 1, 0, 2, 2, 0]])

# pre_learned_Q_values_6x6 = np.random.rand(36, 4)  # Random Q-values for demonstration
obstacle_positions = [(2, 3), (3, 1), (4, 3)]
grid_world = GridWorld(grid_size=(6, 6), obstacle_positions=obstacle_positions)
rewards, steps_per_episode_6x6, policy_progress_6x6 = grid_world.q_learning()
grid_world.plot_policy_progress(policy_progress_6x6)


# Transfer Learning

# Create the new grid world environment (8x8) with different obstacle positions
obstacle_positions_5 = [(2, 3), (3, 1), (4, 3), (5, 2), (2, 5)]

new_grid_world = GridWorld(grid_size=(8,8), obstacle_positions=obstacle_positions_5)

# Assign the resized Q-values array to the new grid world
rewards_transfer, steps_per_episode_transfer, policy_progress_transfer = new_grid_world.q_learning_transfer(grid_world.Q_values,epsilon=0.1)
new_grid_world.plot_policy_progress(policy_progress_transfer)
rewards_scratch, steps_per_episode, policy_progress = new_grid_world.q_learning_scratch()
new_grid_world.plot_policy_progress(policy_progress)

# Plot rewards comparison and other metrics
grid_world.plot_rewards_comparison(rewards_transfer, rewards_scratch)
grid_world.plot_comparison_metrics(rewards_transfer, rewards_scratch,  steps_per_episode_transfer,steps_per_episode)