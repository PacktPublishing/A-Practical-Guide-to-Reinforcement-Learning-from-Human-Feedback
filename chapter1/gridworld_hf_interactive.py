import numpy as np
import matplotlib.pyplot as plt
from chapter1.plotting_utilities import (
    plot_rewards,
    plot_steps_per_episode,
    plot_policy_progress,
    plot_rewards_comparison,
    plot_comparison_metrics,
)
from chapter1.gridworld import GridWorld

class GridWorldHFInteractive(GridWorld):
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

    def q_learning_with_human_interaction(self, num_episodes=10, learning_rate=0.1, discount_factor=0.9, epsilon=0.01, policy_checkpoints=None):
        rewards = []
        steps_per_episode = []
        policy_progress = []
        if policy_checkpoints is None:
            policy_checkpoints = [1, 2]

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

                # Display current state and action for human evaluation
                self.display_state_action(state, next_state)

                # Request human evaluation
                reward_feedback = self.get_human_feedback()
                target = (reward + reward_feedback) + discount_factor * np.max(self.Q_values[self.state_to_index(next_state)])
                self.Q_values[self.state_to_index(state), action] += learning_rate * ((reward + reward_feedback) - self.Q_values[self.state_to_index(state), action])
                total_reward += reward
                state = next_state
                steps += 1
                visited_states.append(state)

            rewards.append(total_reward)
            steps_per_episode.append(steps)

            # if episode in policy_checkpoints:
            policy_progress.append((self.get_optimal_policy(), visited_states, episode))

            if episode % 20 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")

        return rewards, steps_per_episode, policy_progress

    def display_state_action(self, state, next_state):
        fig, ax = plt.subplots(figsize=(6, 6))
        print(f"Current state: {state}, Next state: {next_state}, Goal state: {self.goal_position}")
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                current_state = (y, x)
                if current_state == state:
                    ax.text(x, self.grid_size[0] - y - 1, 'A', va='center', ha='center', fontsize=20, color='red')
                elif current_state == next_state:
                    ax.text(x, self.grid_size[0] - y - 1, 'N', va='center', ha='center', fontsize=20, color='green')
                elif current_state == self.goal_position:
                    ax.text(x, self.grid_size[0] - y - 1, 'G', va='center', ha='center', fontsize=20)
                    ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgreen'))
                elif current_state in self.obstacle_positions:
                    ax.text(x, self.grid_size[0] - y - 1, 'X', va='center', ha='center', fontsize=20)
                    ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgray'))
                elif current_state == self.start_position:
                    ax.text(x, self.grid_size[0] - y - 1, 'S', va='center', ha='center', fontsize=20, color='blue')
                    ax.add_patch(plt.Rectangle((x - 0.5, self.grid_size[0] - y - 1 - 0.5), 1, 1, color='lightblue'))
                else:
                    ax.text(x, self.grid_size[0] - y - 1, '.', va='center', ha='center', fontsize=20)
        ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1))
        ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1))
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title('Current State and Next State')
        plt.show()
        plt.close()

    def get_human_feedback(self):
        print("Was the action successful? (Y/N)")
        feedback = input().strip().lower()
        if feedback == 'y':
            return 1
        elif feedback == 'n':
            return -1
        else:
            return 0

    def get_optimal_policy(self):
        optimal_policy = np.zeros(self.grid_size, dtype=int)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                if state == self.goal_position:
                    optimal_policy[i, j] = -1
                elif state in self.obstacle_positions:
                    optimal_policy[i, j] = -1
                else:
                    valid_actions = []
                    for action in range(self.num_actions):
                        next_state, _ = self.get_next_state(state, self.actions[action])
                        if self.is_valid_state(next_state):
                            valid_actions.append(action)
                    if valid_actions:
                        q_vals = [self.Q_values[self.state_to_index(state), a] for a in valid_actions]
                        best_idx = int(np.argmax(q_vals))
                        optimal_policy[i, j] = valid_actions[best_idx]
                    else:
                        optimal_policy[i, j] = -1
        return optimal_policy

if __name__ == "__main__":
    obstacle_positions = [(2, 3), (3, 1), (4, 3), (5, 2), (2, 5)]
    grid_world = GridWorldHFInteractive(grid_size=(6, 6), obstacle_positions=obstacle_positions)
    rewards, steps_per_episode, policy_progress = grid_world.q_learning_with_human_interaction(
        num_episodes=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    grid_world.plot_policy_progress(policy_progress)