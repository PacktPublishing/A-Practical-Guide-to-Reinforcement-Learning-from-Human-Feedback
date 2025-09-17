import numpy as np
import matplotlib.pyplot as plt
from .plotting_utilities import (
    plot_rewards,
    plot_steps_per_episode,
    plot_policy_progress,
    plot_rewards_comparison,
    plot_comparison_metrics,
)
from .gridworld import GridWorld

class GridWorldHF(GridWorld):
    def q_learning_with_human_feedback(self, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, policy_checkpoints=None, human_feedback=None):
        rewards = []
        steps_per_episode = []
        policy_progress = []  # Store optimal policy at different checkpoints
        if policy_checkpoints is None:
            policy_checkpoints = [num_episodes // 100, num_episodes // 10, num_episodes - 1]

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

                # Check for human feedback and adjust reward accordingly
                if human_feedback is not None:
                    if state in human_feedback and next_state in human_feedback[state]:
                        reward += human_feedback[state][next_state]

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

if __name__ == "__main__":
    # Adjust hyperparameters for experimentation
    obstacle_positions = [(2, 3), (3, 1), (4, 3)]
    grid_world = GridWorldHF(grid_size=(6, 6), obstacle_positions=obstacle_positions)
    rewards, steps_per_episode, policy_progress = grid_world.q_learning(num_episodes=200, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)

    # Plot the learning curve, steps per episode, and policy progress
    grid_world.plot_policy_progress(policy_progress)

    human_feedback = {
        (2, 2): {(2, 3): 1},  # Positive feedback for transitioning from (2, 2) to (2, 3)
        (2, 3): {(2, 4): 1},  # Positive feedback for transitioning from (2, 3) to (2, 4)
        (2, 4): {(2, 5): 1},  # Positive feedback for transitioning from (2, 4) to (2, 5)
        (3, 4): {(2, 4): -1},  # Negative feedback for transitioning from (3, 4) to (2, 4)
        (4, 4): {(3, 4): -1},  # Negative feedback for transitioning from (4, 4) to (3, 4)
        (4, 5): {(5, 5): 1},  # Positive feedback for transitioning from (4, 5) to (5, 5)
    }

    rewards_hf, steps_per_episode_hf, policy_progress_hf = grid_world.q_learning_with_human_feedback(num_episodes=200, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, human_feedback=human_feedback)

    grid_world.plot_policy_progress(policy_progress_hf)

    # Plot rewards comparison and other metrics
    plot_rewards_comparison(rewards_hf, rewards)
    plot_comparison_metrics(rewards_hf, rewards, steps_per_episode_hf, steps_per_episode)