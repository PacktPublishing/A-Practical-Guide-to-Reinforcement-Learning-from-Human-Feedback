import numpy as np
import matplotlib.pyplot as plt
from .plotting_utilities import (
    plot_rewards,
    plot_steps_per_episode,
    plot_policy_progress,
    plot_rewards_comparison,
    plot_comparison_metrics,
)
from .gridworld import GridWorld as BaseGridWorld

class GridWorldTransfer(BaseGridWorld):
    def __init__(self, grid_size, obstacle_positions):
        super().__init__(grid_size, obstacle_positions)
    
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

# Main block
if __name__ == "__main__":
    obstacle_positions = [(2, 3), (3, 1), (4, 3)]
    grid_world = GridWorldTransfer(grid_size=(6, 6), obstacle_positions=obstacle_positions)
    rewards, steps_per_episode_6x6, policy_progress_6x6 = grid_world.q_learning()
    grid_world.plot_policy_progress(policy_progress_6x6)

    # Transfer Learning
    obstacle_positions_5 = [(2, 3), (3, 1), (4, 3), (5, 2), (2, 5)]
    new_grid_world = GridWorldTransfer(grid_size=(8,8), obstacle_positions=obstacle_positions_5)
    rewards_transfer, steps_per_episode_transfer, policy_progress_transfer = new_grid_world.q_learning_transfer(grid_world.Q_values,epsilon=0.1)
    new_grid_world.plot_policy_progress(policy_progress_transfer)
    rewards_scratch, steps_per_episode, policy_progress = new_grid_world.q_learning_scratch()
    new_grid_world.plot_policy_progress(policy_progress)
    plot_rewards_comparison(rewards_transfer, rewards_scratch)
    plot_comparison_metrics(rewards_transfer, rewards_scratch, steps_per_episode_transfer, steps_per_episode)