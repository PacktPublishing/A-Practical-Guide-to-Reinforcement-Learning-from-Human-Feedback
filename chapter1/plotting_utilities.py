import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards, title='Q-learning Performance'):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.show()


def plot_steps_per_episode(steps_per_episode, title='Steps per Episode'):
    plt.plot(steps_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(title)
    plt.show()


def plot_policy_progress(policy_progress, grid_size, action_symbols, start_position, goal_position, obstacle_positions):
    if not policy_progress:
        print('No policy checkpoints to plot')
        return

    num_checkpoints = len(policy_progress)
    fig, axes = plt.subplots(1, num_checkpoints, figsize=(6 * num_checkpoints, 6))
    if num_checkpoints == 1:
        axes = [axes]

    for i, (policy, visited_states, episode) in enumerate(policy_progress):
        ax = axes[i]
        ax.set_title(f'Policy at Episode {episode}')

        for y in range(policy.shape[0]):
            for x in range(policy.shape[1]):
                state = (y, x)
                if state == goal_position:
                    ax.text(x, grid_size[0] - y - 1, 'G', va='center', ha='center', fontsize=20)
                    ax.add_patch(plt.Rectangle((x - 0.5, grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgreen'))
                elif state in obstacle_positions:
                    ax.text(x, grid_size[0] - y - 1, 'X', va='center', ha='center', fontsize=20)
                    ax.add_patch(plt.Rectangle((x - 0.5, grid_size[0] - y - 1 - 0.5), 1, 1, color='lightgray'))
                elif state == start_position:
                    ax.text(x, grid_size[0] - y - 1, 'S', va='center', ha='center', fontsize=20, color='red')
                    ax.add_patch(plt.Rectangle((x - 0.5, grid_size[0] - y - 1 - 0.5), 1, 1, color='lightblue'))
                else:
                    action = policy[y, x]
                    if 0 <= action < len(action_symbols):
                        ax.text(x, grid_size[0] - y - 1, action_symbols[action], va='center', ha='center', fontsize=20)
                    else:
                        ax.text(x, grid_size[0] - y - 1, '?', va='center', ha='center', fontsize=20)

        # Plot the path taken
        for j in range(len(visited_states) - 1):
            x1, y1 = visited_states[j][1], grid_size[0] - visited_states[j][0] - 1
            x2, y2 = visited_states[j + 1][1], grid_size[0] - visited_states[j + 1][0] - 1
            ax.plot([x1, x2], [y1, y2], color='blue', linewidth=3)

        ax.set_xticks(np.arange(-0.5, policy.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, policy.shape[0], 1))
        ax.grid(color='black', linestyle='-', linewidth=1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()


def plot_rewards_comparison(rewards_transfer, rewards_scratch, policy_epochs=None):
    # Calculate convergence speed
    def convergence_speed(rewards):
        if not rewards:
            return len(rewards)
        return next((i for i, reward in enumerate(rewards) if reward >= 0.9 * max(rewards)), len(rewards))

    convergence_speed_transfer = convergence_speed(rewards_transfer)
    convergence_speed_scratch = convergence_speed(rewards_scratch)

    # Calculate total reward
    total_reward_transfer = sum(rewards_transfer)
    total_reward_scratch = sum(rewards_scratch)

    # Smoothed curves using moving averages
    window_size = 10
    if len(rewards_transfer) >= window_size:
        rewards_transfer_smoothed = np.convolve(rewards_transfer, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size // 2, len(rewards_transfer_smoothed) + window_size // 2), rewards_transfer_smoothed,
                 label='Transfer Learning', linestyle='--', color='blue', alpha=0.7)
    else:
        plt.plot(rewards_transfer, label='Transfer Learning', linestyle='--', color='blue', alpha=0.7)

    if len(rewards_scratch) >= window_size:
        rewards_scratch_smoothed = np.convolve(rewards_scratch, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size // 2, len(rewards_scratch_smoothed) + window_size // 2), rewards_scratch_smoothed,
                 label='Training from Scratch', linestyle='--', color='orange', alpha=0.7)
    else:
        plt.plot(rewards_scratch, label='Training from Scratch', linestyle='--', color='orange', alpha=0.7)

    # Highlight transfer learning epochs if provided
    if policy_epochs is not None:
        for epoch in policy_epochs:
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


def plot_comparison_metrics(rewards_transfer, rewards_scratch, steps_transfer, steps_scratch):
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
