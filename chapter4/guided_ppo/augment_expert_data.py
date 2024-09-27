import json
import time
import numpy as np
import gymnasium as gym

def perturb_expert_actions(env, expert_data, noise_level=0.1, num_episodes=100):
    augmented_episodes = []
    
    for _ in range(num_episodes):
        # Reset the environment at the start of a new episode
        state,_ = env.reset()  # Start from the initial state of the environment
        new_episode = []
        
        # Iterate over expert data and perturb actions
        for entry in expert_data:
            # Perturb the action with noise
            perturbed_action = entry['action'] + np.random.normal(0, noise_level)
            perturbed_action = np.clip(perturbed_action, -1.0, 1.0)  # Ensure within action bounds
            
            # Step the environment with the perturbed action to get the next state
            next_state, reward, done, _ , _= env.step([perturbed_action])
            
            # Create a new entry for the new episode
            new_entry = {
                'state': state.tolist(),
                'action': float(perturbed_action),
                'next_state': next_state.tolist(),
                'reward': reward,
                'done': done
            }
            new_episode.append(new_entry)
            
            # Update state for the next iteration
            state = next_state
            
            # Break if the episode is done
            if done:
                break
        
        # Append the new episode to the list of augmented episodes
        augmented_episodes.append(new_episode)
    
    return augmented_episodes

def random_policy(obs):
    # Random action between -1 and 1
    return np.random.uniform(-1.0, 1.0)

def suboptimal_policy(obs):
    position, velocity = obs
    # Apply acceleration opposite to velocity half the time
    if np.random.uniform() < 0.5:
        return np.random.uniform(-1.0,-0.5)
    else:
        return np.random.uniform(0.5,1.0)
    
def aggressive_policy(obs):
    position, velocity = obs
    # Always accelerate in the direction of velocity to maximize speed
    return np.random.uniform(0.7,1.0)

def conservative_policy(obs):
    position, velocity = obs
    # Apply minimal acceleration
    return np.random.uniform(-0.2,0.2)

def generate_trajectories(env, num_trajectories, max_steps_per_episode):
    policies = [aggressive_policy, conservative_policy, random_policy, suboptimal_policy]
    trajectories = []
    for i in range(num_trajectories):
        obs, _ = env.reset()
        episode = []
        # Choose a policy randomly for each trajectory
        policy = np.random.choice(policies)
        for t in range(max_steps_per_episode):
            action = policy(obs)

            new_obs, reward, done, truncated, _ = env.step([action])
            obs = new_obs
        # Create a new entry for the new episode
            new_entry = {
                    'state': obs.tolist(),
                    'action': action,
                    'next_state': new_obs.tolist(),
                    'reward': reward,
                    'done': done
                }
            episode.append(new_entry)
            if done or truncated:
                break
        trajectories.append(episode)
    return trajectories


if __name__=="__main__":
        # Initialize the Mountain Car environment
    env = gym.make('MountainCarContinuous-v0')
        # Load action trajectories from JSON file
    with open('action_trajectories.json', 'r') as f:
        trajectories = json.load(f)

    # Assuming you have loaded your expert episodes into a variable called `expert_data`
    augmented_episodes = perturb_expert_actions(env, trajectories, noise_level=0.2, num_episodes=400)
        # Don't forget to close the environment after use
    # env.close()
    
    # env = gym.make('MountainCarContinuous-v0', render_mode='human')
    # non_expert_episodes = generate_trajectories(env,num_trajectories=200,max_steps_per_episode=400)
    # augmented_episodes += non_expert_episodes
    env.close()
        # Save the action trajectories to a JSON file
    with open('perturbed_expert_trajectories.json', 'w') as f:
        json.dump( augmented_episodes, f)

    print(f"Augmented trajectories saved to 'perturbed_expert_trajectories.json'.")


