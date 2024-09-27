import gymnasium as gym
import numpy as np
import pygame
import time
import json

# Initialize Pygame
pygame.init()

# Create the environment
env = gym.make('MountainCarContinuous-v0', render_mode="human")

# Reset the environment
state,_ = env.reset()

# Define the action space
action_space = env.action_space

# Print the action and observation spaces
print("Action Space:", action_space)
print("Observation Space:", env.observation_space)
print("Initial State:", state)

# Instructions for the user
instructions = """
Press 'a' to accelerate left,
'd' to accelerate right,
's' to stop,
'q' to quit.
"""
print(instructions)

# Define the default action
action = 0.0
# To store action trajectories
trajectories = []
episode = []

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check for key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        action = action_space.low[0]*np.random.uniform(0.2,0.5)  # Limited acceleration to the left
    elif keys[pygame.K_d]:
        action = action_space.high[0]*np.random.uniform(0.2,0.5)  # Limited acceleration to the right
    elif keys[pygame.K_s]:
        action = 0.0  # No acceleration (stop)
    elif keys[pygame.K_q]:
        print("Quitting...")
        running = False  # Exit the loop if 'q' is pressed

    # Take a step in the environment
    next_state, reward, done, _,_ = env.step([action])
    print("DONE",done)
        # Store the state and action
    episode.append({
        'state': state.tolist(),
        'action': action,
        'next_state': next_state.tolist(),
        'reward': reward,
        'done': done
    })


    # If the episode is done, reset the environment
    if done:
        state,_ = env.reset()
        trajectories.append(episode.copy())
    else:
        state = next_state

    # Sleep briefly to control the speed of the loop
    time.sleep(0.05)  # Adjust this for desired responsiveness

# Save the action trajectories to a JSON file
with open('expert_trajectories.json', 'w') as f:
    json.dump(trajectories, f)

print(f"Demonstrated trajectories saved to 'expert_trajectories.json'.")

# Clean up
env.close()
pygame.quit()
