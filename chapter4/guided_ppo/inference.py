import os
import random
import time
import yaml
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from agent import Agent
from utils import load_model, get_latest_checkpoint



# Inference function
def inference(env, model, device, episodes=5, render=True):
    model.eval()  # Ensure the model is in evaluation mode
    for episode in range(episodes):
        state, _ = env.reset(seed=args['seed'])
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.Tensor(state).to(device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(state_tensor)
            action = action.cpu().numpy().flatten()

            # Step the environment
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    

if __name__ == "__main__":
    # Load config from YAML
    with open("args.yaml", "r") as f:
        args = yaml.safe_load(f)

    # Check if running inference
    run_inference = True  # Set to True to run inference, otherwise False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize a vectorized environment with num_envs=1 for single inference
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, args.capture_video, "inference_run", args.gamma)])
    env = gym.make(args['env_id'], render_mode="human")

    # Initialize the agent
    agent = Agent(env).to(device)

    if run_inference:
        checkpoint_dir = 'checkpoints'
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        print(latest_checkpoint)
        # Load the final trained model
        checkpoint = load_model(agent, latest_checkpoint) # Assuming 'checkpoints/ppo_checkpoint10.pth' is the final saved model
        # Run inference
        print("Starting Inference...")
        inference(env, agent, device, episodes=5, render=True)
        env.close()