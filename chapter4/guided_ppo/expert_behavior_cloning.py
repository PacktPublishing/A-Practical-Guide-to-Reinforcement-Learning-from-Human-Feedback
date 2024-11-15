import json
import yaml
import time
import torch
import gymnasium as gym
import numpy as np
from agent import Agent
from env_setup import make_env
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from utils import save_model, load_model

def extract_expert_data(expert_trajectories):
    states = []
    actions = []
    
    for trajectory in expert_trajectories:
        for step in trajectory:
            states.append(step["state"])
            actions.append(step["action"])
    
    return states, actions

def train_behavior_cloning(agent, expert_trajectories, num_epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # Prepare data
    states, actions = zip(*expert_trajectories)  # Unzip into states and actions
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)

    # Training loop
    for epoch in range(num_epochs):
        perm = torch.randperm(states.size(0))
        for i in range(0, states.size(0), batch_size):
            indices = perm[i:i + batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]

            # Forward pass
            action_means = agent.actor_mean(batch_states)
            if agent.is_discrete:
                action_probs = nn.functional.softmax(action_means, dim=-1)
                log_probs = action_probs.log()[range(batch_actions.size(0)), batch_actions]
            else:
                action_logstd = agent.actor_logstd.expand_as(action_means)
                action_std = torch.exp(action_logstd)
                # Calculate the log probabilities for continuous actions
                dist = Normal(action_means, action_std)
                log_probs = dist.log_prob(batch_actions).sum(dim=1)
            # action, log_prob, _ ,_ = agent.get_action_and_value(batch_states,batch_actions)


            # Compute loss (negative log likelihood)
            loss = -log_probs.mean()
            # loss = nn.functional.mse_loss(action_means, batch_actions)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

def test_behavior_cloning(agent, expert_trajectories, num_episodes=5):
    agent.eval()  # Set agent to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")
    agent = agent.to(device)
    
    # Extract states and actions for testing
    states, true_actions = extract_expert_data(expert_trajectories)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    true_actions = torch.tensor(true_actions, dtype=torch.float32).to(device).unsqueeze(1)

    for episode in range(num_episodes):
        with torch.no_grad():
            action_means = agent.actor_mean(states)
            if agent.is_discrete:
                action_probs = nn.functional.softmax(action_means, dim=-1)
                predicted_actions = action_probs.argmax(dim=-1, keepdim=True)
            else:
                action_logstd = agent.actor_logstd.expand_as(action_means)
                action_std = torch.exp(action_logstd)
                dist = Normal(action_means, action_std)
                predicted_actions = dist.sample()

        # Compare predicted actions with true actions
        print(f'Episode {episode + 1}:')
        for t in range(predicted_actions.size(0)):
            print(f'Timestep {t}: Predicted Action: {predicted_actions[t].cpu().numpy()}, True Action: {true_actions[t].cpu().numpy()}')

# Inference function
def inference(env, model, device, episodes=5, render=True):
    model.eval()  # Ensure the model is in evaluation mode
    for episode in range(episodes):
        state, _ = env.reset(seed=args['seed'])
        total_reward = 0
        done = False
        while not np.all(done):
            state_tensor = torch.Tensor(state).to(device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(state_tensor)
            action = action.cpu().numpy().flatten()

            # Step the environment
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if np.any(done):
                print(done)
                print(reward)
                time.sleep(2)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


if __name__=='__main__':

    # Load config from YAML
    with open("args.yaml", "r") as f:
        args = yaml.safe_load(f)

    # Check if running inference
    run_inference = False  # Set to True to run inference, otherwise False

    # Setup environment
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"{args['env_id']}__{args['exp_name']}__{args['seed']}__{int(time.time())}"# # Save the model
    checkpoint_path = f'checkpoints/bc_checkpoint_{1}.pth'
    

    if not run_inference:
        envs = gym.vector.SyncVectorEnv([make_env(args['env_id'], i, args['capture_video'], run_name, args['gamma']) for i in range(args['num_envs'])])

        # Initialize agent
        agent = Agent(envs).to(device)

        with open('perturbed_expert_trajectories.json', 'r') as f:
            expert_episodes = json.load(f)

        states, actions = extract_expert_data(expert_episodes)

        train_behavior_cloning(agent, list(zip(states, actions)), num_epochs=50, batch_size=64,lr=1e-5)
        
        save_model(agent, checkpoint_path)
        envs.close()
    else:

        # Test on a small subset of expert trajectories
        # test_behavior_cloning(agent, expert_episodes[:5])  # Testing on the first 5 expert trajectories
        # Run inference
        env = gym.make(args['env_id'], render_mode="human")
        # envs = gym.vector.SyncVectorEnv([make_env(args['env_id'], i, args['capture_video'], run_name, args['gamma']) for i in range(args['num_envs'])])
        # Initialize agent
        agent = Agent(env).to(device)   

        # Load the model from checkpoint
        load_model(agent, checkpoint_path)
        print("Starting Inference...")
        inference(env, agent, device, episodes=20, render=True)
        env.close()

