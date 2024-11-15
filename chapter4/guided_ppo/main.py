import yaml
import time
import gymnasium as gym
import torch
from agent import Agent
from env_setup import make_env
from utils import save_model, load_model
from training import train
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # Load config from YAML
    with open("args.yaml", "r") as f:
        args = yaml.safe_load(f)

    # Setup environment
    run_name = f"{args['env_id']}__{args['exp_name']}__{args['seed']}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(args['env_id'], i, args['capture_video'], run_name, args['gamma']) for i in range(args['num_envs'])])

    # Initialize agent
    agent = Agent(envs)

    # Start training

    writer = SummaryWriter(f"runs/{run_name}")
    train(agent, envs, writer, args)

    envs.close()
    writer.close()


