import time
import os
import torch
import json
import numpy as np
from utils import save_model, load_model, get_latest_checkpoint

with open('expert_trajectories.json', 'r') as f:
    expert_episodes = json.load(f)

def compute_advantages_and_returns(rewards, values, next_value, next_done, dones, args):
    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args['num_steps'])):
        if t == args['num_steps'] - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + args['gamma'] * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args['gamma'] * args['gae_lambda'] * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns

def train(agent, envs, writer, args):

    # Prepare for training
    batch_size = int(args['num_envs'] * args['num_steps'])
    minibatch_size = batch_size // args['num_minibatches']
    num_iterations = args['total_timesteps'] // batch_size
    

    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")
    agent = agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args['learning_rate'], eps=1e-5)
    
    obs = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    rewards = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    dones = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    values = torch.zeros((args['num_steps'], args['num_envs'])).to(device)

    # Define the checkpoints directory
    checkpoint_dir = 'checkpoints'

    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created directory: {checkpoint_dir}")
    
    if args['bc_init']:
        load_model(agent, checkpoint_dir+f'/bc_checkpoint_{1}.pth')
        print("Behavioral Cloning Initialized")
        
        global_step = 0
        start_iteration = 0
    else:
        # Load the latest checkpoint if it exists
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            print(f"Loading the latest checkpoint from: {latest_checkpoint}")
            checkpoint = load_model(agent, latest_checkpoint)

            global_step = checkpoint.get('global_step', 0)
            start_iteration = checkpoint.get('start_iteration', 0)
            print(f"Resumed training from checkpoint: {latest_checkpoint} at iteration {start_iteration}, global step {global_step}")
        else:
            global_step = 0
            start_iteration = 0
            print("No checkpoint found. Starting from scratch.")
        

    next_obs, _ = envs.reset(seed=args['seed'])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args['num_envs']).to(device)

    for iteration in range(start_iteration, num_iterations + 1):
        if args['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args['learning_rate']

        for step in range(args['num_steps']):
            global_step += args['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
                

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)


            # Logging
            for info in infos.get("final_info", []): #infos['final_info']:
                if info and 'episode' in info:
                    print(f"Global step: {global_step}, episodic return: {info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
                    writer.add_scalar("charts/episodic_length", info['episode']['l'], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_advantages_and_returns(rewards, values, next_value, next_done, dones, args)

        # Flatten the data for training
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = None
        kl_exceed_count = 0

        for epoch in range(args['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args['clip_coef'], 1 + args['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args['ent_coef'] * entropy_loss + v_loss * args['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args['max_grad_norm'])
                optimizer.step()

                if args['target_kl'] is not None:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args['clip_coef']).float().mean().item())
                    if approx_kl > args['target_kl']:
                        kl_exceed_count += 1
                        print(f"KL Divergence exceeded in iteration {iteration}, epoch {epoch}")
                        break  # Break out of the loop if KL exceeds
            if kl_exceed_count > 3:  # If KL exceeds for 3 epochs, break early
                print(f"Stopping early due to KL divergence exceeding threshold for {kl_exceed_count} epochs.")
                break

                
        # Continue with PPO updates, logging, and model saving
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging losses
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if approx_kl is not None:
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # Save model periodically
        if (iteration % 2 == 0 and iteration > num_iterations / 20) or iteration == num_iterations:
            save_model(agent, f'checkpoints/ppo_checkpoint_{iteration}.pth', global_step, iteration)


