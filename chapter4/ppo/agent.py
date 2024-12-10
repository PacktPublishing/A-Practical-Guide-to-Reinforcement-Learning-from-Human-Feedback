import torch
import torch.nn as nn
import numpy as np
from gymnasium import vector
from torch.distributions.normal import Normal

# Agent class with actor-critic architecture
class Agent(nn.Module):
    def __init__(self, envs):
        """
        Initializes the Agent with an actor-critic architecture.

        Args:
            envs: The environment or a vectorized environment from which 
                  the observation and action spaces are derived.

        Raises:
            ValueError: If the environment type is unsupported.
        """
        super().__init__()
        
        if isinstance(envs, vector.VectorEnv):
            # Multi-environment case
            env_shape = envs.single_observation_space.shape
            env_action_shape = envs.single_action_space.shape
        else:
            # Single environment case
            env_shape = envs.observation_space.shape
            env_action_shape = envs.action_space.shape
        
        # Determine if the action space is discrete or continuous
        if len(env_action_shape) == 1 and env_action_shape[0] > 1:
            # Discrete action space
            self.is_discrete = True
            self.num_actions = env_action_shape[0]
        elif len(env_action_shape) == 1 and env_action_shape[0] == 1:
            # Single action case; could be treated as continuous
            self.is_discrete = False
        else:
            # Assume continuous action space
            self.is_discrete = False

        # Define the critic network
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(env_shape), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Define the actor network
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(env_shape), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, np.prod(env_action_shape)), std=0.01),
        )
        
        # For continuous action space
        if not self.is_discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env_action_shape)))

    def get_value(self, x):
        """Returns the value of the input state."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Computes the action and its associated value.

        Args:
            x: The input state.
            action: An optional action to evaluate.

        Returns:
            action: The sampled or provided action.
            log_prob: Log probability of the action.
            entropy: Entropy of the action distribution.
            value: The value of the input state from the critic.
        """
        action_mean = self.actor_mean(x)
        
        if self.is_discrete:
            # For discrete action spaces, use a softmax layer
            action_probs = nn.functional.softmax(action_mean, dim=-1)
            if action is None:
                action = action_probs.multinomial(num_samples=1).squeeze(1)
            log_prob = action_probs.log()[range(action.size(0)), action]
            entropy = -(action_probs * action_probs.log()).sum(dim=-1)
        else:
            # For continuous action spaces
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)

        return action, log_prob, entropy, self.critic(x)

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """Initializes layers with specific configurations."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer