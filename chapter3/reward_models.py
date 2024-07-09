import torch
import torch.nn as nn
import torch.optim as optim

# RewardModel class definition
class RewardModel(nn.Module):
    def __init__(self, state_size=2, hidden_size=64, output_size=3):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_size * 2, hidden_size)  # Adjusted to take both state and next_state as input
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output 3 rewards: progress,  safety, and goal-efficiency,

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)  # Concatenate state and next_state
        x = torch.relu(self.fc1(x))
        rewards = self.fc2(x)
        return rewards