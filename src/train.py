from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn
import torch.nn.functional as F

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeedQNetwork_Plus(nn.Module):
    def __init__(self, env, thresholds):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        width = 128

        # Calculate the number of additional features
        self.num_binary_vars = state_dim  # One binary variable per feature
        self.num_multiplications = int(state_dim * (state_dim - 1) / 2)  # Combination of pairs

        # Updated input dimension to account for the new features
        updated_input_dim = state_dim + self.num_binary_vars + self.num_multiplications

        self.bn1 = nn.BatchNorm1d(updated_input_dim)
        self.fc1 = nn.Linear(updated_input_dim, width)
        
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.fc3 = nn.Linear(width, action_dim)

        # Thresholds for creating binary variables
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)

    def forward(self, x):
        # Create binary variables
        binary_vars = (x < self.thresholds).float()

        # Calculate feature multiplications
        multiplications = []
        for i in range(x.size(1) - 1):
            for j in range(i + 1, x.size(1)):
                multiplications.append(x[:, i] * x[:, j])
        multiplications = torch.stack(multiplications, dim=1)

        # Concatenate original features, binary variables, and multiplications
        #print(x.shape,binary_vars.shape,multiplications.shape)
        x = torch.cat([x, binary_vars, multiplications], dim=1)

        # Forward pass
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        
        x = self.fc3(x)
        return x

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        network.eval()
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
class ProjectAgent:
    def __init__(self):
        device = "cpu"
        tresholds = [800000,500,1100,30,1000,20000]
        self.network = DeedQNetwork_Plus(env,tresholds).to(device)

    def act(self, observation, use_random=False):
        return greedy_action(self.network, observation)  # Greedy action based on the network

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self):
        path = "./model_dir/dqn_best.pth"
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
