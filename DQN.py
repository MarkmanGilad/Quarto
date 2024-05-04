import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 68 # state: board = 4 * 4 * 4 = 64 + action (3) 
layer1 = 64
layer2= 128
layer3= 256
layer4= 512
layer5= 256
layer6= 128
layer7= 64
output_size = 1 # Q(state, action)
gamma = 0.99 


class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        # self.linear2= nn.Linear(layer1, layer2)
        # self.linear3= nn.Linear(layer2, layer3)
        # self.linear4= nn.Linear(layer3, layer4)
        # self.linear5= nn.Linear(layer4, layer5)
        # self.linear6= nn.Linear(layer5, layer6)
        # self.linear7= nn.Linear(layer6, layer7)
        self.output = nn.Linear(layer1, output_size)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        # x = self.linear2(x)
        # x = F.leaky_relu(x)
        # x = self.linear3(x)
        # x = F.leaky_relu(x)
        # x = self.linear4(x)
        # x = F.leaky_relu(x)
        # x = self.linear5(x)
        # x = F.leaky_relu(x)
        # x = self.linear6(x)
        # x = F.leaky_relu(x)
        # x = self.linear7(x)
        # x = F.leaky_relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return self.MSELoss(Q_value, Q_new)
    
    def load_params_1(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict_1'])

    def load_params_2(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict_2'])

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)