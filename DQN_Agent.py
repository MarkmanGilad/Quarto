import torch
import random
import math
import numpy as np
from DQN import DQN
from Constant import *
from State import State
from Quarto import Quarto as env

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path and player == 1:
            self.DQN.load_params_1(parametes_path)
        elif parametes_path and player == -1:
            self.DQN.load_params_2(parametes_path)
        self.player = player
        self.train = train
        self.env= env
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def getAction (self, state:State, epoch = 0, events= None, train = True, graphics = None, black_state = None) -> tuple:
        actions = self.env.legal_actions(state)
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        state_tensor = state.toTensor()
        flattend_actions = [action[0]+ (action[1],) for action in actions]
        action_np = np.array(flattend_actions)
        action_tensor = torch.from_numpy(action_np)
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions (self, states_tensor, dones) -> torch.tensor:
        actions = []
        for i, board in enumerate(states_tensor):
            if dones[i].item():
                actions.append((0,0,0))
            else:
                state = State.tensorToState(state_tensor=board, player=self.player)
                action = self.getAction(state, train=False)
                actions.append((action[0][0], action[0][1], action[1]))
        return torch.tensor(np.array(actions))

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    # def load_params (self, path):
    #     self.DQN.load_params(path)



    def __call__(self, events= None, state=None):
        return self.getAction(state)
