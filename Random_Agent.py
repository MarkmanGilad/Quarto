import Quarto
from State import State
import random


class Random_Agent:
    def __init__(self, player:int, env:Quarto = None) -> None:
        self.player = player
        self.env : Quarto = env
        

    def getAction(self, events= None, state : State = None, train=False):
        legal_actions = self.env.legal_actions(state)
        return random.choice(legal_actions)