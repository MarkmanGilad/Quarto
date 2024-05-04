import numpy as np
import pygame
from Graphics import Graphics
from Constant import *
from State import State
from Quarto import Quarto
import random
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from MinMaxAgent import MinMaxAgent
from DQN_Agent import DQN_Agent

import time

def main():
   
    #לעשות 
    run = True
    clock = pygame.time.Clock()
    graphics = Graphics()
    pygame.display.update()
    env = Quarto()
    player1 = Human_Agent(player=1,graphics=graphics, env=env) #1
    #player1 = Random_Agent(player=1, env=env)
    #player1 = DQN_Agent(env=env, player=1, parametes_path='Data/checkpoint1.pth', train=False)

    player2 = Human_Agent(player=-1,graphics=graphics, env=env) #-1
    #player2 = Random_Agent(player=-1, env=env)
    #player2 = MinMaxAgent(player= -1,depth= 3, environment=env)
    #player2 = DQN_Agent(player = -1, train=False, env=env)
    player = player1

    while(run):
        graphics.DrawBoard(env.state)
        pygame.display.update()
        
        clock.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
               break
        action = player.getAction(events=events, state=env.state)
        if action:
            env.move(state=env.state, action=action)
            graphics.DrawBoard(env.state)
            pygame.display.update()
            if env.is_end_of_game(env.state):
                if env.state.end_of_game == 1:
                    graphics.Win(state=env.state)
                    pygame.display.update()
                    env.state.setPiece(*action[0],action[1])
                    env.state = env.startState()
                    print(player.player, "Win")
                elif env.state.end_of_game == 2:
                    graphics.Draw(state=env.state)
                    pygame.display.update()
                    env.state.setPiece(*action[0],action[1])
                    env.state = env.startState()
                    print("Draw!")
                pygame.time.delay(8000)
                player = player1
                continue

            if player == player1:
                player = player2
            else:
                player = player1
               
        time.sleep(0.02)

    pygame.quit()

if __name__ == '__main__':
    main()