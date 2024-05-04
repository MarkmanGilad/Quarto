import numpy as np 
import torch
from State import State
from Quarto import Quarto
import Tester

env= Quarto()


env.move(env.state, ((1,1), 5))
env.move(env.state, ((1,2), 6))
env.move(env.state, ((2,3), 7))

board = env.state.board
print(board)

print(abs(board[0]).sum(axis=1)) # row
print(abs(board[1]).sum(axis=0)) # col

# value of state when my turn:
# if win = + 20
# if 
#   piece picked can end of game = + 10
# else:
#   for every triple that has a piece that can be EOG = -5
