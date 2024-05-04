from State import State
from Constant import *
import numpy as np
import random
import torch

class Quarto:
    
    def __init__(self, state:State = None) -> None:
        if (state):
            self.state=state
        else:
            self.state = self.startState()
            self.Winner = 0
    
    def startState(self):
        board = np.zeros([4,4,4])
        Piece =random.randint(1,16)
        Listbool = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True] 
        return State(board, Listbool, Piece, 1)

    def move (self, state : State, action):
        row_col, piece = action
        state.setPiece(*row_col,state.Piece)
        state.ListBool[state.Piece-1] = False
        state.Piece = piece
        state.player = state.player * -1

    def get_next_state (self, state, action):
        next_state = state.copy()
        self.move(next_state, action)
        return next_state

    def getListbool(self, state):
        Listbool = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
        for row in range(4):
            for col in range(4):
                if state.board[0,row,col] != 0:
                    Listbool[state.getPiece(row,col)-1] = False
        return Listbool        

    def legal_actions (self, state):
        empty =np.where(state.board[0] == 0)
        empty_row_col = list(zip(empty[0],empty[1]))
        lst= []
        for rowcol in empty_row_col:
            for i, piece in enumerate(state.ListBool):
                if piece and i != state.Piece-1:
                    lst.append((rowcol, i+1))
        if lst:
            return lst
        else: # full board
            return [(empty_row_col[0],-1)]

    def is_end_of_game(self, state: State): #USE ACTION (ACTION)
        colorboard = state.board[0]
        shapeboard = state.board[1]
        #hollowboard= state.board[2]
        sizeboard = state.board[3]
        row_sumcolor = abs(np.sum(colorboard, axis=1))
        #row_sumhollow = abs(np.sum(hollowboard, axis=1))
        row_sumsize = abs(np.sum(sizeboard, axis=1)) 
        row_sumshape = abs(np.sum(shapeboard, axis=1))
        #col_sumhollow = abs(np.sum(hollowboard, axis=0))
        col_sumsize = abs(np.sum(sizeboard, axis=0))  
        col_sumcolor = abs(np.sum(colorboard, axis=0))
        col_sumshape = abs(np.sum(shapeboard, axis=0))
        diagonals = [abs(np.trace(colorboard)), abs(np.trace(np.fliplr(colorboard))),abs(np.trace(shapeboard)), abs(np.trace(np.fliplr(shapeboard))), abs(np.trace(sizeboard)), abs(np.trace(np.fliplr(sizeboard)))]
        if 4 in row_sumcolor or 4 in row_sumshape or 4 in col_sumcolor or 4 in col_sumshape or 4 in row_sumsize or 4 in col_sumsize or 4 in diagonals:
            state.end_of_game = 1
            self.Winner = state.player * -1 # 1, -1
            return True
        if 0 not in colorboard:
            state.end_of_game = 2
            self.Winner = 0 # draw
            return True
        return False
    
    @staticmethod
    def actionToTensor(action):
        return torch.from_numpy(np.array((action[0][0], action[0][1], action[1])))
    
    def reward (self, state : State, action = None) -> tuple:
        if (self.is_end_of_game(state)):
            return self.Winner, True
        return 0, False