from Graphics import Graphics
import pygame
import time
import Quarto
from State import State
#from Constant import *

class Human_Agent:
    def __init__(self, player:int, graphics:Graphics=None, env:Quarto = None) -> None:
        self.player = player
        self.mode = 1 # 1 pick row_col; 2 - pick - piece
        self.row_col = None
        self.original_state = None
        self.graphics = graphics
        self.env = env
        

    def getAction(self, events= None, state : State = None):

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if self.mode == 1: # pick row_col
                    row_col = self.graphics.get_row_Col(pos)
                    if row_col == None:
                        return None
                    if state.getPiece(*row_col) == 0: # row_col legal
                        self.row_col = row_col
                        self.original_state = state.copy()
                        state.setPiece(*row_col,state.Piece)
                        state.ListBool[state.Piece-1] = False
                        state.Piece = None
                        self.mode = 2
                        if self.env.is_end_of_game(state):
                            self.mode = 1
                            self.env.state = self.original_state
                            return self.row_col, 1

                        return None
                    else:
                        return None
                if self.mode == 2: # pick piece
                    piece = self.graphics.get_piece(pos)
                    if piece == None:
                        return None
                    if state.ListBool[piece-1]:  # piece legal
                        self.mode = 1
                        self.env.state = self.original_state
                        return self.row_col, piece
                    else:
                        return None
                
                return None
        return None

