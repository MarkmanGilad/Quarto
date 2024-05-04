import numpy as np
from Constant import *
import torch

class State:
    def __init__(self, board, Listbool=None, Piece=None, player = 1, endofgame = 0):
        self.board = board # shape [4,4,4] = [surface, row, col]
        self.ListBool = Listbool
        self.Piece = Piece
        self.player = player
        self.end_of_game = endofgame
        
    def converTo2DBoard (self):
        board2d = np.zeros([4,4])
        for row in range(4):
            for col in range(4):
                board2d[row, col] = self.getPiece(row,col)
        return board2d

    def getPiece (self, row, col):
        piece = tuple(self.board[:,row,col])
        if piece == (0,0,0,0):
            return 0
        return pieceList.index(piece) +1 

    def setPiece(self, row, col, piece_num):
        piece = pieceList[piece_num-1] # (1,1,1,1)
        self.board[:,row,col] = piece

    def __eq__(self, other) ->bool:
        b1 = np.equal(self.board, other.board).all()
        return np.equal(self.board, other.board).all()

    def __hash__(self) -> int:
        return hash(repr(self.board) + repr(self.player))
    
    def copy (self):
        newBoard = np.copy(self.board)
        newList = self.ListBool.copy()
        return State(board=newBoard, Listbool=newList, Piece=self.Piece, player=self.player)
    
    def toTensor(self, device = torch.device('cpu')) -> tuple:
        board_np = self.board.reshape(-1, 64)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        piece_tensor = torch.tensor(self.Piece, dtype=torch.float32, device=device).reshape(-1, 1)
        state_tensor = torch.cat((board_tensor,piece_tensor), dim=1)
        return state_tensor.reshape(-1)
    
    @staticmethod
    def tensorToState(state_tensor, player):
        board= state_tensor[0:64].reshape([4,4,4]).cpu().numpy()
        Piece = state_tensor[64].cpu().numpy()
        state = State(board, player=player, Piece=Piece) 
        state.ListBool= state.getListbool()
        return state
    
    def getListbool(self):
        Listbool = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
        for row in range(4):
            for col in range(4):
                if self.board[0,row,col] != 0:
                    Listbool[self.getPiece(row,col)-1] = False
        return Listbool        