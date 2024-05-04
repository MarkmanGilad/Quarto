import numpy as np
import pygame
from pyparsing import col
import time
from State import State
from Constant import *


pygame.init()
class Graphics():

    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Quarto')

    def DrawBoard(self, state:State):
        self.FillScreen()
        self.draw_all_Circles(state)
        self.drawSide(state)
        pygame.draw.circle(self.win, OLCOLOR, [375,370],350, 5)
        TextFont = pygame.font.SysFont("Segoe Print", 70)
        TextFontP= pygame.font.SysFont("Segoe Print", 45)
        TextHeader = TextFont.render("Quarto", 1, (26, 26, 26))
        player = 1
        if state.player == 1:
            player =1
        else:
            player = 2
        TextPlayer = TextFontP.render("Player " + str(player) + " Turn", 1, (24,24,24))
        self.win.blit(TextHeader, (900,40))
        self.win.blit(TextPlayer, (875, 125))

    def Win(self, state:State):
        pygame.draw.rect(self.win, BGCOLOR, 
                 pygame.Rect(875, 140, 375, 70))
        TextFontP= pygame.font.SysFont("Segoe Print", 45)
        player = 1
        if state.player == 1:
            player =2
        else:
            player = 1
        TextPlayer = TextFontP.render("Player " + str(player) + " Win!", 1, (24,24,24))
        self.win.blit(TextPlayer, (875, 125))

    def Draw(self, state:State):
        pygame.draw.rect(self.win, BGCOLOR, 
                 pygame.Rect(875, 140, 375, 70))
        TextFontP= pygame.font.SysFont("Segoe Print", 45)
        TextPlayer = TextFontP.render("Draw!", 1, (24,24,24))
        self.win.blit(TextPlayer, (925, 125))

    def drawSide(self, state:State):
        for i in range(16):
            if state.ListBool[i] == True:
                self.win.blit(Listside[i][0], (Listside[i][1]))
        if state.Piece:
            posx, posy = Listside[state.Piece-1][1]
            posx += 25
            posy += 25
            if state.Piece == 1 or state.Piece == 5 or state.Piece ==9 or state.Piece == 13:
                pygame.draw.circle(self.win, RED, (posx+5, posy+5),48, 5)
            elif state.Piece == 2 or state.Piece == 6 or state.Piece ==10 or state.Piece == 14:
                pygame.draw.circle(self.win, RED, (posx+10, posy+10),55, 5)
            elif state.Piece==3 or state.Piece == 7 or state.Piece == 11 or state.Piece == 15:
                pygame.draw.circle(self.win, RED, (posx+12, posy+12),45, 5)
            else:
                pygame.draw.circle(self.win, RED, (posx+23, posy+23),55, 5)
    def FillScreen(self):
        self.win.fill(BGCOLOR)
    
    def draw_all_Circles(self, state:State):
        
        convert_dict = {
            (0,3):(0,0),
            (1,2):(1,0),
            (1,4):(0,1),
            (2,1):(2,0),
            (2,3):(1,1),
            (2,5):(0,2),
            (3,0):(3,0),
            (3,2):(2,1),
            (3,4):(1,2),
            (3,6):(0,3),
            (4,1):(3,1),
            (4,3):(2,2),
            (4,5):(1,3),
            (5,2):(3,2),
            (5,4):(2,3),
            (6,3):(3,3)
        }
        for row in range(ROW):
            for col in range(COL):
                self.draw_Circle(state, (row, col), convert_dict)

    def draw_Circle(self, state:State, ROWandCOL, convertdict):
        board = state.converTo2DBoard()
        Position = self.get_Position(ROWandCOL)
        Color = BGCOLOR

        if(ROWandCOL in convertdict):
            Color = OLCOLOR
            pygame.draw.circle(self.win, Color, [Position[0]+50,Position[1]+50],60, 5)
            dictrow, dictcol = convertdict[ROWandCOL]
            if(board[dictrow, dictcol] == 1):
                self.win.blit(SWSH,[Position[0]+18,Position[1]+20])
            elif(board[dictrow, dictcol] == 2):
                self.win.blit(BWSH,[Position[0]+13,Position[1]+13])
            elif(board[dictrow, dictcol] == 3):
                self.win.blit(SWCH,[Position[0]+11,Position[1]+13])
            elif(board[dictrow, dictcol] == 4):
                self.win.blit(BWCH,[Position[0]+1,Position[1]+1])
            elif(board[dictrow, dictcol] == 5):
                self.win.blit(SWSF,[Position[0]+20,Position[1]+20])
            elif(board[dictrow, dictcol] == 6):
                self.win.blit(BWSF,[Position[0]+13,Position[1]+13])       
            elif(board[dictrow, dictcol] == 7):
                self.win.blit(SWCF,[Position[0]+11,Position[1]+13])
            elif(board[dictrow, dictcol] == 8):
                self.win.blit(BWCF,[Position[0]+1,Position[1]+1])
            elif(board[dictrow, dictcol] == 9):
                self.win.blit(SBSH,[Position[0]+20,Position[1]+20])
            elif(board[dictrow, dictcol] == 10):
                self.win.blit(BBSH,[Position[0]+13,Position[1]+13])
            elif(board[dictrow, dictcol] == 11):
                self.win.blit(SBCH,[Position[0]+13,Position[1]+13])
            elif(board[dictrow, dictcol] == 12):
                self.win.blit(BBCH,[Position[0]+2,Position[1]+3])
            elif(board[dictrow, dictcol] == 13):
                self.win.blit(SBSF,[Position[0]+20,Position[1]+20])
            elif(board[dictrow, dictcol] == 14):
                self.win.blit(BBSF,[Position[0]+13,Position[1]+13])
            elif(board[dictrow, dictcol] == 15):
                self.win.blit(SBCF,[Position[0]+13,Position[1]+13])
            elif(board[dictrow, dictcol] == 16):
                self.win.blit(BBCF,[Position[0]+2,Position[1]+3])
   
    def get_Position(self, ROWandCOL):
        row, col = ROWandCOL
        X = row * CIRCLE_SIZE + CIRCLE_SIZE//2 + FRAME
        Y = col * CIRCLE_SIZE + CIRCLE_SIZE//2 + FRAME
        return X,Y

    def get_row_Col(self, pos): 
        x, y = pos
        if(x > 60 and x < 145 and y > 325 and y < 420 ):
            #(0,0)
            return (0,0)
        if(x > 145 and x <245 and y >410 and y <510):
            #(0,1)
            return (0,1)
        if(x >230 and x < 335 and y > 500 and y< 600):
            #(0,2)
            return (0,2)
        if(x >320 and x < 410 and y > 590 and y< 690):
            #(0,3)
            return (0,3)
        if(x >145 and x < 245 and y > 230 and y< 330):
            #(1,0)
            return (1,0)
        if(x >230 and x < 335 and y > 320 and y< 420):
            #(1,1)
            return (1,1)
        if(x >320 and x < 420 and y > 410 and y< 510):
            #(1,2)
            return (1,2)
        if(x >410 and x < 515 and y > 500 and y< 600):
            #(1,3)
            return (1,3)
        if(x >235 and x < 335 and y > 140 and y< 235):
            #(2,0)
            return (2,0)
        if(x >325 and x < 420 and y > 235 and y< 330):
            #(2,1)
            return (2,1)
        if(x >410 and x < 515 and y > 320 and y< 420):
            #(2,2)
            return (2,2)
        if(x >500 and x < 600 and y > 410 and y< 510):
            #(2,3)
            return (2,3)
        if(x >320 and x < 420 and y > 50 and y< 150):
            #(3,0)
            return (3,0)
        if(x >410 and x < 510 and y > 135 and y< 250):
            #(3,1)
            return (3,1)
        if(x >500 and x < 600 and y > 230 and y< 330):
            #(3,2)
            return (3,2)
        if(x >590 and x < 700 and y > 320 and y< 420):
            #(3,3)
            return (3,3)
        return None

    def get_piece(self, pos):
        x, y = pos
        if(x > 810 and x < 880 and y > 215 and y < 280 ):
            #SWSH 1
            return 1
        if(x > 810 and x <880 and y >305 and y <380):
            #BWSH 2
            return 2
        if(x >810 and x < 880 and y > 330 and y< 480):
            #(0,2)
            return 3
        if(x >810 and x < 880 and y > 355 and y< 580):
            #(0,3)
            return 4
        if(x >905 and x < 990 and y > 215 and y< 280):
            #(1,0)
            return 5
        if(x >905 and x < 990 and y > 305 and y< 380):
            #(1,1)
            return 6
        if(x >905 and x < 990 and y > 330 and y< 480):
            #(1,2)
            return 7
        if(x >905 and x < 990 and y > 355 and y< 580):
            #(1,3)
            return 8
        if(x >1005 and x < 1090 and y > 215 and y< 280):
            #(2,0)
            return 9
        if(x >1005 and x < 1090 and y > 305 and y< 380):
            #(2,1)
            return 10
        if(x >1005 and x < 1090 and y > 330 and y< 480):
            #(2,2)
            return 11
        if(x >1005 and x < 1090 and y > 355 and y< 580):
            #(2,3)
            return 12
        if(x >1105 and x < 1190 and y > 215 and y< 280):
            #(3,0)
            return 13
        if(x >1105 and x < 1190 and y > 305 and y< 380):
            #(3,1)
            return 14
        if(x >1105 and x < 1190 and y > 330 and y< 480):
            #(3,2)
            return 15
        if(x >1105 and x < 1190 and y > 355 and y< 580):
            #(3,3)
            return 16
        return None


