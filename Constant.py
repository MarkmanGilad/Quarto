import pygame

FPS = 60
HEIGHT = 800
WIDTH = 1300
FRAME = 10
CIRCLE_SIZE = 90
PADDING = 0
ROW, COL = 7, 7

# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.01
epsiln_decay = 5000

#RGB
OLCOLOR = (202,164,114) # OL = OutLine
BGCOLOR = (26,65,75)
BROWN = (205,102,29)
RED = (255,48,48)
WHITE = (255, 255, 255)
BLACK = (40,40,40)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (151,255,255)

#SIZE
SMALLSQUARESIZE = (60,60)
BIGSQUARESIZE = (73,73)

    #Pieces - Brown
    #Squares-----------------------------------------------------------------------
SBSF = pygame.image.load('Photos/SmallBrownSquareFull.jpeg') # Small Brown Square Full
SBSF = pygame.transform.scale(SBSF,(60,60))

BBSF = pygame.image.load('Photos/BigBrownSquareFull.jpeg') # Big Brown Square Full
BBSF = pygame.transform.scale(BBSF, (73,73))

SBSH = pygame.image.load('Photos/SmallBrownSquareHole.jpeg') # Small Brown Square Hole
SBSH = pygame.transform.scale(SBSH, (60,60))

BBSH = pygame.image.load('Photos/BigBrownSquareHole.jpeg') # Big Brown Square Hole
BBSH = pygame.transform.scale(BBSH, (73,73))

    #Circles-----------------------------------------------------------------------
SBCF = pygame.image.load('Photos/SmallBrownCircleFull.png') #Small Brown Circle Full
SBCF = pygame.transform.scale(SBCF, (75,75))

BBCF = pygame.image.load('Photos/BigBrownCircleFull.png') #Big Brown Circle Full
BBCF = pygame.transform.scale(BBCF, (96,96))

SBCH = pygame.image.load('Photos/SmallBrownCircleHole.png') # Small Brown Circle Hole
SBCH = pygame.transform.scale(SBCH, (75,75))

BBCH = pygame.image.load('Photos/BigBrownCircleHole.png') # Big Brown Circle Hole
BBCH = pygame.transform.scale(BBCH, (96,96))
 

    #Pieces - White
    #Squares------------------------------------------------------------------------
SWSH = pygame.image.load('Photos/SmallWhiteSquareHole.png') # Small White Square Hole
SWSH = pygame.transform.scale(SWSH,(60,60))

BWSH = pygame.image.load('Photos/BigWhiteSquareHole.png') # Big White Square Hole
BWSH = pygame.transform.scale(BWSH,(73,73))

SWSF = pygame.image.load('Photos/SmallWhiteSquareFull.jpg') # Small White Square Full
SWSF = pygame.transform.scale(SWSF, (60,60))

BWSF = pygame.image.load('Photos/BigWhiteSquareFull.jpg') # Big White Square Full
BWSF = pygame.transform.scale(BWSF, (73,73))
    
    #Circles-------------------------------------------------------------------------
SWCH = pygame.image.load('Photos/SmallWhiteCircleHole.png')
SWCH = pygame.transform.scale(SWCH, (75,75))

BWCH = pygame.image.load('Photos/BigWhiteCircleHole.png')
BWCH = pygame.transform.scale(BWCH, (96,96))

SWCF = pygame.image.load('Photos/SmallWhiteCircleFull.png')
SWCF = pygame.transform.scale(SWCF, (75,75))

BWCF = pygame.image.load('Photos/BigWhiteCircleFull.png')
BWCF = pygame.transform.scale(BWCF, (96,96))


#Side
Listside = [(SWSH,(816,220)),(BWSH,(810,310)),(SWCH, (810, 400)),(BWCH, (800, 485)),(SWSF,(916,220)),(BWSF,(910,310)),
    (SWCF, (910, 400)),(BWCF, (900, 485)),(SBSH, (1016,220)),(BBSH, (1010,310)),(SBCH, (1010, 400)),(BBCH, (1000, 485)),
    (SBSF, (1116,220)),(BBSF, (1110,310)),(SBCF, (1110, 400)),(BBCF, (1100, 485))]

#Pieces
# PieceList = {1:"SWSH", 2:"BWSH", 3:"SWCH", 4:"BWCH",5:"SWSF", 6:"BWSF",7:"SWCF",8:"BWCF",9:"SBSH",10:"BBSH",11:"SBCH",12:"BBCH",13:"SBSF",14:"BBSF",15:"SBCF",16:"BBCF"}
# piece = (color[1-W` -1-B], shape[1-Square; -1 - circle], hole[1-hole; -1-no hole], size[1-small; -1 - big])
pieceList = [(1,1,1,1), (1,1,1,-1), (1,-1,1,1), (1,-1,1,-1), (1,1,-1,1), (1,1,-1,-1), (1,-1,-1,1), (1,-1,-1,-1), (-1,1,1,1), (-1,1,1,-1), (-1,-1,1,1), (-1,-1,1,-1), (-1,1,-1,1), (-1,1,-1,-1), (-1,-1,-1,1), (-1,-1,-1,-1)]