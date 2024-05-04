from Quarto import Quarto
from State import State
MAXSCORE = 1000

class MinMaxAgent:

    def __init__(self, player, depth, environment: Quarto = None):
        self.player = player
       # self.opponent = player * -1
        self.depth = depth
        self.environment : Quarto = environment

    

    def evaluate (self, gameState : State):
        score = 0
        
        if self.environment.is_end_of_game(gameState):
            if gameState.end_of_game == 1 and self.player == gameState.player:
                score+=100
            elif gameState.end_of_game == 1 and self.player != gameState.player:
                score+=-100
            elif gameState.end_of_game == 2:
                score += 0
        elif self.environment.is_end_of_game(gameState) == False:
            score+=0
        
        return score

    def getAction(self, events, state: State):
        value, bestAction = self.minMax(state)
        return bestAction

    def minMax(self, state:State):
        visited = set()
        depth = 0
        return self.max_value(state, visited, depth)
        
    def max_value (self, state:State, visited:set, depth):
        
        value = -MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(state, action)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.min_value(newState, visited,  depth + 1)
                if newValue > value:
                    value = newValue
                    bestAction = action

        return value, bestAction 

    def min_value (self, state:State, visited:set, depth):
        
        value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(state, action)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.max_value(newState, visited,  depth + 1)
                if newValue < value:
                    value = newValue
                    bestAction = action

        return value, bestAction 

