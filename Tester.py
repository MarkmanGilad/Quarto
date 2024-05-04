from Random_Agent import Random_Agent
from Quarto import Quarto as env
from DQN_Agent import DQN_Agent
import torch

class Tester:
    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        draw = 0
        games = 0

        while games < games_num:
            print(games, end= "\r" )
            action = player.getAction(state=env.state, train=False)
            env.move(action=action, state=env.state)
            player = self.switchPlayers(player)
           # print("player", player.player)
            if env.is_end_of_game(env.state): 
                if env.state.end_of_game == 1:
                    playerlost = player.player
                    if playerlost == -1:
                        player1_win += 1
                    elif playerlost ==1:
                        player2_win += 1
                if env.state.end_of_game == 2:
                    draw+=1
                env.state = env.startState()
                games += 1
                player = self.player1
        return player1_win, player2_win, draw     

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    env = env()
    checkpoint = torch.load('Data/checkpoint8.pth')
    player1 = DQN_Agent(env=env, player=1, parametes_path=None, train=False)
    #player2 = DQN_Agent(env=env, player=-1, parametes_path=None, train=False)
    player1.DQN.load_state_dict(checkpoint['model_state_dict_1'])
    #player2.DQN.load_state_dict(checkpoint['model_state_dict_2'])
    #player1= Random_Agent(env=env, player=1)
    player2 = Random_Agent(env=env, player=-1)
    test = Tester(env,player1, player2)
    print(test.test(1000))