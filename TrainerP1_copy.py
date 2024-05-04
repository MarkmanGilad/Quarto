from Quarto import Quarto as env
import wandb
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Random_Agent import Random_Agent
import torch
from Tester import Tester
import os
from Constant import *

# region ####### parameters
epochs = 2000000
start_epoch = 0
C = 50
learning_rate = 0.0001
batch_size = 64
env = env()
MIN_Buffer = 4000
#endregion

def main ():
    
    #region ######### init ########
    player1 = DQN_Agent(player=1, env=env,parametes_path=None)
    player2 = DQN_Agent(player=-1, env=env,parametes_path=None)
    player_hat = DQN_Agent(player=1, env=env, train=False)
    player2.DQN = player1.DQN
    Q = player1.DQN
    Q_hat = Q.copy()
    # Q_hat.train = False
    player_hat.DQN = Q_hat
    
    
    buffer = ReplayBuffer(path=None) # None

    results, avgLosses =  [], []
    avgLoss = 0
    loss = torch.Tensor([0])
    start_epoch = 0
    res, best_res = 0, -200
    loss_count = 0
    step = 0
    best_random = -100
    best_random_params = None
    score, score1, score2 = 0, 0, 0
    scores1, scores2 = [], []
    
    
    tester1 = Tester(player1=player1, player2=Random_Agent(player=-1, env=env), env=env)
    tester2 = Tester(player1=Random_Agent(player=1,env=env), player2=player2, env=env)
    #random_results = [] #torch.load(random_results_path)   # []
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,100000*15, gamma=0.90)
    #endregion 

    #region ######## checkpoint Load ############
    File_Num = 3
    checkpoint_path = f"Data/checkpoint{File_Num}.pth"
    buffer_path = f"Data/buffer{File_Num}_1.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player1.DQN.load_state_dict(checkpoint['model_state_dict_1'])
        player_hat.DQN.load_state_dict(checkpoint['model_state_dict_1'])
        optim.load_state_dict(checkpoint['optimizer_state_dict_1'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict_1'])
        buffer = torch.load(buffer_path)
        results = checkpoint['results_1']
        avgLosses = checkpoint['avglosses_1']
        avgLoss = avgLosses[-1]

    player1.DQN.train()
    player_hat.DQN.eval()
    #endregion
    
    #region ######## wandb #########
    wandb.init(
        #set the wandb project where this run will be logged
        project="Quarto",
        resume=False,
        id=f"Quarto Run - {File_Num}",
        config={
            "name": f"Quarto Run - {File_Num}",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "start_epoch": start_epoch,
            "decay": epsiln_decay,
            "gamma": 0.99,
            "batch_size": batch_size,
            "C": C
        }
    )
    #endregion
    
    for epoch in range(start_epoch, epochs):
        step = 0
        print(f'epoch = {epoch}', end='\r')
        state_1 = env.startState()
        state_2 = None
        while not env.is_end_of_game(state_1):
            #region ########## Sample Environement ##########
            action_1 = player1.getAction(state_1, epoch=epoch)
            after_state_1 = env.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = env.reward(after_state_1, action_1) 
            step+=1
            if state_2 is not None:
                buffer.push(state_2, action_2, reward_1*-1, after_state_1, end_of_game_1)
            if end_of_game_1:
                res += reward_1
                buffer.push(state_1, action_1, reward_1, after_state_1, True)
                break
            state_2 = after_state_1
            action_2 = player2.getAction(state=state_2, epoch=epoch)
            after_state_2 = env.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = env.reward(after_state_2, action_2)
            step+=1
            if end_of_game_2:
                res += reward_2
                buffer.push(state_2, action_2, reward_2*-1, after_state_1, True)
            buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer) < MIN_Buffer:
                continue
            #endregion

            #region ############# Train NN ###############
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states, actions)
            # next_actions = player_hat.get_Actions(next_states, dones) # DQN
            next_actions = player1.get_Actions(next_states, dones) # DDQN
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions) 

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            #endregion
        
        if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())

        #region ######### log & save ###########
        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            wandb.log ({
                "loss": avgLoss,
                "result": res,
                "score": score,
                "scores1": score1,
                "scores2": score2
                })
            if best_res < res:      
                best_res = res
            res = 0

        if (epoch) % 1000 == 0 and epoch > 0:
            test1 = tester1(100)
            score1 = test1[0]-test1[1]
            if best_random < score1:
                best_random = score1
                best_random_params = player1.DQN.state_dict()                  
            print(test1)
            test2 = tester2(100)
            score2 = test2[1]-test2[0]
            print(test2)
            scores1.append(score1)
            scores2.append(score2)


        if (epoch) % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch, 
                'results_1': results, 
                'avglosses_1':avgLosses, 
                'model_state_dict_1': player1.DQN.state_dict(),
                'optimizer_state_dict_1': optim.state_dict(),
                'scheduler_state_dict_1': scheduler.state_dict(),
                'best_random_params_1': best_random_params
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)
        #endregion

        print (f'epoch={epoch} loss={loss.item():.5f} avgloss={avgLoss:.5f} step={step}',  end=" ")
        print (f'learning rate={scheduler.get_last_lr()[0]} path={checkpoint_path} res= {res} best_res = {best_res}')


if __name__ == '__main__':
    main()


