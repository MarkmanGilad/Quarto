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

epochs = 2000000
start_epoch = 0
C = 3
learning_rate = 0.0001
batch_size = 64
env = env()
MIN_Buffer = 4000

def main ():
    
    player1 = DQN_Agent(player=1, env=env,parametes_path=None)
    player1_hat = DQN_Agent(player=1, env=env, train=False)
    Q1 = player1.DQN
    Q1_hat = Q1.copy()
    player1_hat.DQN = Q1_hat
    
    player2 = DQN_Agent(player=-1, env=env,parametes_path=None)
    player2_hat = DQN_Agent(player=-1, env=env, train=False)
    Q2 = player2.DQN
    Q2_hat = Q2.copy()
    player2_hat.DQN = Q2_hat

    
    #player2 = Random_Agent(player=-1, env=env)   

    buffer1 = ReplayBuffer(path=None) # None
    buffer2 = ReplayBuffer(path=None) # None

    results1, avgLosses1 =  [], []
    results2, avgLosses2 =  [], []
    avgLoss1 = 0
    avgLoss2 = 0
    loss1 = torch.Tensor([0])
    loss2 = torch.Tensor([0])
    start_epoch = 0
    res1, best_res1 = 0, -200
    loss_count1 = 0
    res2, best_res2 = 0, -200
    loss_count2 = 0
    step = 0
    best_random1= -100
    best_random_params1 = None
    score1 = 0 
    best_random2= -100
    best_random_params2 = None
    score2 = 0 
    
    tester1 = Tester(player1=player1, player2=Random_Agent(player=-1, env=env), env=env)
    tester2 = Tester(player1=Random_Agent(player=1, env=env), player2=player2, env=env)
    #random_results = [] #torch.load(random_results_path)   # []
    
    # init optimizer
    optim1 = torch.optim.Adam(Q1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optim1,100000*15, gamma=0.90)
    optim2 = torch.optim.Adam(Q2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optim2,100000*15, gamma=0.90)
    
    #region ######## checkpoint Load ############
    File_Num =8
    checkpoint_path = f"Data/checkpoint{File_Num}.pth"
    buffer_path1 = f"Data/buffer{File_Num}_1.pth"
    buffer_path2 = f"Data/buffer{File_Num}_2.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player1.DQN.load_state_dict(checkpoint['model_state_dict_1'])
        player1_hat.DQN.load_state_dict(checkpoint['model_state_dict_1'])
        optim1.load_state_dict(checkpoint['optimizer_state_dict_1'])
        scheduler1.load_state_dict(checkpoint['scheduler_state_dict_1'])
        buffer1 = torch.load(buffer_path1)
        buffer2=torch.load(buffer_path2)
        results1 = checkpoint['results_1']
        avgLosses1 = checkpoint['avglosses_1']
        avgLoss1 = avgLosses1[-1]
        results2 = checkpoint['results_2']
        avgLosses2 = checkpoint['avglosses_2']
        avgLoss2 = avgLosses2[-1]

    player1.DQN.train()
    player1_hat.DQN.eval()
    player2.DQN.train()
    player2_hat.DQN.eval()
    #endregion

    #region wandb
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
            # Sample Environement
            action_1 = player1.getAction(state_1, epoch=epoch)
            after_state_1 = env.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = env.reward(after_state_1, action_1) 
            step+=1
            if state_2 is not None:
                buffer2.push(state_2, action_2, reward_1*-1, after_state_1, end_of_game_1)
            if end_of_game_1:
                res1 += reward_1
                res2 += (reward_1 *-1)
                buffer1.push(state_1, action_1, reward_1, after_state_1, True)
                break
            
            state_2 = after_state_1
            action_2 = player2.getAction(state=state_2)
            after_state_2 = env.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = env.reward(after_state_2, action_2)
            step+=1
            if end_of_game_2:
                res1 += reward_2
                res2 += (reward_2 *-1)
                buffer2.push(state_2, action_2, reward_2*-1, after_state_1, True)
            buffer1.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer1) < MIN_Buffer:
                continue
            
            # Train P1 NN DDQN
            states, actions, rewards, next_states, dones = buffer1.sample(batch_size)
            Q_values = Q1(states, actions)
            #next_actions = player1_hat.get_Actions(next_states, dones) #DDQN player1.get_Actions(next_states, dones)
            next_actions = player1.get_Actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q1_hat(next_states, next_actions) 

            loss1 = Q1.loss(Q_values, rewards, Q_hat_Values, dones)
            loss1.backward()
            optim1.step()
            optim1.zero_grad()
            scheduler1.step()

            if loss_count1 <= 1000:
                avgLoss1 = (avgLoss1 * loss_count1 + loss1.item()) / (loss_count1 + 1)
                loss_count1 += 1
            else:
                avgLoss1 += (loss1.item()-avgLoss1)* 0.00001 


            # Train P2 NN DDQN
            states, actions, rewards, next_states, dones = buffer2.sample(batch_size)
            Q_values = Q2(states, actions)
            next_actions = player2.get_Actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q2_hat(next_states, next_actions) 

            loss2= Q2.loss(Q_values, rewards, Q_hat_Values, dones)
            loss2.backward()
            optim2.step()
            optim2.zero_grad()
            scheduler2.step()

            if loss_count2 <= 1000:
                avgLoss2 = (avgLoss2 * loss_count2 + loss2.item()) / (loss_count2 + 1)
                loss_count2 += 1
            else:
                avgLoss2 += (loss2.item()-avgLoss2)* 0.00001 


        if epoch % C == 0:
                Q1_hat.load_state_dict(Q1.state_dict())
                Q2_hat.load_state_dict(Q2.state_dict())

        if (epoch+1) % 100 == 0:
            print(f'\nres= {res1}')
            avgLosses1.append(avgLoss1)
            results1.append(res1)
            avgLosses2.append(avgLoss2)
            results2.append(res2)
            wandb.log ({
                "loss": avgLoss1,
                "result": res1,
                "score": score1
                })
            if best_res1 < res1:      
                best_res1 = res1
            if best_res2 < res2:      
                best_res2 = res2
            res1 = 0
            res2=0

        if (epoch+1) % 1000 == 0:
            test = tester1(100)
            score1 = test[0]-test[1]
            if best_random1 < score1:
                best_random1 = score1
                best_random_params1 = player1.DQN.state_dict()                  
            print("P1:" , test)
            test = tester2(100)
            score2 = test[0]-test[1]
            if best_random2 < score2:
                best_random2 = score2
                best_random_params2 = player2.DQN.state_dict()                  
            print("P2:" ,test)

        if (epoch) % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch, 
                'results_1': results1, 
                'avglosses_1':avgLosses1, 
                'model_state_dict_1': player1.DQN.state_dict(),
                'optimizer_state_dict_1': optim1.state_dict(),
                'scheduler_state_dict_1': scheduler1.state_dict(),
                'best_random_params_1': best_random_params1,

                'results_2': results2, 
                'avglosses_2':avgLosses2, 
                'model_state_dict_2': player2.DQN.state_dict(),
                'optimizer_state_dict_2': optim2.state_dict(),
                'scheduler_state_dict_2': scheduler2.state_dict(),
                'best_random_params_2': best_random_params2
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer1, buffer_path1)
            torch.save(buffer2, buffer_path2)
        

        print (f'epoch={epoch} loss={loss1.item():.5f} avgloss={avgLoss1:.5f} step={step}',  end=" ")
        print (f'learning rate={scheduler1.get_last_lr()[0]} path={checkpoint_path} res= {res1} best_res = {best_res1}')


if __name__ == '__main__':
    main()


