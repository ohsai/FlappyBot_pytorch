import ple.games.flappybird as FlappyBird
from ple import PLE

from itertools import count
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

import RL
import RLplot
import RLip

import pickle
import datetime

def main():
    # Setup If GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print("Use GPU to Compute? : "+ str(use_cuda))

    # Setup Hyperparameters, Networks, memory, optimizer
    reward_system = {
            "positive":0.9,
            "negative":0.0,
            "tick": 0.1,
            "loss": -1.1,
            }
    PIPEGAP = 200
    BATCH_SIZE = 32
    learning_rate = 1e-4
    MEMORY_SIZE = 30000
    GAMMA = 0.99
    EPS_START = 0.0001
    EPS_END = 0.0001
    EPS_DECAY = 3000000
    OBSERVE = 30000
    FRAME_PER_ACTION = 1
    TARGET_UPDATE = 10
    num_episodes = 1000
    for PIPEGAP in [101,150,200,250,300]:
        for BATCH_SIZE in [16,32,64,128,256,512]:
            for learning_rate in [1e-2,1e-3,1e-4,1e-5,1e-6]:
                experiment(device, 
                    reward_system, PIPEGAP,BATCH_SIZE,learning_rate, 
                    MEMORY_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,OBSERVE,
                    FRAME_PER_ACTION,TARGET_UPDATE,num_episodes)
    
def experiment(device,
        reward_system, PIPEGAP, BATCH_SIZE, learning_rate, 
        MEMORY_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, OBSERVE, 
        FRAME_PER_ACTION, TARGET_UPDATE, num_episodes, 
        save_model = False, load_model = False, load_model_path_prefix = None):
    expected_q_value = 0
    
    policy_net = RL.DQN().to(device)
    target_net = RL.DQN().to(device)
    if load_model:
        policy_net.load_state_dict(torch.load(load_model_path_prefix + "_policy_net.mdl"))
        target_net.load_state_dict(torch.load(load_model_path_prefix + "_target_net.mdl"))
    else :
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(),lr=learning_rate)
    memory = RL.ReplayMemory(MEMORY_SIZE)

    #Setup Game environment
    game = FlappyBird.FlappyBird(pipe_gap=PIPEGAP)
    env = PLE(game, fps=30,display_screen=True,force_fps=True,reward_values=reward_system)

    #Setup plot    
    RLplot.plot_init()
    episode_durations = []

    # Main part with game execution

    env.init()
    steps_done = 0;
    infinity = False

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset_game()
        state = env.getScreenRGB()
        state = RLip.BCHW_format(state)
        frames = (state,state,state,state)
        state = RLip.last_4_frames(state,frames[1], frames[2],frames[3])

        for t in count():
            # Select an action
            action,steps_done = RL.select_action(state,policy_net,steps_done,device,EPS_START,EPS_END,EPS_DECAY,OBSERVE)
            if steps_done % FRAME_PER_ACTION != 0:
                action = torch.tensor([[1]],device=device,dtype=torch.long)
            
            # Perform an action
            reward = env.act(env.getActionSet()[action[0,0]])
            next_state = env.getScreenRGB()
            done = env.game_over()
            reward = torch.tensor([reward], device=device)
            
            # Formatting next state for network
            if not done:
                next_state = RLip.BCHW_format(next_state)
                frames = (next_state, frames[0], frames[1], frames[2])
                next_state = RLip.last_4_frames(next_state,frames[1],frames[2],frames[3])
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward) # edit

            # Move to the next state
            state = next_state
            
            # Print Log of training info
            if steps_done <= OBSERVE:
                state_of_training = "observe"
            elif steps_done > OBSERVE and steps_done <= OBSERVE + EPS_DECAY :
                state_of_training = "explore"
            else:
                state_of_training = "train"
            print("TIMESTEP", steps_done, "/ STATE", state_of_training,\
                 "/ ACTION", action[0,0].data,"/ REWARD", reward[0].data,"/ Expected_Q",expected_q_value)

            # Perform one step of the optimization (on the target network)
            if steps_done > OBSERVE :
                RL.optimize_model(policy_net,target_net,memory,optimizer,device,BATCH_SIZE,GAMMA)
                if done:
                    episode_durations.append(t + 1)
                    RLplot.plot_durations(episode_durations)
                    break
                if t > 10000 :
                    infinity = True;
                    episode_durations.append(t + 1)
                    RLplot.plot_durations(episode_durations)
                    break;
            else:
                if done:
                    break

        # Update the target network
        if i_episode % TARGET_UPDATE == 0 and steps_done > OBSERVE:
            target_net.load_state_dict(policy_net.state_dict())
        if infinity :
            break;
    # End training process
    # Save experiment result 
    data={
            "data" : episode_durations,
            'pipe_gap' : PIPEGAP,
            'reward_values' : reward_system,
            'BATCH_SIZE' : BATCH_SIZE,
            'learning_rate' : learning_rate,
            'MEMORY_SIZE' : MEMORY_SIZE,
            'GAMMA' : GAMMA,
            'EPS_START' : EPS_START,
            'EPS_END' : EPS_END,
            'EPS_DECAY' : EPS_DECAY,
            'OBSERVE' : OBSERVE,
            'FRAME_PER_ACTION' : FRAME_PER_ACTION,
            'TARGET_UPDATE' : TARGET_UPDATE,
            'num_episodes' : num_episodes
            }
    filenameprefix = './result/Expe_'+ datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename = filenameprefix + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # Save model if said so
    if save_model:
        torch.save(policy_net.state_dict(),filenameprefix + '_policy_net.mdl')
        torch.save(target_net.state_dict(),filenameprefix + '_target_net.mdl')

    # Save plot figure
    plotname = filenameprefix + '.png'
    RLplot.plot_end(plotname)

main()
