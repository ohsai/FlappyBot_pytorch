import ple.games.flappybird as FlappyBird
from ple import PLE
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision.transforms as T
import cv2

game = FlappyBird.FlappyBird(pipe_gap=300)
env = PLE(game, fps=30,display_screen=True,force_fps=True,
        reward_values={
            "positive":0.9,
            "negative":0.0,
            "tick": 0.1,
            "loss": -1.1,
            })

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

print("Use GPU to Compute? : "+ str(use_cuda))

# Import Done



# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #Saves a transition.
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Replay Memory Done

# Deep Q-Network

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4,padding = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4, stride=2, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.mp3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512,2)

    def forward(self, input_tensor):
       # print(input_tensor.shape)
        x = F.relu(self.conv1(input_tensor))
        #print(x.shape)
        x = self.bn1(self.mp1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.bn2(x)
        #x = self.bn2(self.mp2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        #x = self.bn3(self.mp3(x))
        x = self.bn3(x)
        #print(x.shape)
        #print(x.view(x.size(0),-1).data.shape)
        x =  F.relu(self.fc1(x.view(x.size(0), -1)))
        #print(x.shape)
        x = self.fc2(x)
        return x


# Deep Q-Network Done

# Hyperparameters and Utilities

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.0001
EPS_END = 0.0001
EPS_DECAY = 3000000
observe_or_train = 0
OBSERVE = 20000
FRAME_PER_ACTION = 1
expected_q_value = 0

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(),lr=0.000001)
memory = ReplayMemory(30000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold and observe_or_train > OBSERVE:
        policy = model(Variable(state, volatile=True).type(FloatTensor))
        think = policy.data.max(1)[1].view(1, 1)
        #if(think[0][0] == 0):
            #print("think! jump!")
        return think
    else:
        rng = LongTensor([[random.randrange(2)]])
        #if(rng[0][0] == 0):
            #print("random! jump!")
        return rng
            


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def image_thresholding(x):
    if x < 0.01:
        return 0
    else:
        return 1
'''
resize = T.Compose([T.ToPILImage(),
                    T.Grayscale()])
resize2 = T.Compose([T.RandomRotation((90,90)),
                    T.Resize((80,80)),
                    T.ToTensor(),
                    lambda x : 0 if x < 0.01 else 1
                    ])
'''
def BCHW_format(state_screen):
    state_screen = cv2.resize(state_screen,(80,80))
    state_screen = cv2.cvtColor(state_screen, cv2.COLOR_BGR2GRAY)
    ret, state_screen = cv2.threshold(state_screen, 1, 255, cv2.THRESH_BINARY)
    #print(state_screen)
    #plt.imshow(state_screen, interpolation='nearest')
    #plt.show()
    #input()
    state_screen = np.reshape(state_screen,(80,80,1))
    state_screen = torch.from_numpy(state_screen)
    state_screen = state_screen.permute(2,0,1)
    state_screen = state_screen.unsqueeze(0).type(Tensor)
    return state_screen

def last_4_frames(frame1, frame2, frame3, frame4):
    frames_concatenated = torch.cat((frame1,frame2,frame3,frame4),1)
    return frames_concatenated

# Hyperparameters and Utilities Done

# Optimization
last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_q_value = expected_state_action_values

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Optimization Done

# Main part with game execution


env.init()

num_episodes = 10000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset_game()
    state = env.getScreenRGB()
    state = BCHW_format(state)
    #print(state.shape)
    frames = (state,state,state,state)
    state = last_4_frames(state,frames[1], frames[2],frames[3])

    #print("New Episode")
    for t in count():
        # Select and perform an action
        #env.render()
        action = select_action(state)
        if observe_or_train % FRAME_PER_ACTION != 0:
            action = torch.LongTensor([[1]])
            #print(action)
        reward = env.act(env.getActionSet()[action[0,0]])
        next_state = env.getScreenRGB()
        done = env.game_over()
        #next_state, reward, done, _ = env.step(action[0, 0])
        
        #print(reward)
        reward = Tensor([reward])
        
        if not done:
            next_state = BCHW_format(next_state)
            frames = (next_state, frames[0], frames[1], frames[2])
            next_state = last_4_frames(next_state,frames[1],frames[2],frames[3])
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward) # edit

        # Move to the next state
        state = next_state
        
        #print training info
        if observe_or_train <= OBSERVE:
            state_of_training = "observe"
        elif t > OBSERVE and t <= OBSERVE + EPS_DECAY :
            state_of_training = "explore"
        else:
            state_of_training = "train"
        print("TIMESTEP", observe_or_train, "/ STATE", state_of_training,\
             "/ ACTION", action[0,0],"/ REWARD", reward[0],"/ Expected_Q",expected_q_value)

        # Perform one step of the optimization (on the target network)
        if observe_or_train > OBSERVE :
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        else:
            if done:
                break
        #just observe or train
        observe_or_train += 1


# Main part with game execution Done

# Save Duration Data to text file
textfile = open('./episode_duration.txt','a')
textfile.write('###############New Experiment##########\n')
for duration in episode_durations:
    print(duration)
    textfile.write("%s\n" % duration)

textfile.close()

plt.ioff()
plt.show()
