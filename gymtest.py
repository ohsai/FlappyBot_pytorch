
# Import

import gym
from gym.wrappers import Monitor
import gym_ple
import math
import random
import numpy as np
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
import torchvision.transforms as T

env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])
outdir = './random-agent-results'
env = Monitor(env,directory=outdir,force=True)
env.seed(0)

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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(672, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x =  self.head(x.view(x.size(0), -1))
        return x


# Deep Q-Network Done

# Hyperparameters and Utilities

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.001
EPS_DECAY = 1000

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


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

resize = T.Compose([T.ToPILImage(),
                    T.Resize(45, interpolation=Image.CUBIC),
                    T.ToTensor()])

def BCHW_format(state_screen):
    #print(state_screen.shape)
    state_screen = state_screen.transpose((2,0,1))
    state_screen = torch.from_numpy(state_screen)
    state_screen = resize(state_screen).unsqueeze(0).type(Tensor)
    #print(state_screen.shape)
    return state_screen

def last_4_frames(frame1, frame2, frame3, frame4):
    4_frames_concatenated = torch.cat((frame1,frame2,frame3,frame4),2)
    return 4_frames_concatenated

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

#logger = logging.getLogger()
#logger.setLevel(logging.INFO)

num_episodes = 2000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = BCHW_format(state)

    4_frames = [state,state,state,state]
    state = last_4_frames(state,4_frames[1],4_frames[2],4_frames[3])

    print("New Episode")
    for t in count():
        # Select and perform an action
        env.render()
        action = select_action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        
        if reward < 0:
            reward = -10
        else:
            reward = 1
        #print(reward)
        
        reward = Tensor([reward])
        
        if not done:
            next_state = BCHW_format(next_state)
        else:
            next_state = None

        # Store the transition in memory
        4_frames = [next_state, 4_frames[0], 4_frames[1], 4_frames[2]]
        next_state = last_4_frames(next_state,4_frames[1],4_frames[2],4_frames[3])
        memory.push(state, action, next_state, reward) # edit

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')

# Main part with game execution Done

# Save Duration Data to text file
textfile = open('./episode_duration.txt','a')
textfile.write('###############New Experiment##########\n')
for duration in episode_durations:
    print(duration)
    textfile.write("%s\n" % duration)

textfile.close()

env.close()
plt.ioff()
plt.show()
