# Deep Q-Network
# Reinforcement Learning

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

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

#steps_done = 0

def select_action(state,policy_net,steps_done,device,EPS_START,EPS_END,EPS_DECAY,OBSERVE):
    #global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold and steps_done > OBSERVE:
        with torch.no_grad():
            return (policy_net(state).max(1)[1].view(1, 1),steps_done)
    else:
        return (torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long),steps_done)
           
# Optimization
last_sync = 0

def optimize_model(policy_net,target_net,memory,optimizer,device,BATCH_SIZE,GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

