
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


#env = gym.make('CartPole-v0').unwrapped
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
        #print("forward")
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        x =  self.head(x.view(x.size(0), -1))
        #print(x.shape)
        #print("return")
        return x


# Deep Q-Network Done

# Hyperparameters and Utilities

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

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
        #print("Hey")
        #print(Variable(state,volatile=True).type(FloatTensor).shape)
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


#data = np.zeros( (512,512,3), dtype=np.uint8)
#data[256,256:512] = [255,0,0]
#plt.figure()
#plt.imshow(data)
#plt.show()
#state_test = env.reset()
#print("Hello World!")
#print(env.action_space)
#print(env.observation_space.high)
#print(env.observation_space.low)


num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = BCHW_format(state)
    #last_screen = get_screen()
    #current_screen = get_screen()
    #state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        env.render()
        #env.render(mode='rgb_array')
        action = select_action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        reward = Tensor([reward])

        # Observe new state
        #image = np.random.randint(255,size=(512,288,3)).astype('uint8')
        #plt.imshow(BCHW_format(state_test).cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
        #image = next_state
        #plt.imshow(image)
        #plt.title('Example extracted screen')
        #plt.show()
        #print(image)
        #image = next_state

        #print(next_state)
        #last_screen = current_screen
        #current_screen = get_screen()
        if not done:
            #next_state = current_screen - last_screen
            next_state = BCHW_format(next_state)
            #image = np.transpose(next_state.squeeze(),(1,2,0))
            #plt.imshow(image)
            #plt.title('Example extracted screen')
            #plt.show()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
#env.render(close=True)
env.close()
#plt.ioff()
#plt.show()

# Main part with game execution Done

