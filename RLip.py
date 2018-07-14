import numpy as np
import torch
import cv2

def image_thresholding(x):
    if x < 0.01:
        return 0
    else:
        return 1

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
    state_screen = state_screen.unsqueeze(0).type(torch.cuda.FloatTensor)
    return state_screen

def last_4_frames(frame1, frame2, frame3, frame4):
    frames_concatenated = torch.cat((frame1,frame2,frame3,frame4),1)
    return frames_concatenated
