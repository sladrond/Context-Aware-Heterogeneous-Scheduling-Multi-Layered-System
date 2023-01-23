"""An environment for Hetasks: Deep RL Heterogenous task scheduler for constrainded devices using experiences Dataset"""
import sys

import torch
import psutil

sys.path.append('../')
import numpy as np
import Utils as util


class HetasksEnvDataset():

    def __init__(self, actions, obs_size, path, filename,filename2):
        # Use when needed in Jetson Nano
        # mp.set_sharing_strategy('file_system')
        self.state = None
        self.actions = actions
        self.action_space = len(actions)
        self.current_action = None
        self.reader = util.ExpsReader(path, filename,filename2)
        self.counter=0

    def step(self, action):
        #Read from file
        reward=-1
        try:
            self.state, reward = self.reader.get_exp(self.state,action)
        except:
            pass

        self.counter+=1
        return self.state, reward, self.is_terminal_state(), self.get_info()

    def is_terminal_state(self):
        # When the average delay per category is closest to the threadshold
        if self.counter == 5:
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def reset(self):
        self.state = np.zeros((43,))
        return self.state

    def get_info(self):
        pass

    def find_exp(self,r_dict):
        return self.reader.get_exp_detail(r_dict)

# explicitly define the outward facing API of this module
__all__ = [HetasksEnvDataset.__name__]
