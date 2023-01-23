"""An environment for Hetasks: Deep RL Heterogenous task scheduler for constrainded devices"""
import sys

import torch
import psutil
sys.path.append('../')
import numpy as np
import time, math
import Dispatcher
import torch.multiprocessing as mp
import pandas
import zmq
import pickle, zlib
from torchvision import transforms
from PIL import Image
import random

def run_dispatcher(model_label, task_list, dispatcher, processor, batch_number):
    dispatcher.execute_task_laptop(task_list, model_label, processor, batch_number, )


class HetasksEnv():

    def __init__(self, actions, obs_size, log_paths, th, batch_size, input_queue, emd_dev,logger):
        # Use when needed -- Jetson Nano
        #mp.set_sharing_strategy('file_system')
        self.state = None
        self.action_space = len(actions)
        self.actions = actions
        self.obs_size = obs_size
        self.start_step = time.time()
        self.end_step = time.time()
        self.agent_state = np.zeros(obs_size)
        self.log_paths = log_paths
        self.batch_size = batch_size
        self.dispatcher = Dispatcher.Dispatcher()
        self.batch_number = 0
        self.input_q = input_queue
        self.stats = mp.Queue()
        self.gpu_q = []
        self.cpu_q = []
        self.server_q = []
        self.delta = 2
        self.th = th
        self.tasks_dispatched=0
        self.cpu_mem_tot = psutil.virtual_memory().total * 8
        self.embed_dev = emd_dev
        self.ipadd= '127.0.0.1'# ES - ip address
        self.port='5555' # Communication Port
        self.logger=logger
        self.counter=0
        self.current_action=None
        self.possible_tasks=[0,3]


        if self.embed_dev == False:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_mem_tot = info.total
            nvidia_smi.nvmlShutdown()
        else:
            self.gpu_mem_tot=41000

    def run_dispatcher_mp(self, model_label, task_list, processor, ):
        self.dispatcher.execute_task_laptop(task_list, model_label, processor, self.batch_number,
                                            self.stats, )

    def step(self, action):
        action = self.actions[action]
        self.current_action=action
        while self.update_queues(action) == False:
            time.sleep(1)
        else:
            self.batch_number += 1
            self.start_step = time.time()

            for idx, element in enumerate(action):
                if element[1] == 'gpu' and len(self.gpu_q)>0:
                    model_gpu = element[0];
                    type_task = self.possible_tasks[idx]
                    if not self.dispatcher.is_loaded(model_gpu, 'gpu', type_task):
                        self.dispatcher.load_model_in_processor(model_gpu, 'gpu', type_task)
                    list_gpu = [task for task in self.gpu_q if task.get_type() == type_task]
                    self.run_dispatcher_mp(model_gpu, list_gpu, 'gpu')

                if element[1] == 'cpu' and len(self.cpu_q)>0:
                    model_cpu = element[0];
                    type_task = self.possible_tasks[idx]
                    if not self.dispatcher.is_loaded(model_cpu, 'cpu', type_task):
                        self.dispatcher.load_model_in_processor(model_cpu, 'cpu', type_task)
                    list_cpu = [task for task in self.cpu_q if task.get_type() == type_task]
                    self.run_dispatcher_mp(model_cpu, list_cpu, 'cpu')

                if element[1]=='edge' and len(self.server_q)>0:
                    self.stats.put(self.dispatcher.send_task(self.server_q, self.ipadd, self.port))          

            self.server_q=[]
            self.gpu_q = []
            self.cpu_q = []

            self.end_step = time.time()
            self.set_state(self.start_step, self.end_step)
            reward = self.get_reward()
        #print("# TASKS DISPATCHED: ", self.tasks_dispatched)
        return self.state, reward, self.is_terminal_state(), self.get_info()

    def check_cpu(self, size):
        avail = ((8 - self.state[2]) * self.cpu_mem_tot) / 1e6
        data = size / 1e6
        if data >= avail:
            return False
        else:
            return True

    def check_gpu(self, size):
        avail = ((1 - self.state[3]) * self.gpu_mem_tot) / 1e6
        data = size / 1e6
        if data >= avail:
            return False
        else:
            return True

    def update_queues(self, action):
        t_enqueued = (len(self.server_q) + len(self.cpu_q) + len(self.gpu_q))
        if t_enqueued >= self.batch_size:
            return True
        else:
            tot_task = np.min([self.input_q.qsize(), self.batch_size])
            data_sizes_gpu = [];
            size_cpu = 0  # sizes are in bytes
            data_sizes_cpu = [];
            size_gpu = 0

            for _ in range(tot_task):
                one_task = self.input_q.get()
                index=[one_task.get_type() if one_task.get_type()== 0 else 1][0]
                if action[index][1] == 'gpu':
                    self.gpu_q.append(one_task)
                    data_sizes_gpu.append(one_task.get_size())
                elif action[index][1] == 'cpu':
                    self.cpu_q.append(one_task)
                    data_sizes_cpu.append(one_task.get_size())
                else:
                    self.server_q.append(one_task)

            if len(data_sizes_cpu): size_cpu = max(data_sizes_cpu)

            if self.check_cpu(size_cpu) and self.check_gpu(size_gpu):
                if (len(self.server_q) + len(self.cpu_q) + len(self.gpu_q))>0:
                    return True
                else:
                    return False
            else:
                return  False


    def get_accuracy(self):
        acc=[]
        for element in self.current_action:
            if 'b0' in element[0]: 
                num = random.choice(np.array(range(1,70)))/100
            else:
                num = random.choice(np.array(range(30,100)))/100
            acc.append(num)
        
        res=np.mean(acc)
        return res

    def get_avg_curr(self):
        res=(self.state[4]-2700)/(5000-2700)
        return res

    def get_reward(self):
        weights = np.array([0.5,0.1,0.4])
        thr=np.array([-11.5,-0.5,-0.8])
        delay = (self.end_step - self.start_step) / self.batch_size
        #shift = -0.5 + self.delta
        acc =  self.get_accuracy()
        avg_curr =self.get_avg_curr()
        current_data=np.array([delay,acc,avg_curr])
        current_data=np.sum([current_data,thr], axis=0)
        current_data[0]=current_data[0]/60
        x=float(np.dot(weights,current_data))
        reward = 1 / (1 + np.exp(x))
        return reward


    def is_terminal_state(self):
        # When the average delay per category is closest to the threadshold
        if self.counter == 10:
            self.counter=0
            return True
        else:
            self.counter+=1
            try:
                return max(self.th) > self.avg_delay
            except:
                return False

    def get_com_fts(self, begin, end):
        if self.embed_dev == True:
            res=self.logger.get_com_fts()
            cpu_load = res['CPU_LOAD']
            gpu_load = res['GPU_LOAD']
            cpu_mem = res['MEM']
            gpu_mem = cpu_mem
            current = res['CURR']
            return np.array([cpu_load, cpu_mem, gpu_load, gpu_mem,current])
        else:
            com = pandas.read_csv(self.log_paths['computing'], delimiter=',')
            cpu_load = com[(com['TIME'] > begin) & (com['TIME'] < end)]['cpu_load'].mean()
            gpu_load = com[(com['TIME'] > begin) & (com['TIME'] < end)]['gpu_load'].mean()
            cpu_mem = com[(com['TIME'] > begin) & (com['TIME'] < end)]['cpu_mem'].mean()
            gpu_mem = com[(com['TIME'] > begin) & (com['TIME'] < end)]['gpu_mem'].mean()
            return np.array([cpu_load, cpu_mem, gpu_load, gpu_mem])

    def get_net_fts(self, begin, end):
        net = pandas.read_csv(self.log_paths['network'], delimiter=',')
        net_top = [col for col in net.columns if col != 'TIME']
        net_ft = np.zeros(shape=(len(net_top),), dtype=float)
        while net[(net['TIME'] > begin)].empty:
            time.sleep(1)
            net = pandas.read_csv(self.log_paths['network'], delimiter=',')
            net_top = [col for col in net.columns if col != 'TIME']
            net_ft = np.zeros(shape=(len(net_top),), dtype=float)
        else:
            for idx, key in enumerate(net_top):
                net_ft[idx] = net[(net['TIME'] > begin)][key].mean()
        return np.array(net_ft)

    def get_model_fts(self):
        return np.hstack(np.array(self.dispatcher.get_model_fts()))

    def get_tasks_fts(self):
        task_dist = np.zeros(len(self.th))
        if self.stats.empty(): return np.zeros(2 + len(self.th))

        avg_delay = 0.0;
        tot_tasks = 0.0
        while not self.stats.empty():
            elem = self.stats.get()
            task_dist[self.possible_tasks.index(elem[1])] += elem[0]
            tot_tasks += elem[0]
            avg_delay = elem[2]

        task_dist = task_dist / tot_tasks
        avg_delay = avg_delay / tot_tasks
        self.tasks_dispatched+=tot_tasks
        self.avg_delay = avg_delay
        return np.hstack((tot_tasks, task_dist, avg_delay))


    def set_state(self, begin, end):
        fts_com = self.get_com_fts(begin, end)
        fts_mod = self.get_model_fts()
        fts_tas = self.get_tasks_fts()
        fts_net = self.get_net_fts(begin,end)
        self.state = np.nan_to_num(np.concatenate((fts_com, fts_mod,fts_tas,fts_net)))
        np.set_printoptions(suppress=True)

    def reset(self):
        self.state = np.zeros((42,))
        return self.state

    def get_info(self):
        pass


# explicitly define the outward facing API of this module
__all__ = [HetasksEnv.__name__]
