import os
import datetime
from pathlib import Path
import numpy as np

import actions
from metrics import MetricLogger
from agent import Hetasks
from environment import HetasksEnv
import TaskGen

import sys
from logs.log_device import Log_device
from logs.log_experiences import Log_experiences
import torch.multiprocessing as mp
from multiprocessing import Process
import time



if __name__ == "__main__":
    #Use this when needed -- Jetson Nano
    #mp.set_start_method("spawn")
    #mp.set_sharing_strategy('file_system')
    tg = TaskGen.TaskGen()
    input_q = mp.Queue()
    p_tasks = mp.Process(target=tg.start_test, args=(input_q,))
    p_tasks.start()
    logger_device= Log_device(0.01)

    save_dir_logs = Path('logs') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir_logs.mkdir(parents=True)

    path_com_log = save_dir_logs / 'logs_cpu_gpu.csv'
    path_net_log = save_dir_logs / 'logs_net.csv'

    p_logger1 = Process(target=logger_device.start_log, args=(path_com_log,))
    p_logger1.start()
    p_logger2 = Process(target=logger_device.start_log_net, args=(path_net_log,))
    p_logger2.start()
    time.sleep(10)

    # Initialize Hetasks environment
    # Limit the action-space to
    #   3 processors (CPU, GPU, Edge)
    #   2 models per type of tasks
    #   2 types of tasks
    # Apply Wrappers to environment
    computing_ft_size=4
    network_ft_size=4
    models_ft_size=4
    tasks_ft_size=4
    obs_size=np.array((computing_ft_size,models_ft_size,tasks_ft_size))
    batch_size=10
    num_type_task=2
    path_logs={'computing': path_com_log, 'network':path_net_log}
    th=np.array([0.15,0.12]) #Delay threasholds
    emd_dev=True
    env = HetasksEnv(actions.TWO_TASKS, obs_size,path_logs, th,batch_size, input_q, emd_dev,logger_device)
    env.reset()

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None#Path('checkpoints/2022-10-31T17-00-19/hetask_net_1.chkpt')

    hetasks = Hetasks(state_dim=tuple((2,4,4)),
                      action_dim=env.action_space, save_dir=save_dir, checkpoint=checkpoint)

    logger = MetricLogger(save_dir)
    
    fieldnames=['TIMESTAMP','STATE','ACTION','REWARD','NEXT_STATE']
    logger_exp = Log_experiences(save_dir_logs,fieldnames)
    tot_task=3000
    task_dispatched=0
    state = env.reset()
    while task_dispatched < tot_task:
        
        try:
            # Choose action
            action = np.random.choice(env.action_space)

            # Act
            next_state, reward, done, info = env.step(action)          

            #Store experience
            dict_logs={'TIMESTAMP':time.time(),'STATE':state,'ACTION':action,'REWARD':reward,'NEXT_STATE':next_state}
            logger_exp.add_line(dict_logs)

            # Replay memory
            #hetasks.cache(state, next_state, action, reward, done)

            # Learn
            #q, loss = hetasks.learn()

            # Logging
            #logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            task_dispatched=env.tasks_dispatched

        #logger.log_episode()
            
        except Exception as e:
            print(e)
            print("Exception catched")
    p_tasks.join()
    p_logger1.join()
    p_logger2.join()

