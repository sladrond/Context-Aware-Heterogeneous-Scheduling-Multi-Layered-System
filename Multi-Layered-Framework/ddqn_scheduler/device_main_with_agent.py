import datetime
from pathlib import Path
import numpy as np
import argparse

import actions
from metrics import MetricLogger
from agent import Hetasks
from environment_from_files import HetasksEnvDataset
from logs.log_experiences import Log_experiences, Log_plots
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that will learn from experiences saved in a file')
    parser.add_argument("-sp","--scenario_path", help="Path to the folder that contains the data of a specific scenario. "
                                                "Usually a folder of folders", default="")
    parser.add_argument("-f", "--filename",
                        help="File name of the experiences saved", default="")
    parser.add_argument("-a", "--agent",
                        help="Agent name", default="")
    parser.add_argument("-fd", "--filename_detail",
                        help="Details in the file", default="")
    parser.add_argument("-l", "--learning",
                        help="Whether or not learning is activated", default="")
    parser.add_argument("-chk", "--checkpoint",
                        help="The path of the checkpoint", default="")
    args=parser.parse_args()
    save_dir_logs = Path('logs') / args.agent / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir_logs.mkdir(parents=True)

    time.sleep(10)

    computing_ft_size = 4; network_ft_size = 4
    models_ft_size = 4; tasks_ft_size = 4
    obs_size = np.array((computing_ft_size, models_ft_size, tasks_ft_size))
    env = HetasksEnvDataset(actions.TWO_TASKS, obs_size, args.scenario_path, args.filename, args.filename_detail)
    env.reset()

    save_dir = Path('checkpoints') / args.agent / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None #Path('checkpoints/'+ args.checkpoint)
    hetasks = Hetasks(state_dim=tuple((2, 4, 4)),
                      action_dim=env.action_space, save_dir=save_dir, checkpoint=checkpoint)

    fieldnames=['TIMESTAMP','STATE','ACTION','REWARD','NEXT_STATE']
    logger_exp = Log_experiences(save_dir_logs,fieldnames)

    fieldnames=['TIMESTAMP','STATE', 'ACTION', 'REWARD', 'NEXT_STATE', 'DELAY','ACCURACY', 'ACTION_NAME','CURRENT']
    logger_exp_detail = Log_plots(save_dir_logs,fieldnames)

    logger = MetricLogger(save_dir)

    episodes = 1000000
    e=0
    nfound_counter=0
    found_counter=0
    k=100

    while k>0.05:

        state = env.reset()
        e+=1

        while True:

            # Get action
            action = hetasks.act(state)

            # Act
            next_state, reward, done, info = env.step(action)

            if e % 1000 == 0:
                dict_logs = {'TIMESTAMP': time.time(), 'STATE': state, 'ACTION': action, 'REWARD': reward,
                             'NEXT_STATE': next_state}
                logger_exp.add_line(dict_logs)

            if reward == -1:
                nfound_counter+=1
            else:
                found_counter+=1
                if e % 1000 == 0:
                    res = env.find_exp({'ACTION': action, 'REWARD': reward, 'NEXT_STATE': next_state,
                                            'STATE': state})
                    if res != -1:
                        logger_exp_detail.add_line(res)
            k = 1-(found_counter/(nfound_counter+found_counter))

            # Memory replay
            hetasks.cache(state, next_state, action, reward, done)

            if args.learning == "True":
                # Learn
                q, loss = hetasks.learn()

                # Logging
                logger.log_step(reward, loss, q)
            else:
                pass
                #print("Not learning")

            # Update state
            state = next_state

            # Check if end of game
            if done:
                break

        logger.log_episode()

        if e % 1000 == 0:
            print("NOT FOUND COUNTER: ", (nfound_counter/(nfound_counter+found_counter))*100, " % ")
            print("FOUND COUNTER: ", (found_counter/(nfound_counter+found_counter))*100, " % ")
            logger.record(
                episode=e,
                epsilon=hetasks.exploration_rate,
                step=hetasks.curr_step
            )
