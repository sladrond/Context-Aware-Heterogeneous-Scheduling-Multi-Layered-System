#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
import zmq
import zlib
import pickle
import Dispatcher
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from efficientnet_pytorch import EfficientNet
from transformers import SegformerFeatureExtractor, SegformerForImageClassification,SegformerForSemanticSegmentation
import torch.multiprocessing as mp
import numpy as np
import time
from logs.log_laptop import Log_com_laptop, Log_net_laptop
from pathlib import Path
import datetime
from multiprocessing import Process
import pandas

logger_com= Log_com_laptop(0.01)
logger_net= Log_net_laptop(0.01)
save_dir_logs = Path('results_only_server') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir_logs.mkdir(parents=True)
path_com_log = save_dir_logs / 'logs_cpu_gpu.csv'
path_net_log = save_dir_logs / 'logs_net.csv'

path_logs={'computing': path_com_log, 'network':path_net_log}

p_logger2 = Process(target=logger_net.start_log_net, args=(path_net_log,))
p_logger2.start()
p_logger1 = Process(target=logger_com.start_log, args=(path_com_log,))
p_logger1.start()

time.sleep(10)

dispatcher=Dispatcher.Dispatcher()
TASK_MODEL={
            3:
                    [{'high':'nvidia/segformer-b1-finetuned-ade-512-512', 'object': SegformerForSemanticSegmentation.from_pretrained},
                    {'low':'nvidia/segformer-b0-finetuned-ade-512-512', 'object': SegformerForSemanticSegmentation.from_pretrained}],
            0:
                    [{'high':'efficientnet-b4', 'object':EfficientNet.from_pretrained },
                    {'low':'efficientnet-b0', 'object':EfficientNet.from_pretrained } ]}

def load_models():  
    dispatcher.server=True 
    for type_task, value in TASK_MODEL.items():
        model_gpu=value[0]['high']
        dispatcher.load_model_in_processor(model_gpu, 'gpu', type_task)
        model_gpu=value[1]['low']
        dispatcher.load_model_in_processor(model_gpu, 'gpu', type_task)

def get_com_fts(begin, end):
    com = pandas.read_csv(path_logs['computing'], delimiter=',')
    com_top = [col for col in com.columns if col != 'time']
    com_ft = np.zeros(shape=(len(com_top),), dtype=float)
    while com[(com['time'] > begin)].empty:
        time.sleep(1)
        com = pandas.read_csv(path_logs['computing'], delimiter=',')
        com_top = [col for col in com.columns if col != 'time']
        com_ft = np.zeros(shape=(len(com_top),), dtype=float)
    else:
        for idx, key in enumerate(com_top):
            com_ft[idx] = com[(com['time'] > begin)][key].mean()
        return np.nan_to_num(com_ft)

def get_net_fts(begin, end):
    net = pandas.read_csv(path_logs['network'], delimiter=',')
    net_top = [col for col in net.columns if col != 'TIME']
    net_ft = np.zeros(shape=(len(net_top),), dtype=float)
    while net[(net['TIME'] > begin)].empty:
        time.sleep(1)
        net = pandas.read_csv(path_logs['network'], delimiter=',')
        net_top = [col for col in net.columns if col != 'TIME']
        net_ft = np.zeros(shape=(len(net_top),), dtype=float)
    else:
        for idx, key in enumerate(net_top):
            net_ft[idx] = net[(net['TIME'] > begin)][key].mean()
        #print(np.nan_to_num(net_ft))
        return np.nan_to_num(net_ft)

def get_model_fts():
    return np.hstack(np.array(dispatcher.get_model_fts()))

def get_tasks_fts(stats):
    tasks_dispatched = 1
    possible_tasks=[0,3]
    task_dist = np.zeros(2,)
    if stats.empty(): return np.zeros(3)

    avg_delay = 0.0;
    tot_tasks = 0.0
    while not stats.empty():
        elem = stats.get()
        task_dist[possible_tasks.index(elem[1])] += elem[0]
        tot_tasks += elem[0]
        avg_delay = elem[2]

    task_dist = task_dist / tot_tasks
    avg_delay = avg_delay / tot_tasks
    tasks_dispatched+=tot_tasks
    tasks_dist = np.hstack(task_dist)
    avg_delay = avg_delay
    return np.hstack((tot_tasks, task_dist, avg_delay))

def get_state(begin,end,stats):
    fts_com = get_com_fts(begin, end)
    fts_mod = get_model_fts()
    fts_tas = get_tasks_fts(stats)
    fts_net = get_net_fts(begin,end)
    
    return np.nan_to_num(np.concatenate((fts_com, fts_mod,fts_tas,fts_net,[0])))

def start():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    batch_number=0
    stats_0=mp.Queue()
    stats_3=mp.Queue()
    while True:
        #  Wait for next request from client
        message = socket.recv()
        #print("Received request: %s" % message)
        p = zlib.decompress(message)
        #print("received decompressed ", p)
        t=pickle.loads(p) #Tasks received
        #end=time.time()
        #print("type: ", t.get_type())
        #model.load_state_dict(torch.load(message, map_location="cuda:0"))
        #print("list of tasks ", len(t))

        #print("type of data ", t[0][0].get_type())
        #print(t)
        begin=time.time()
        list_0=[t[0][i] for i in range(len(t[0])) if t[0][i].get_type()==0]
        list_3=[t[0][i] for i in range(len(t[0])) if t[0][i].get_type()==3]
        '''
        if 'edge' in t[1][0][1] and 'b0' not in t[1][0][0]:
            model_label_0=TASK_MODEL[0][0]['high']        
        elif 'edge' in t[1][0][1] and 'b0' in t[1][0][0]:
            model_label_0=TASK_MODEL[0][1]['low']

        if 'edge' in t[1][0][1] and 'b0' not in t[1][0][0]:
            model_label_3=TASK_MODEL[3][0]['high']        
        elif 'edge' in t[1][0][1] and 'b0' in t[1][0][0]:
            model_label_3=TASK_MODEL[3][1]['low']
        '''
        model_label_0 = t[1][0][0] 
        model_label_3 = t[1][1][0]
        #print("MODEL FOR 0 ", model_label_0)
        #print("MODEL FOR 3 ", model_label_3)
        dispatcher.execute_task_laptop(list_0, model_label_0, 'gpu', batch_number,stats_0, )
        dispatcher.execute_task_laptop(list_3, model_label_3, 'gpu', batch_number,stats_3, )

        end=time.time()
        tmp=np.zeros((3,))

        if stats_0.qsize() > 0:
            tmp+=np.array(stats_0.get())
            state=get_state(begin,end,stats_0)
        
        if stats_3.qsize() > 0:
            tmp+=np.array(stats_3.get())
            state=get_state(begin,end,stats_3)
        
        batch_number+=1

 
        #print("STATS FROM SERVER: ", state)
    
        #tmp=np.zeros((3,))
        #  Send reply back to client
        p = pickle.dumps(state)
        z = zlib.compress(p)
        socket.send(z)
        #socket.send(b'done')
        #print("done")