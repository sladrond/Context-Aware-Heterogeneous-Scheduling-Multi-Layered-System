import numpy as np
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from efficientnet_pytorch import EfficientNet
from transformers import SegformerFeatureExtractor, SegformerForImageClassification,SegformerForSemanticSegmentation

MODEL={'facebook/s2t-small-librispeech-asr':None,
       'facebook/wav2vec2-base-960h':None,
       'roberta-base': RobertaForQuestionAnswering.from_pretrained,
       'distilbert-base-uncased': DistilBertForQuestionAnswering,
       'efficientnet-b0': EfficientNet.from_pretrained,
       'efficientnet-b4': EfficientNet.from_pretrained,
       'nvidia/segformer-b0-finetuned-ade-512-512': SegformerForSemanticSegmentation.from_pretrained,
       'nvidia/segformer-b1-finetuned-ade-512-512': SegformerForSemanticSegmentation.from_pretrained}

#Restrictions:
TASK_MODEL_PROCESSOR={
		     3:
                      [{'model':'nvidia/segformer-b0-finetuned-ade-512-512', 'processors':['cpu','gpu','edge']},
                      {'model':'nvidia/segformer-b1-finetuned-ade-512-512', 'processors':['cpu','gpu','edge']}],
		     2:
                      [{'model':'facebook/s2t-small-librispeech-asr', 'processors':['cpu','edge']},
                      {'model':'facebook/wav2vec2-base-960h', 'processors':['cpu','edge']}],
                     1:
                      [{'model':'roberta-base', 'processors':['cpu','edge']},
                      {'model':'distilbert-base-uncased', 'processors':['cpu','gpu','edge']}],
                     0:
                      [{'model':'efficientnet-b0', 'processors':['cpu','gpu','edge']},
                      {'model':'efficientnet-b4', 'processors':['cpu','gpu','edge']}]}
'''
def get_actions(num_type_tasks,keys):
    act_set = set()
    for key in TASK_MODEL_PROCESSOR:
        if key in keys:
            type_task = key
            for tuple in TASK_MODEL_PROCESSOR[key]:
                model = tuple['model']
                for p in tuple['processors']:
                    act_set.add((str(type_task), str(model), str(p)))
    act_list=list(act_set)
    cnt_type=np.zeros(num_type_tasks)
    act_type=[]
    #Sort the types of tasks
    for idx,e in enumerate(keys):
        cnt_type[idx] = len([element for element in act_list if int(element[0])==e])
        act_type.append([element for element in act_list if int(element[0]) == e])
    tot_actions = int(np.prod(cnt_type))
    actions = np.zeros(shape=(tot_actions,num_type_tasks),dtype=object)
    #Create the tuples
    j=0; it=0;
    one_type=[element for element in act_list if int(element[0])==keys[0]]
    for i in range(tot_actions):
        if it < cnt_type[j]:
            actions[i,j] = (one_type[it][1],one_type[it][2])
            it+=1
        else:
            it=0
            actions[i, j] = (one_type[it][1],one_type[it][2])
    i=0
    for j in range(1,(len(keys))):
        one_type = [element for element in act_list if int(element[0]) == keys[j]]
        it=0; start=0
        while i < tot_actions:
            if it < cnt_type[j]:
                actions[i, j] = (one_type[it][1], one_type[it][2])
                it+=1
                i+=1
            elif start>0:
                it=0
                while it < start and i< tot_actions and it<cnt_type[j]:
                    actions[i, j] = (one_type[it][1], one_type[it][2])
                    i+=1
                    it+=1
                start+=1
                it=start
            else:
                start+=1
                it=start
    return actions
'''


def aux_actions(a, pos, idx):
    j=0
    for i in range(len(a)):
        if j < len(pos[idx]):
            a[i,idx]= pos[idx][j]
            j+=1
        else:
            j=0
            a[i, idx] = pos[idx][j]
            j+=1
    return a

def get_next(j,cnt,i,s):
    if i%s==0:
        j=cnt
    else:
        j=(cnt+(i%s))%s
    print("J",j, "CNT", cnt, "i", i)
    return j

# only works with 2 tasks
def get_actions(num_type_tasks,keys):
    task_model=[]
    for key in TASK_MODEL_PROCESSOR:
        if key in keys:
            type_task = key
            for tuple in TASK_MODEL_PROCESSOR[key]:
                model = tuple['model']
                for p in tuple['processors']:
                    task_model.append((str(type_task), str(model), str(p)))
    pos = np.empty((num_type_tasks,), object)
    pos = [[] for _ in range(num_type_tasks)]
    i=0
    for key in keys:
        for e in task_model:
            if int(e[0]) == key:
                pos[i].append((e[1],e[2]))
        i+=1
    a=np.empty((np.prod([len(e) for e in pos]),len(pos)),dtype=object)
    a=aux_actions(a,pos,0)
    j=0; idx=1;cnt=-1
    for i in range(len(a)):
        if i % len(pos[idx]) == 0:
            cnt = cnt + 1
        if i>= len(pos[idx]):
            j=get_next(j,cnt,i,len(pos[idx]))
            a[i, idx] = pos[idx][j]
        else:
            a[i, idx] = pos[idx][j]
            j+=1
    final=[]
    for i in range(len(a)):
        final.append((a[i,0],a[i,1]))
    return final


# actions for the two models, three types of tasks, three possible processors
# Tasks: 0-image, 1-text, 2-audio
#num_type_tasks = 3
#THREE_TASKS = get_actions(num_type_tasks,[0,1,3])

# actions for the two models, two types of tasks, three possible processors
# 0-image, 1-text
num_type_tasks = 2
TWO_TASKS_0 = get_actions(num_type_tasks,[0,3])

if(len(set(TWO_TASKS_0)) == len(TWO_TASKS_0)):
    print("all elements are unique")

TWO_TASKS_1 = get_actions(num_type_tasks,[0,3])

TWO_TASKS = get_actions(num_type_tasks,[0,3])

for idx,element in enumerate(TWO_TASKS_1):
    if element !=TWO_TASKS_0[idx]:
        print(element, "IS DIFFERENT FROM ", TWO_TASKS_0[idx])

#for idx, element in enumerate(TWO_TASKS_1):
#    print(idx, " ", element )

#print("TWO TASKS")
#print(TWO_TASKS)



