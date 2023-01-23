import Dispatcher
import TaskGen
import torch.multiprocessing as mp
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertConfig
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, RobertaConfig
from transformers import SegformerFeatureExtractor, SegformerForImageClassification,SegformerForSemanticSegmentation
from efficientnet_pytorch import EfficientNet
import time
import pickle

#Parameters
#freq_0=['180','80','40','16']
#freq_3=['20','20','10','4']
#cc=['2','4','8','20']
#tot_tasks=2
#processors=['cpu','gpu']
#device=['nano','xavier']
#res=['low','high']
#prob_0=['b0','b4']
#prob_3=['b0','b1']


def main(freq_0,freq_3,cc,tot_tasks,processors,device,res,prob_0,prob_3):
	# When there is a task, dispatch it on cpu
	# Initialize dispatcher
	dispatcher = Dispatcher.Dispatcher()

	# Initialize task generator
	tg = TaskGen.TaskGen()
	input_q = mp.Queue()
	delay=3

	# Start task generator
	p_tasks = mp.Process(target=tg.start_test_0_3, args=(input_q,int(freq_0),int(freq_3),tot_tasks,delay,))
	p_tasks.start()

	k = 0
	stats = mp.Queue()
	tot_delays = {}
	enq_delays = {}
	load_inf_delays = {}
	timeline = {}
	dist = {}
	timestamp = {}
	while k<tot_tasks:
			if (input_q.qsize() > 0):
					task = input_q.get()
					if task.get_type() == 3:
							try:
									#dispatcher.remove_model('distilbert-base-uncased'+'-'+processors[m]+'-1')
									dispatcher.remove_model('efficientnet-'+prob_0+ '-' + processors + '-0')
									#print("segformer-b0-uncased model removed ")
							except:
									pass
							model='nvidia/segformer-'+prob_3+'-finetuned-ade-512-512'
							type_task=3
							start = time.time()
							if not dispatcher.is_loaded(model, processors, type_task):
									dispatcher.load_model_in_processor(model, processors , type_task)
									print("Image loaded type 3")
							dispatcher.execute_task_laptop([task], model, processors, k, stats)
							end = time.time()
							tot_delays[k] = (end - task.get_timestamp())
							load_inf_delays[k] = (end-start)
							enq_delays[k] = (start- task.get_timestamp())
							timeline[k] = time.time()
							timestamp[k] = task.get_timestamp()
							dist[k] = 3
							k += 1

					elif task.get_type() == 0:
							try:
									#dispatcher.remove_model('distilbert-base-uncased'+'-'+processor+'-1')
									dispatcher.remove_model('nvidia/segformer-'+prob_3+'-finetuned-ade-512-512' + '-' + processors + '-3')
									#print(" distilbert-base-uncased model removed ")
							except:
									pass
							model= 'efficientnet-'+prob_0
							type_task=0
							start = time.time()
							if not dispatcher.is_loaded(model, processors, type_task):
									dispatcher.load_model_in_processor(model, processors , type_task)
									#print("Image loaded efficienet-b4")
							dispatcher.execute_task_laptop([task], model, processors, k, stats)
							end = time.time()
							tot_delays[k] = (end - task.get_timestamp())
							load_inf_delays[k] = (end-start)
							enq_delays[k] = (start- task.get_timestamp())
							timeline[k] = time.time()
							timestamp[k] = task.get_timestamp()
							dist[k] = 0
							k += 1

					elif task.get_type() == 1:
							try:
									dispatcher.remove_model('nvidia/segformer-'+prob_3+'-finetuned-ade-512-512'+'-'+processors+'-3')
									#print("nvidia/segformer-b0-finetuned-ade-512-512 model removed ")
							except:
									pass
							model='distilbert-base-uncased'
							type_task=1
							start = time.time()
							if not dispatcher.is_loaded(model, processors, type_task):
									dispatcher.load_model_in_processor(model, processors, type_task)
									#print("Text loaded")
							dispatcher.execute_task_laptop([task], model, processors, k, stats)
							end = time.time()
							tot_delays[k] = (end - task.get_timestamp())
							load_inf_delays[k] = (end-start)
							enq_delays[k] = (start- task.get_timestamp())
							timeline[k] = time.time()
							timestamp[k] = task.get_timestamp()
							dist[k] = 1
							k += 1

					else:
							print("not recognized")

			else:
					pass
					#print("waiting ... ")

	with open('./results/'+device+'_'+processors+'_'+res+'_res_tot_delays_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(tot_delays, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./results/'+device+'_'+processors+'_'+res+'_res_enq_delays_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(enq_delays, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./results/'+device+'_'+processors+'_'+res+'_res_load_inf_delays_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(load_inf_delays, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./results/'+device+'_'+processors+'_'+res+'_res_timeline_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(timeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./results/'+device+'_'+processors+'_'+res+'_res_dist_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./results/'+device+'_'+processors+'_'+res+'_res_timestamp_'+prob_0+'_'+cc+'.pk', 'wb') as handle:
			pickle.dump(timestamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
	time.sleep(delay)
	print("DONE!!!")
	p_tasks.terminate()
	print("DONE tasks process!!!")
	p_tasks.join()
	print("JOIN task process!!!")
	input_q.close()
	print("Stats close!!!")
	stats.close()
	print("del dispatcher!!!")
	del dispatcher
	del tg
	del p_tasks
	del input_q

if __name__ == "__main__":
	freq_0='40'
	freq_3='10'
	cc='8' 
	tot_tasks=200
	processors='cpu'
	device='xavier'
	res='low'
	prob_0='b0'
	prob_3='b0'

	main(freq_0,freq_3,cc,tot_tasks,processors,device,res,prob_0,prob_3)
