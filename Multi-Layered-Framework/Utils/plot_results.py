import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
processor = 'cpu'
big_proc = 'CPU'
device = 'jetson'
freq = '100'
prob = 'b4'

obj_dist = pd.read_pickle('./' + device +'_'+ processor + '_low_res_dist_' + prob + '_' + freq + '.pk')
obj_tot_del = pd.read_pickle('./' + device+ '_'+processor + '_low_res_tot_delays_' + prob + '_' + freq + '.pk')
obj_enq_del = pd.read_pickle('./' +device+'_'+ processor + '_low_res_enq_delays_' + prob + '_' + freq + '.pk')
obj_inf_del = pd.read_pickle('./' +device+'_'+ processor + '_low_res_load_inf_delays_' + prob + '_' + freq + '.pk')
obj_tim = pd.read_pickle('./' +device+'_'+ processor + '_low_res_timeline_' + prob + '_' + freq + '.pk')
obj_timestamp = pd.read_pickle('./' +device+'_'+ processor + '_low_res_timestamp_' + prob + '_' + freq + '.pk')

tot_delays = [obj_tot_del[key] for key in obj_tot_del]
tot_enq_del = [obj_enq_del[key] for key in obj_enq_del]
tot_load_inf_del = [obj_inf_del[key] for key in obj_inf_del]
timeline = [obj_tim[key] - obj_tim[0] for key in obj_tim]
distrib = [obj_dist[key] for key in obj_dist]
timestamp = [obj_timestamp[key] - obj_timestamp[0] for key in obj_timestamp]

hist, bin_edges = np.histogram(timestamp, bins=10)
print(bin_edges)

distrib_0 = np.zeros((11,))
distrib_1 = np.zeros((11,))

changed = False

avg_delays_0 = []
avg_delays_1 = []

indexes_0 = []
indexes_1 = []

delays_0 = 0
delays_1 = 0

index_0 = []
index_1 = []

for idx, el in enumerate(distrib):
    if idx == 0:
        pass
    if idx + 1 < len(distrib):
        if distrib[idx] != distrib[idx+1]:
            if distrib[idx] == 0:
                index_0.append(idx)
                delays_0 += tot_delays[idx]
                avg_delays_0.append(delays_0/len(index_0))
                indexes_0.append(index_0)
                index_0=[]
                delays_0=0
            elif distrib[idx] == 3:
                index_1.append(idx)
                delays_1 += tot_delays[idx]
                avg_delays_1.append(delays_1/len(index_1))
                indexes_1.append(index_1)
                delays_1=0
                index_1=[]
        else:
            if distrib[idx] == 0:
                delays_0 += tot_delays[idx]
                index_0.append(idx)
            elif distrib[idx] == 3:
                delays_1 += tot_delays[idx]
                index_1.append(idx)
    else:
        if distrib[idx] == 0:
            delays_0 += tot_delays[idx]
            index_0.append(idx)
            avg_delays_0.append(delays_0/len(index_0))
            indexes_0.append(index_0)

        elif distrib[idx] == 3:
            delays_1 += tot_delays[idx]
            index_1.append(idx)
            avg_delays_1.append(delays_1/len(index_1))
            indexes_1.append(index_1)

delays_0 = np.empty((len(timestamp),))
delays_0[:] = np.NaN

delays_1 = np.empty((len(timestamp),))
delays_1[:] = np.NaN

j=0;idx=0
for j in range(len(indexes_0)):
    idx = indexes_0[j][0]
    while idx in indexes_0[j]:
        delays_0[idx] = avg_delays_0[j]
        idx+=1

j=0;idx=0
for j in range(len(indexes_1)):
    idx = indexes_1[j][0]
    while idx in indexes_1[j]:
        delays_1[idx] = avg_delays_1[j]
        idx+=1

k = 0
for idx, element in enumerate(distrib):
    if float(bin_edges[k]) <= float(timestamp[idx]) <= float(bin_edges[k + 1]):
        if element == 0:
            distrib_0[k] += 1
        else:
            distrib_1[k] += 1
    else:
        k += 1

# print(hist)
# hist_arr=[np.array(hist[k],bin_edges[k]) for k in range(len(hist)) if hist[k]>0]

print("total size: ", len(tot_delays))

plt.figure(1)
plt.plot(tot_delays)
plt.yscale('log')
plt.xlabel("Occurencies (#)")
plt.ylabel("Total delay (s)")
plt.title("Total delay per task sent - " + big_proc)
plt.savefig('./' + device + '_tot_delay_' + processor + '_' + freq + '_' + prob + '.jpg')

plt.figure(2)
plt.plot(tot_enq_del)
plt.yscale('log')
plt.xlabel("Occurencies (#)")
plt.ylabel("Total delay enqueued (s)")
plt.title("Total delay before loading and inference - " + big_proc)
plt.savefig('./' + device + '_enqueued_' + processor + '_' + freq + '_' + prob + '.jpg')

plt.figure(3)
plt.plot(timeline, tot_load_inf_del)
plt.yscale('log')
plt.xlabel("Recorded Time (s)")
plt.ylabel("Total loading and inference delay (s)")
plt.title("Total loading and inference delay - " + big_proc)
plt.savefig('./' + device + '_load_inf_delay_' + processor + '_' + freq + '_' + prob + '.jpg')

plt.figure(4)
barsize = 10
plt.bar(bin_edges - barsize, height=list(distrib_0), width=barsize, color='blue',
        label=' Image Classification / Efficienet-b4')
plt.bar(bin_edges + barsize, height=list(distrib_1), width=barsize, color='red',
        label='Image Segmentation / Segformer-b0')
plt.xlabel("Recorded Time (s)")
plt.ylabel("Type of task count (#)")
plt.title("Tasks distribution - " + big_proc)
plt.legend()
plt.savefig('./' + device + '_dist_' + processor + '_' + freq + '_' + prob + '.jpg')

plt.figure(5)
# barsize=1
# hist, bin_edges = np.histogram(timestamp, bins=99)
# plt.bar(bin_edges-barsize,height=list(delays_0), width=barsize,  color='blue')
# plt.bar(bin_edges+barsize,height=list(delays_1), width=barsize,  color='red')
plt.plot(timestamp, delays_0, color='blue', label='Image Classification / Efficienet-b4')
plt.plot(timestamp, delays_1, color='red', label='Image Segmentation / Segformer-b0')
# plt.yscale('log')
plt.xlabel("Occurencies (#)")
plt.ylabel("Average delay (s)")
plt.title("Total delay per task sent - " + big_proc)
#plt.ylim([0.0, 1.5])
plt.legend()
plt.savefig('./' + device + '_tot_delay_div_' + processor + '_' + freq + '_' + prob + '.jpg')
'''
plt.figure(6)
#list_freq=[100,30,20,10,8,4,2]
list_freq=[4,6,8,14,20]
contx_sw=[1,3,5,10,13,25,50]
processors = ['gpu','cpu']
prob=['b0','b4']
res=['low','high']
big_proc = 'GPU'
type_of_task = [0,3]
delays_gpu = np.empty((len(list_freq),len(type_of_task), len(res)),object)
delays_gpu = [[[[] for _ in range(len(res))] for _ in range(len(type_of_task))] for _ in range(len(list_freq))]
delays_cpu = np.empty((len(list_freq),len(type_of_task), len(res)),object)
delays_cpu = [[[[] for _ in range(len(res))] for _ in range(len(type_of_task))] for _ in range(len(list_freq))]
avg_delays_gpu = np.zeros((len(list_freq),len(type_of_task), len(res)))
avg_delays_cpu = np.zeros((len(list_freq),len(type_of_task), len(res)))
avg_delays_gpu_std = np.zeros((len(list_freq),len(type_of_task), len(res)))
avg_delays_cpu_std = np.zeros((len(list_freq),len(type_of_task), len(res)))
for processor in processors:
    for i,freq in enumerate(list_freq):
        device = 'nano'
        #freq = str(freq)
        for k in range(len(prob)):
            #path='/Users/sharon/Documents/Documents/DatasetMultiplatform/'+device+'/'
            path = '/home/sharon/Documents/Research/DatasetMultiplatform/' + device + '/'
            obj_dist = pd.read_pickle(path+ device + '_' + processor + '_'+res[k]+'_res_dist_' + prob[k] + '_' + str(freq) + '.pk')
            obj_tot_del = pd.read_pickle(
                path + device + '_' + processor + '_'+res[k]+'_res_tot_delays_' + prob[k] + '_' + str(freq) + '.pk')
            obj_enq_del = pd.read_pickle(
                path + device + '_' + processor + '_'+res[k]+'_res_enq_delays_' + prob[k] + '_' + str(freq) + '.pk')
            obj_inf_del = pd.read_pickle(
                path + device + '_' + processor + '_'+res[k]+'_res_load_inf_delays_' + prob[k] + '_' + str(freq) + '.pk')
            obj_tim = pd.read_pickle(path + device + '_' + processor + '_'+res[k]+'_res_timeline_' + prob[k] + '_' + str(freq) + '.pk')
            obj_timestamp = pd.read_pickle(
                path + device + '_' + processor + '_'+res[k]+'_res_timestamp_' + prob[k] + '_' + str(freq) + '.pk')

            tot_delays = [obj_tot_del[key] for key in obj_tot_del]
            tot_enq_del = [obj_enq_del[key] for key in obj_enq_del]
            tot_load_inf_del = [obj_inf_del[key] for key in obj_inf_del]
            timeline = [obj_tim[key] - obj_tim[0] for key in obj_tim]
            distrib = [obj_dist[key] for key in obj_dist]
            timestamp = [obj_timestamp[key] - obj_timestamp[0] for key in obj_timestamp]

            if processor == 'gpu':
                #tot_delays[0]=tot_delays[0]-4
                '''
                for idx,element in enumerate(type_of_task):
                    tot_del = [];
                    for task in  distrib:
                        if task == element:
                            tot_del.append(tot_load_inf_del[idx])
                    delays_gpu[i][idx]=np.mean(tot_del)/contx_sw[i]
                    print("tot_del ", len(tot_del))
                '''
                for idx,element in enumerate(type_of_task):
                    if element == 0:
                        start = [1]
                    else:
                        start = [];
                    end = [d for d in range(len(distrib) - 1) if (distrib[d] != distrib[d + 1] and distrib[d] == element)]
                    [start.append(d + 1) for d in range(len(distrib) - 1) if
                     (distrib[d] != distrib[d + 1] and distrib[d + 1] == element)]
                    if distrib[len(distrib)-1] == element : end.append(len(distrib)-1)
                    if len(end) == 0:
                        delays_gpu[i][idx][k].append(([tot_load_inf_del]))
                        #delays_gpu[i][idx].append(tot_load_inf_del[1])
                    else:
                        delays_gpu[i][idx][k].append([(tot_load_inf_del[start[j]:end[j]]) for j in range(len(end))])
                        #np.mean(np.array(tot_delays))
                        #delays_gpu[i][idx].append([(tot_load_inf_del[start[j]]) for j in range(len(start))])
            else:
                for idx, element in enumerate(type_of_task):
                    if element == 0:
                        start = [1]
                    else:
                        start = [];
                    end = [d for d in range(len(distrib) - 1) if (distrib[d] != distrib[d + 1] and distrib[d] == element)]
                    [start.append(d + 1) for d in range(len(distrib) - 1) if
                     (distrib[d] != distrib[d + 1] and distrib[d + 1] == element)]
                    if distrib[len(distrib)-1] == element : end.append(len(distrib)-1)
                    if len(end) == 0:
                        delays_cpu[i][idx][k].append(([tot_load_inf_del]))
                    else:
                        delays_cpu[i][idx][k].append([(tot_load_inf_del[start[j]:end[j]]) for j in range(len(end))])

for i,_ in enumerate(list_freq):
    for idx, _ in enumerate(type_of_task):
        for jdx,_ in enumerate(prob):
            avg_delays_gpu_std[i][idx][jdx] = np.min(np.hstack(delays_gpu[i][idx][jdx][0]))
            avg_delays_gpu[i][idx][jdx] = np.mean(np.hstack(delays_gpu[i][idx][jdx][0]))
            avg_delays_cpu_std[i][idx][jdx] = np.min(np.hstack(delays_cpu[i][idx][jdx][0]))
            avg_delays_cpu[i][idx][jdx] = np.mean(np.hstack(delays_cpu[i][idx][jdx][0]))

x_axis = list_freq
plt.plot(x_axis,avg_delays_cpu[:,0,0],color='red' ,label='Image Classification  / CPU', marker='o')
plt.yscale('log')
plt.plot(x_axis,avg_delays_cpu[:,1,0],color='red' ,label='Image Segmentation  / CPU', marker='*')
plt.yscale('log')
plt.plot(x_axis,avg_delays_gpu[:,0,0],color='blue' ,label='Image Classification  / GPU', marker='o')
plt.yscale('log')
plt.plot(x_axis,avg_delays_gpu[:,1,0],color='blue' ,label='Image Segmentation  / GPU', marker='*')
plt.yscale('log')
plt.title("Average delay vs Context switching - "+ (device))
plt.xlabel("Frequency of context switching")
plt.ylabel("Average delay (s)")
plt.tight_layout()
plt.legend()
plt.savefig('./' + device +'_'+res[0]+ '_final_results.jpg')
print("delays gpu \n", avg_delays_gpu)
print("delays cpu \n", avg_delays_cpu)
#plt.xlim(100,0)

plt.figure(7)
#plt.plot(x_axis,delays_cpu[:,0],color='red' ,label='Image Classification  / CPU', marker='o')
plt.plot(x_axis,avg_delays_cpu[:,1,0],color='red' ,label='Image Segmentation  / CPU', marker='*')
#plt.yscale('log')
#plt.plot(x_axis,delays_gpu[:,0],color='blue' ,label='Image Classification  / GPU', marker='o')
plt.plot(x_axis,avg_delays_gpu[:,1,0],color='blue' ,label='Image Segmentation  / GPU', marker='*')
#plt.yscale('log')
plt.title("Image Segmentation - Average delay vs Context switching - "+ (device))
plt.xlabel("Frequency")
plt.ylabel("Average delay (s)")
plt.tight_layout()
plt.legend()
plt.savefig('./' + device +'_' +res[0]+ '_image_segmentation'+'_final_results.jpg')

plt.figure(8)
plt.plot(x_axis,avg_delays_cpu[:,0,0],color='red' ,label='Image Classification  / CPU', marker='o')
#plt.yscale('log')
#plt.plot(x_axis,delays_cpu[1:6,1],color='red' ,label='Image Segmentation  / CPU', marker='*')
plt.plot(x_axis,avg_delays_gpu[:,0,0],color='blue' ,label='Image Classification  / GPU', marker='o')
#plt.yscale('log')
#plt.plot(x_axis,delays_gpu[1:6,1],color='blue' ,label='Image Segmentation  / GPU', marker='*')
plt.title("Image Classification - Average delay vs Context switching - "+ (device))
plt.xlabel("Frequency")
plt.ylabel("Average delay (s)")
plt.tight_layout()
plt.legend()
plt.savefig('./' + device +'_' +res[0]+ '_image_class'+'_final_results.jpg')

plt.figure(9)
plt.plot(x_axis,avg_delays_gpu[:,0,0],color='firebrick' ,label='Image Classification' , marker='o',linewidth=0.7)
plt.plot(x_axis,avg_delays_gpu[:,1,0],color='navy' ,label='Image Segmentation', marker='*', linewidth=0.7)
plt.title(" Context switching in GPU - "+ (device[0].upper()+device[1:]))
plt.xlabel("Frequency")
plt.ylabel("Average delay (s)")
plt.tight_layout()
plt.legend()
plt.savefig('./' + device +'_' +res[0]+ '_gpu_'+'_final_results.jpg')

plt.figure(10)
plt.plot(x_axis,avg_delays_cpu[:,0,0],color='firebrick' ,label='Image Classification', marker='o',linewidth=0.7)
plt.plot(x_axis,avg_delays_cpu[:,1,0],color='navy' ,label='Image Segmentation', marker='*', linewidth=0.7)
plt.title(" Context switching in CPU - "+ (device[0].upper()+device[1:]),fontsize=16)
plt.xlabel("Frequency")
plt.ylabel("Average delay (s)")
plt.tight_layout()
plt.legend()
plt.savefig('./' + device +'_' +res[0]+ '_cpu_'+'_final_results.jpg')

device_name = 'JN'
fig=plt.figure(11)
ax= plt.subplot(111)
ax.plot(x_axis,avg_delays_cpu[:,0,0],color='firebrick' ,label='Image Classification / Low Accuracy', marker='o',linewidth=0.7,markersize=10)
ax.plot(x_axis,avg_delays_cpu[:,1,0],color='navy' ,label='Image Segmentation / Low Accuracy', marker='*', linewidth=0.7,markersize=10)
ax.plot(x_axis,avg_delays_cpu[:,0,1],color='darkorange' ,label='Image Classification / High Accuracy', marker='^',linewidth=0.7,markersize=10)
ax.plot(x_axis,avg_delays_cpu[:,1,1],color='darkgreen' ,label='Image Segmentation / High Accuracy', marker='+', linewidth=0.7,markersize=10)
plt.title(" Context-Switching in CPU - "+ (device_name), fontsize=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.20),shadow=False, fontsize=14)
plt.xlabel("Frequency",fontsize=16)
plt.ylabel("Average delay (s)",fontsize=16)
plt.tight_layout()
plt.savefig('./' + device +'_both_cpu_'+'_final_results.jpg')

fig=plt.figure(12)
ax = plt.subplot(111)
ax.plot(x_axis,avg_delays_gpu[:,0,0],color='firebrick' ,label='Image Classification / Low Accuracy', marker='o',linewidth=0.7,markersize=10)
#plt.fill_between(x_axis,avg_delays_gpu[:,0,0]+avg_delays_gpu_std[:,0,0],avg_delays_gpu[:,0,0]-avg_delays_gpu_std[:,0,0],
#                 facecolor='firebrick',alpha=0.1)
ax.plot(x_axis,avg_delays_gpu[:,1,0],color='navy' ,label='Image Segmentation / Low Accuracy', marker='*', linewidth=0.7,markersize=10)
#plt.fill_between(x_axis,avg_delays_gpu[:,1,0]+avg_delays_gpu_std[:,1,0],avg_delays_gpu[:,1,0]-avg_delays_gpu_std[:,1,0],
#                 facecolor='navy',alpha=0.1)
ax.plot(x_axis,avg_delays_gpu[:,0,1],color='darkorange' ,label='Image Classification / High Accuracy', marker='^',linewidth=0.7,markersize=10)
#plt.fill_between(x_axis,avg_delays_gpu[:,0,1]+avg_delays_gpu_std[:,0,1],avg_delays_gpu[:,0,1]-avg_delays_gpu_std[:,0,1],
#                 facecolor='darkorange',alpha=0.1)
ax.plot(x_axis,avg_delays_gpu[:,1,1],color='darkgreen' ,label='Image Segmentation / High Accuracy', marker='+', linewidth=0.7,markersize=10)
#plt.fill_between(x_axis,avg_delays_gpu[:,1,1]+avg_delays_gpu_std[:,1,1],avg_delays_gpu[:,1,1]-avg_delays_gpu_std[:,1,1],
#                 facecolor='darkgreen',alpha=0.1)
plt.title(" Context-Switching in GPU - "+ (device_name), fontsize=16)
plt.xlabel("Frequency",fontsize=16)
plt.ylabel("Average delay (s)",fontsize=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.20),shadow=False, fontsize=14)
#ax.legend(loc='best')
plt.tight_layout()
#plt.legend()
plt.savefig('./' + device +'_both_gpu_'+'_final_results.jpg')



plt.show()
