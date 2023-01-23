
import logging
import threading
import time
import psutil
import os
import argparse
import numpy as np
from jtop import jtop,JtopException
from typing import List, Dict
import json
import csv
import subprocess
from multiprocessing import Process


class Log_device():
	def __init__(self,delay):
		self.delay = delay
		pass

	def set_delay(delay):
		self.delay=delay

	def try_again_get_com_fts(self,counter):
		start=time.time()
		cpu_load=0; gpu_load=0; mem=0; swap=0; curr=0;
		try:
			while time.time() < start+self.delay:
				with jtop() as jetson:
					while jetson.ok():
						tmp= jetson.stats
						influx_json= {"jetson": tmp}
						if tmp['CPU1'] != 'OFF': cpu_load += tmp['CPU1']/2
						if tmp['CPU2'] != 'OFF': cpu_load += tmp['CPU2']/2
						if tmp['CPU3'] != 'OFF': cpu_load += tmp['CPU3']/2
						if tmp['CPU4'] != 'OFF': cpu_load += tmp['CPU4']/2
						gpu_load += tmp['GPU']
						mem += (tmp['RAM']/4100000)*100
						swap += (tmp['SWAP']/2000000)*100
						curr += (tmp['power avg'])
						tmp_cpu = (tmp['Temp CPU'])
						tmp_gpu = (tmp['Temp GPU'])
			
						return {'CPU_LOAD':cpu_load,'GPU_LOAD':gpu_load,'MEM':mem,'SWAP':swap,'CURR':curr, 'TEMP_CPU':tmp_cpu, 'TEMP_GPU': tmp_gpu}
		except:
			if counter == 0:
				return {'CPU_LOAD':4,'GPU_LOAD':1,'MEM':1,'SWAP':1,'CURR':3600, 'TEMP_CPU':44, 'TEMP_GPU': 44}
			self.try_again_get_com_fts(counter-1)

	def get_com_fts(self):
		start=time.time()
		cpu_load=0; gpu_load=0; mem=0; swap=0; curr=0;
		try:
			while time.time() < start+self.delay:
				with jtop() as jetson:
					while jetson.ok():
						tmp= jetson.stats
						influx_json= {"jetson": tmp}
						if tmp['CPU1'] != 'OFF': cpu_load += tmp['CPU1']/2
						if tmp['CPU2'] != 'OFF': cpu_load += tmp['CPU2']/2
						if tmp['CPU3'] != 'OFF': cpu_load += tmp['CPU3']/2
						if tmp['CPU4'] != 'OFF': cpu_load += tmp['CPU4']/2
						gpu_load += tmp['GPU']
						mem += (tmp['RAM']/4100000)*100
						swap += (tmp['SWAP']/2000000)*100
						curr += (tmp['power avg'])
						tmp_cpu = (tmp['Temp CPU'])
						tmp_gpu = (tmp['Temp GPU'])
			
						return {'CPU_LOAD':cpu_load,'GPU_LOAD':gpu_load,'MEM':mem,'SWAP':swap,'CURR':curr, 'TEMP_CPU':tmp_cpu, 'TEMP_GPU': tmp_gpu}
		except:
			self.try_again_get_com_fts(3)
			
	def start_log(self,filename):
		with jtop() as jetson:
			with open(filename, 'w') as csvfile:
				fieldnames = ['TIME', 'CPU_LOAD','GPU_LOAD','MEM','SWAP','CURR','TEMP_CPU','TEMP_GPU']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writeheader()
				print(jetson.ok())
				while jetson.ok():
					tmp= jetson.stats
					influx_json= {"jetson": tmp}
					cpu_load = sum([tmp['CPU1'], tmp['CPU2'], tmp['CPU3'], tmp['CPU4']])/4
					gpu_load = tmp['GPU']
					mem= (tmp['RAM']/4100000)*100
					swap = (tmp['SWAP']/2000000)*100
					curr = (tmp['power avg'])
					tmp_cpu = (tmp['Temp CPU'])
					tmp_gpu = (tmp['Temp GPU'])
					sol = {'TIME':time.time(),'CPU_LOAD':cpu_load,'GPU_LOAD':gpu_load,'MEM':mem,'SWAP':swap,
					'CURR':curr,'TEMP_CPU':tmp_cpu, 'TEMP_GPU': tmp_gpu}
					writer.writerow(sol)
					time.sleep(self.delay)
	
	def start_log_net(self, filename):
		with open(filename, 'w') as csvfile:
			fieldnames = ['TIME','IP_TOT_RECV_PKTS','IP_FWD','IP_UNK', 'IP_IN_PKTS_DISC',
				'IP_IN_PKTS_DEL','IP_REQ_SENT','IP_OUT_PKTS_DROP','TCP_ACT_CONN','TCP_PASS_CONN',
				'TCP_FAIL_CONN_ATTMP','TCP_CONN_RES_RECV',
				'TCP_CONN_ESTAB','TCP_SEGM_RECV','TCP_SEGM_SENT','TCP_SEGM_RETANS',
				'TCP_BAD_SEGM_RECV','TCP_REST_SENT', 'WLAN0_RX_BYTES',
				'WLAN0_RX_PKTS', 'WLAN0_RX_ERR','WLAN0_RX_DROP', 'WLAN0_RX_OVERRUN',
				'WLAN0_RX_MCAST','WLAN0_TX_BYTES','WLAN0_TX_PKTS','WLAN0_TX_ERR',
				'WLAN0_TX_DROP','WLAN0_TX_OVERRUN','WLAN0_TX_MCAST']

			writer = csv.DictWriter(csvfile, fieldnames)
			writer.writeheader()
			while(True):	
				cmd=['netstat','-sw'] 
				tcp_stats = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
				out_1=[int(tcp_stats[idx].strip().split()[0]) for idx in range(2,8)]

				cmd=['netstat','-st'] 
				tcp_stats = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
				out_3=[int(tcp_stats[idx].strip().split()[0]) for idx in range(6,13)]
				
				cmd=['ip', '-s', 'link', 'show', 'wlan0']
				wlan0_stats= subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
				out_4=[int(wlan0_stats.decode('utf-8').strip().split()[idx]) for idx in range(26,32)]
				out_5=[int(wlan0_stats.decode('utf-8').strip().split()[idx]) for idx in range(39,45)]
				final=[time.time()]+out_1+out_3+out_4+out_5
				res_dct = {fieldnames[i]: final[i] for i in range(0, len(final))}
				writer.writerow(res_dct)
				time.sleep(self.delay)



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d")
	args = parser.parse_args()
	delay = float(args.d)
	logger=Log_device(delay)

	sol=[]
	try:
		p1 = Process(target=logger.start_log, args=('logs_cpu_gpu.csv',))
		p1.start()
		p2 = Process(target=logger.start_log_net, args=('logs_net.csv',))
		p2.start()
		p1.join()
		p2.join()

	except JtopException as e:
		print(e)
	except KeyboardInterrupt:
		print("Closed with CTRL-C")
	except IOError:
		print("I/O error")
