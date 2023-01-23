import logging
import threading
import time
import psutil
import os
import argparse
import numpy as np
import pynvml as nvidia_smi
import csv
import pingparsing
from multiprocessing import Process
import subprocess

class Log_com_laptop():
	def __init__(self,delay):
		self.delay=delay
		pass

	def set_delay(self,delay):
		self.delay=delay

	def get_status(self):
		mem = []
		load = []

		x = threading.Thread(target=self.thread_function, args=(1,self.delay,mem,load))
		y=time.time()
		x.start()
		x.join()

		mem = np.array(mem)
		load = np.array(load)

		if self.processor == 'cpu':
			sol = [100-load[load.nonzero()].mean(),mem[mem.nonzero()].mean()*100]
		elif self.processor == 'gpu':
			sol = [load.mean(),mem[mem.nonzero()].mean()]

		return sol

	def start_log(self,filename):
		with open(filename, 'w') as csvfile:
			fieldnames = ['time', 'cpu_load','gpu_load','cpu_mem', 'gpu_mem','']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			while True:
				cpu_load = psutil.cpu_percent(interval=None)
				mem_stats = psutil.virtual_memory()
				cpu_mem =  mem_stats.available/mem_stats.total

				nvidia_smi.nvmlInit()
				handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
				res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
				mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
				gpu_load = res.gpu
				gpu_mem = 100 * (mem_res.used / mem_res.total)

				sol = {'time':time.time(),'cpu_load':cpu_load,'gpu_load':gpu_load,'cpu_mem':cpu_mem,'gpu_mem':gpu_mem}
				writer.writerow(sol)
				time.sleep(self.delay)


class Log_net_laptop:
	"examples: IP_outgoing_pkts_dropped	TCP_num_fast_retransmission	TCP_segments_received	TCP_segments_retransmitted	TCP_segments_sent	Total_IP_pkts_delivered	Total_IP_pkts_received	Total_bytes_received_at_IP	Total_bytes_sent_at_IP	UDP_pkts_received	UDP_pkts_sent	beacon signal avg	expected throughput (Mbps)	rx bitrate	signal signal avg tx bitrate	tx failed	tx retries"
	def __init__(self,count=5,destination="128.195.54.86",delay=3):
		process = subprocess.Popen(['netstat', '-s'],stdout = subprocess.PIPE,stderr = subprocess.PIPE)
		stdout, stderr = process.communicate()
		for idx, element in enumerate(stdout.decode("utf-8").split('\n')): print("id: ", idx, element.strip())

		self.delay=delay
		self.result = self.transmitter.ping()
		self.dict_result = self.ping_parser.parse(self.result).as_dict()

	def start_log(self, filename):
		with open(filename, 'w') as csvfile:
			fieldnames = ['TIME', 'IP_TOT_RECV_PKTS', 'IP_FWD', 'IP_UNK', 'IP_IN_PKTS_DISC',
						  'IP_IN_PKTS_DEL', 'IP_REQ_SENT', 'IP_OUT_PKTS_DROP', 'IP_DROP_MISS_RT',
						  'TCP_ACT_CONN', 'TCP_PASS_CONN', 'TCP_FAIL_CONN_ATTMP', 'TCP_CONN_RES_RECV',
						  'TCP_CONN_ESTAB', 'TCP_SEGM_RECV', 'TCP_SEGM_SENT', 'TCP_SEGM_RETANS',
						  'TCP_BAD_SEGM_RECV', 'TCP_REST_SENT', 'WLAN0_RX_BYTES',
						  'WLAN0_RX_PKTS', 'WLAN0_RX_ERR', 'WLAN0_RX_DROP', 'WLAN0_RX_OVERRUN',
						  'WLAN0_RX_MCAST', 'WLAN0_TX_BYTES', 'WLAN0_TX_PKTS', 'WLAN0_TX_ERR',
						  'WLAN0_TX_DROP', 'WLAN0_TX_OVERRUN', 'WLAN0_TX_MCAST']

			writer = csv.DictWriter(csvfile, fieldnames)
			writer.writeheader()
			while (True):
				cmd = ['netstat', '-sw']
				tcp_stats = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
				out_1 = [int(tcp_stats[idx].strip().split()[0]) for idx in range(2, 9)]

				cmd = ['netstat', '-st']
				tcp_stats = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
				out_3 = [int(tcp_stats[idx].strip().split()[0]) for idx in range(8, 14)]

				cmd = ['ip', '-s', 'link', 'show', 'wlan0']
				wlan0_stats = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
				out_4 = [int(wlan0_stats.decode('utf-8').strip().split()[idx]) for idx in range(26, 32)]
				out_5 = [int(wlan0_stats.decode('utf-8').strip().split()[idx]) for idx in range(39, 45)]
				final = [time.time()] + out_1 + out_3 + out_4 + out_5
				res_dct = {fieldnames[i]: final[i] for i in range(0, len(final))}
				writer.writerow(res_dct)
				time.sleep(self.delay)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d")
	#parser.add_argument("-p")
	args = parser.parse_args()
	
	#delay = float(args.d)
	#processor = str(args.p)
	logger_com=Log_com_laptop(delay=0.05)
	logger_net=Log_net_laptop()

	p1 = Process(target=logger_net.start_log, args=('../log/logs_net.csv',))
	#p2 = Process(target=logger_com.start_log, args=('../log/logs_com.csv',))
	p1.start()
	#p2.start()
	time.sleep(4)
	p1.join()
	#p2.join()

		