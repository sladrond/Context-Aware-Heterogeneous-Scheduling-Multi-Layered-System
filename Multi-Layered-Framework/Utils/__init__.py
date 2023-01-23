import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
#sys.path.append('/home/sharon/Documents/laptop_github/HeterogeneousTaskScheduler/')
#from ddqn_scheduler.actions import TWO_TASKS
from ddqn_scheduler.actions import TWO_TASKS_0

#sys.path.append('/Users/sharon/Documents/laptop_github/HeterogeneousTaskScheduler/')


class ExpsReader:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def convert_array(self, a):
        if a[0] == '[':
            b = np.zeros(len(a) - 2)  # Create a float array
            b[0] = b[0] + float(a[1]);
            b[len(b) - 1] = b[len(b) - 1] + float(a[len(a) - 2]);
            start = 2
        else:
            b = np.zeros(len(a))  # Create a float array
            b[0] = b[0] + float(a[0][1:]);
            b[len(b) - 1] = b[len(b) - 1] + float(a[len(a) - 1][0:-1]);
            start = 1

        for j in range(start, len(b) - 1):
            b[j] = b[j] + float(a[j])
        return b

    def get_exp(self, action, state):
        folders = os.listdir(self.path)
        for files in folders:
            #print(files)
            try:
                parser = csv.reader(open(os.path.join(self.path, files, self.filename)), delimiter=',')
                i = 0
               
                for l in parser:
                    if i > 0:
                        if i==2:
                            print(len(self.convert_array(np.array(l[0].split()))))
                        a = np.array(l[0].split())  # Split state line
                        b = self.convert_array(a)
                        #print(b)
                        if np.array_equal(b, state) or np.allclose(b, state, atol=1e-01):
                            if int(action) == int(l[1]):
                                c = self.convert_array(np.array(l[3].split()))
                                return c,float(l[2])
                    i += 1
                    #print("i: ",i)
                    
            except NotADirectoryError:
                print("Not a directory exception")
                return -1
        return -1

    def get_df(self, header):
        dfs=[]
        folders = os.listdir(self.path)
        try:
            for files in folders:
                c_df=pd.read_csv(os.path.join(self.path, files, self.filename))
                dfs.append(c_df)
            res=pd.concat(dfs)
            return res[header]
        except Exception as e:
            print(e)
            return None

    def get_stats(self,header,device):
        df = pd.DataFrame(columns=header)
        count=0
        folders = os.listdir(self.path)
        for files in folders:
            try:
                parser = csv.reader(open(os.path.join(self.path, files, self.filename)), delimiter=',')
                i = 0

                for l in parser:
                    if i > 0:
                        if i == 2:
                            print(len(self.convert_array(np.array(l[0].split()))))
                        if device != 'xavier':
                            a= np.array(l[3].split())
                            c = self.convert_array(a)
                            df.loc[count,header[0]] = float(c[12])
                            df.loc[count, header[1]] = float(c[4])
                            df.loc[count, header[2]] = float(l[1])
                            df.loc[count, header[3]] = float(c[10])
                            df.loc[count, header[4]] = float(c[11])
                        else:
                            a = np.array(l[4].split())
                            c = self.convert_array(a)
                            df.loc[count,header[0]] = float(c[12]) #12 if xavier
                            df.loc[count, header[1]] = float(c[4])
                            df.loc[count, header[2]] = float(c[5])
                            df.loc[count, header[3]] = float(l[2])
                            df.loc[count, header[4]] = float(c[10])
                            df.loc[count, header[5]] = float(c[11])
                    i += 1
                    count+=1
                    # print("i: ",i)

            except NotADirectoryError:
                print("Not a directory exception")
                return -1
        return df

