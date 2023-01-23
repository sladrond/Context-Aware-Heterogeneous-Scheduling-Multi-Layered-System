import time
import sys

#Types of tasks 
TYPE = {
    'IMAGE_CLASS': 0,
    'TEXT': 1,
    'AUDIO': 2,
    'IMAGE_SEGM': 3
    }

class Task():
  def __init__(self, type, data, min_score):
    self.type = TYPE[type]
    self.timestamp = time.time()
    self.data = data
    self.requested_score = min_score
    self.cpu_used = 0
    self.gpu_used = 0


  def get_data(self):
    return self.data

  def get_type(self):
    return self.type
  
  def get_timestamp(self):
    return self.timestamp

  def get_requested_score(self):
    return self.requested_score

  def get_size(self):
      return sys.getsizeof(self.data)