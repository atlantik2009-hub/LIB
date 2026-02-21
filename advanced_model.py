import sys, os, time, tempfile, random, shutil
import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt

#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *
from intraday_minutes_lib import *

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# INTRA_MODE
# 1  - 1 minute
# 10 - 10 minutes
# 60 - 1 hour
# 24 - 1 day
# 7  - 1 week
# 31 - 1 month
# 4  - 3 monthes
FRAME_UN  = 0
FRAME_1M  = 1
FRAME_10M = 2
FRAME_1H  = 3
FRAME_1D  = 4
FRAME_1W  = 5
FRAME_1M  = 6
FRAME_3M  = 7


#MODEL_TICKER_COLUMNS = ['Open','High','Low','Close','Volume']

@dataclass
class TimeFrame:
  op : float = -1
  hi : float = -1
  lo : float = -1
  cl : float = -1
  vo : float = -1

@dataclass
class FrameInfo:
  # init method or constructor
  def __init__(self, flag, index_flag):
    self.flag = flag
    self.index_flag = index_flag
    
  conf = FRAME_UN
  flag : bool = False 
  stamp: str
  num  : int = -1
  ticker_info : List[TimeFrame]
  index_flag = False
  index_info: List[TimeFrame]
       
    
@dataclass
class ModelSet:
    
  # init method or constructor
  def __init__(self, conf):
    self.model_config = conf
    
    for i in ("D", "M")
      id = self.model_config.find(i)
      if id >= 0 
        if i == "D":
          self.day_info.flag = True
          num = int(self.model_config[id+3:id+5])
            conf = FRAME_1D 
        elif i == "M":
          self.time_info.flag = True
          num = int(self.model_config[id+3:id+5])
          if(self.model_config[id+1:id+3] == '01')
            conf = FRAME_1M           
          elif(self.model_config[id+1:id+3] == '10')
            conf = FRAME_10M           
            
    LOGS(TRACE, "Config:", conf, "Number:", num)

  model_config: str # Format e.g. O01XXW01XXD01XXH01XXM01XXM10XX
  
  # Days info
  day_info : FrameInfo

  # Time info
  time_info: FrameInfo
  
  
  def get_dataset_struct(self):
    self.pred.append(test_stat.pred)

    
    
    