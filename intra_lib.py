import sys, os, time, tempfile, random, shutil
import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

BATCH_SIZE = 2048

NY_STOCK="NY"
MOEX_STOCK="MOEX"
CRYPT_STOCK="CRYP"

TICKER_ID_FILE = "MODELS\\ticker_id.csv"
TICKER_ID_CLMN = "Ticker_id"
TICKER_ID_FILE_CLMNS = ["Ticker", TICKER_ID_CLMN]

NY_FIRST_TIME_LIST = ["09:30:00"]
NY_LAST_TIME_LIST  = ["16:00:00"]
MX_FIRST_TIME_LIST = ["09:50:00", "10:00:00"]
MX_LAST_TIME_LIST  = ["18:50:00"]
FIRST_TIME_LIST = MX_FIRST_TIME_LIST
LAST_TIME_LIST = MX_LAST_TIME_LIST

CSV_COLUMNS_OLD = ['Datetime','Open','High','Low','Close','Adj Close', 'Volume']
CSV_COLUMNS = ['Datetime','Open','High','Low','Close','Volume']
INDEX_COL = 'Datetime'
MODEL_TICKER_COLUMNS = ['Open','High','Low','Close','Volume']
MODEL_COLUMNS = ['Open','High','Low','Close','Volume','Index']
VOLUME_COLUMN = 'Volume'
NORM_BY_COLUMN = ['Volume']
HIGH_COLUMN = 'High'
LOW_COLUMN = 'Low'  
CLOSE_COLUMN = 'Close'
TYPE_COLUMN = 'Type' 
PRED_COLUMN = 'pred'
DS_PREFIX_COLUMS = ['File', TYPE_COLUMN, 'Datetime']
INDEX_FILE = 'INDEX.csv'
INDEX_FILE2 = 'IMOEX.csv'
SKIPPED_FILES = [INDEX_FILE, INDEX_FILE2]
PREFIX_TICKER_MODEL = "model_ds_"
PREFIX_TICKER_MODEL_EX = "model_ds_5m_"
MODELS_PREFIX_LIST = ['model_ds_10m_MX3_', 'model_ds_10m_MX2_', 'model_ds_10m_MX_', "model_ds_01d_MYL_", "model_ds_01d_MXL_", PREFIX_TICKER_MODEL_EX, PREFIX_TICKER_MODEL]

PRED_STAT_COLUMNS_OLD2 = ['ID', 'Ticker', 'Model', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4']
PRED_STAT_COLUMNS_OLD = ['Ticker', 'Model', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4']
PRED_COLUMNS_LIST_OLD = ['CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4']

PRED_STAT_COLUMNS = ['Ticker', 'Model', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4',
                     'top20_d', 'top10_d', 'top05_d', 'avg_pos', 'avg_neg', 'pos_num', 'neg_num']
PRED_STAT_COLUMNS2 = ['ID', 'Ticker', 'Model', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4',
                     'top20_d', 'top10_d', 'top05_d', 'avg_pos', 'avg_neg', 'pos_num', 'neg_num']

PRED_COLUMNS_LIST = ['CL_HL_P1', 'CL_HL_P2', 'CL_HL_P3', 'CL_HL_P4', 
                     'top20_d', 'top10_d', 'top05_d', 'avg_pos', 'avg_neg', 'pos_num', 'neg_num']

TEST_STAT_COLUMNS = ['File', 'Type', 'Datetime', 'pred']

TICKERS_STAT_COLUMNS = ["TICKER", "POS", "HL", "NEG", "NOT_LOS", "SYNC_ER", "SKIP"]

TICKERS_STAT_FILE = "tickers_stat_file_"
CURRENT_DIR = ".\\"

class RC(Enum):
  OK          = 0
  SYNCH_ERROR = 1
  SKIPPED     = 2

@dataclass
class ModelSettings:
  p_days: int = 10   # Predict days number
  t_days: int = 5    # Target days number
  time_i: str = "1m" # Time interval: 1m, 5m, 15m, 1d
  mode  : str = "L"   # Mode: S(Short) or L(Long)
  percent: float = 1  # %  
  neg_delta: int = 10 # % from percent. For example if percent 1 and neg_delta=10. Nagative cases classified if looses more than 1*0.1=0.1%. Looses up to 0.1 % is NOT considered as negative
                      # 0 % means that ALL loses is classified as Negative cases
  first_day: bool = False # By default the last day is used and so the flag is False
  id_flag: bool = False # By default id file is NOT included  
  ds_name: str = "ML1"
  
  
@dataclass
class ClassifyStat:
  positive_num: int   = 0
  hl_num      : int   = 0 # For test run_mode
  negative_num: int   = 0
  not_looses_num: int = 0 
  syncup_error  : int = 0
  skipped:        int = 0


@dataclass
class TestStat:
  pred      : float = 0
  cl_counter: int = 0
  hl_counter: int = 0
  lo_counter: int = 0
  nl_counter: int = 0
  all_cases : int = 0
  cl_lo     : float = 0
  clhl_lo   : float = 0
  not_looses_lo : float = 0
  delta     : float = 0

def get_time_interval(DATASET_DIR):
  iv = '5m'
  if DATASET_DIR.find('1m') != -1:
    iv = '1m'
  elif DATASET_DIR.find('5m') != -1:
    iv = '5m'
  elif DATASET_DIR.find('10m') != -1:
    iv = '10m'
  elif DATASET_DIR.find('1d') != -1:
    iv = '1d'
  return iv

def get_file_list_by_pref_post(prefix, exception=None, postfix=".csv", path=CURRENT_DIR):
  rc_array = []
  files = os.listdir(path)
  files.sort()
  for f in files:
    if f.find(prefix) == 0 and f.find(postfix) != -1:
      if exception is not None and f.find(exception) != -1:
        continue
                 
      if path == CURRENT_DIR:
        rc_array.append(f)
      else:
        rc_array.append(str(path+f))
  print(rc_array, "\nLength=", len(rc_array))
  return rc_array

def get_params_old(name):
  rc = False
  mode = "U"
  percent = 0
  p = 0
  t = 0
  n = 0
  id_start = name.find("_S")
  if id_start == -1:
    id_start = name.find("_L")
    if id_start == -1: 
      return rc, mode, percent, p, t, n
    else:
      mode = "L"
  else:
    mode = "S"     

  percent = float(name[id_start+2:id_start+5]) # 3 bytes for percent

  offset = id_start 
  id_start = name[offset:].find("_P")
  if id_start == -1: 
    return rc, mode, percent, p, t, n
  id_start += offset     
  p = int(name[id_start+2: id_start+4]) # 2 bytes for prediction

  offset = id_start
  id_start = name[offset:].find("_T")
  if id_start == -1: 
    return rc, mode, percent, p, t, n  
  id_start += offset
  t = int(name[id_start+2: id_start+4]) # 2 bytes for target

  offset = id_start
  id_start = name[offset:].find("_N")
  if id_start == -1: 
    return rc, mode, percent, p, t, n
  id_start += offset  
  n = int(name[id_start+2: id_start+4]) # 2 bytes for normalization

  rc = True
  return rc, mode, percent, p, t, n


def get_params(name):
  rc = False
  mode = "U"
  percent = 0
  p = 0
  t = 0
  n = 0
  iv = '0m'
  id_flag = False
  ds_name="ML1"
  #count = 0
  #for pref in MODELS_PREFIX_LIST:
  #  if name.find(pref) != -1:
  #    if count in [0,1,2]:       
  #      iv = '10m'
  #      #print("iv:", iv)
  #    else:
  #      iv = '5m'
  #    break      
  #  count+=1
  iv = get_time_interval(name)

  #Firstly get ticker name
  ret = get_ticker_name_from_model(name)
  if len(ret) > 0:
    print("get_params(): Ticker name:", ret)
    idx = name.find(ret) + len(ret)
    # Update name to truncete unnecessary part of model name that is relate to the ticker name
    name = name[idx:]
    #print("Truncated model name:", name)
  #else:
  #  print("It is NOT ticker model, ret=", ret)    
  
  id_start = name.find("_S")
  if id_start == -1:
    id_start = name.find("_L")
    if id_start == -1: 
      return rc, mode, percent, p, t, n, iv
    else:
      mode = "L"
  else:
    mode = "S"     
   
  percent = float(name[id_start+2:id_start+5]) # 3 bytes for percent

  offset = id_start 
  id_start = name[offset:].find("_P")
  if id_start == -1: 
    return rc, mode, percent, p, t, n, iv
  id_start += offset     
  p = int(name[id_start+2: id_start+4]) # 2 bytes for prediction

  offset = id_start
  id_start = name[offset:].find("_T")
  if id_start == -1: 
    return rc, mode, percent, p, t, n, iv  
  id_start += offset
  t = int(name[id_start+2: id_start+4]) # 2 bytes for target

  offset = id_start
  id_start = name[offset:].find("_N")
  if id_start == -1: 
    return rc, mode, percent, p, t, n, iv
  id_start += offset  
  n = int(name[id_start+2: id_start+4]) # 2 bytes for normalization

  if name.find("_ID_") > 0:
    id_flag = True
    
  if name.find("_ID_") > 0:
    id_flag = True

  ds_position = name.find("ML")
  if ds_position >= 0:
    ds_name = name[ds_position:ds_position + 3]
  else:   
    ds_position = name.find("MS")      
    if ds_position >= 0:
      ds_name = name[ds_position:ds_position + 3]      
  print("Dataset name:", ds_name) 
        
  rc = True
  return rc, mode, percent, p, t, n, iv, id_flag, ds_name

def get_MODEL_by_model_name(model_name):
  rc = get_params(model_name)
  if rc[0] == False:
    print(model_name, "is NOT ok")
    return None
  #MODE        = rc[1]
  #PERCENT     = float(rc[2])
  #p_days      = int(rc[3])
  #t_days      = int(rc[4])
  # rc[5] Normalization
  # rc[6] time interval
  # rc[7] id_flag
  # rc[8] ds_name

  MODEL = ModelSettings(p_days=int(rc[3]), t_days=int(rc[4]), time_i=rc[6], mode = rc[1], percent = float(rc[2]), neg_delta=0, id_flag = rc[7], ds_name = rc[8])
  print(MODEL)
  return MODEL

def get_base_name(name):
  base_name = ""
  id_start = name.find("_S")
  if id_start == -1:
    id_start = name.find("_L")
    if id_start == -1: 
      return base_name
  id_start+=1
  id_end = name.find("_N")
  if id_end == -1:
    return base_name

  base_name = name[id_start: id_end+4] 
  return base_name

def get_ticker_name_from_model(model_name):
  rc = ""
  start_id = -1
  # Example: model_ds_BYND_L1.0_P10_T03_N10_E12000_T.zip
  # Example: model_ds_5m_ARWR_L1.0_P10_T03_N10_E12000_T.zip
  # Example: model_ds_10m_MX_AFKS_L1.0_P10_T03_N10_E20000_T.zip
  # Example: model_ds_10m_MX2_AFKS_L0.5_P10_T03_N10_E35000_T.zip
  for template in MODELS_PREFIX_LIST:
    id = model_name.find(template)
    if id == -1:
      continue
    else:
      start_id = id + len(template)
      break;
  
  if start_id == -1:
    return rc

  end_id = start_id + model_name[start_id:].find("_")
  rc = model_name[start_id:end_id]
  #print(start_id, end_id, rc)
  if (rc[0] == "L" or rc[0] == "S") and rc[1].isnumeric():
    #print(rc, "is NOT ticker")
    return ""

  return rc

def get_dataset_columns(MODEL):
  ds_columns = ['File', TYPE_COLUMN, 'Datetime']
  #print("1 ds_columns", ds_columns)
  cmn_length = MODEL.p_days+MODEL.t_days
  for column in MODEL_COLUMNS:
    for i in range(cmn_length):
      column_i = column + str(i)
      ds_columns.append(column_i)
  #print("2 ds_columns", ds_columns)
  return ds_columns

def get_dataset_columns_full(MODEL): # Used on step of learning
  ds_columns = get_dataset_columns(MODEL)

  if check_ticker_id_file(MODEL) == True:
    ds_columns.append(TICKER_ID_CLMN)
  return ds_columns


def get_learn_columns(MODEL):
  learn_columns = [TYPE_COLUMN]
  learn_columns.extend(get_learn_columns_clean(MODEL))
  return learn_columns

def get_learn_columns_full(MODEL): # Used on step of learning
  learn_columns = get_learn_columns(MODEL)

  if check_ticker_id_file(MODEL) == True:
    learn_columns.append(TICKER_ID_CLMN)

  return learn_columns


def get_entry_column(MODEL):
  str_out = CLOSE_COLUMN + str(MODEL.p_days-1)
  return str_out

def get_entry_column_volume(MODEL):
  str_out = VOLUME_COLUMN + str(MODEL.p_days-1)
  return str_out

def get_learn_columns_clean(MODEL):
  learn_columns = []
  COMMON_LENGTH = MODEL.p_days
  for column in MODEL_COLUMNS:
    for i in range(COMMON_LENGTH):
      column_i = column + str(i)
      learn_columns.append(column_i)
  return learn_columns

def get_learn_columns_clean_full(MODEL): # Used on step of testing
  learn_columns = get_learn_columns_clean(MODEL)

  if check_ticker_id_file(MODEL) == True:
    learn_columns.append(TICKER_ID_CLMN)

  return learn_columns


def get_close_target_columns_prifix(MODEL, prefix):
  out_columns = []
  out_columns.extend(prefix)
  #print("get_close_target_columns_prifix:", out_columns)
  column = CLOSE_COLUMN
  length = MODEL.t_days
  for i in range(length):
    column_i = column + str(i+MODEL.p_days)
    out_columns.append(column_i)
  #print("get_close_target_columns_prifix:", out_columns)
  return out_columns

def get_close_target_columns(MODEL):
  entry_clmn = CLOSE_COLUMN + str(MODEL.p_days-1)
  prefix = ['File', TYPE_COLUMN, 'Datetime', entry_clmn]
  #print("get_close_target_columns:", prefix)
  return get_close_target_columns_prifix(MODEL, prefix) 

def get_close_target_columns_entry(MODEL):
  entry_clmn = CLOSE_COLUMN + str(MODEL.p_days-1)
  prefix = [entry_clmn]
  #print("get_close_target_columns:", prefix)
  return get_close_target_columns_prifix(MODEL, prefix) 

# 1 - target is positive, 0 - target is negative, -1 NOT looses (NOT used in classification)
def classify_ds(selected_df, model_set, CHECK_TENDENCY=True):
  index_list = list(selected_df.index.values)
  entry_price  = selected_df.at[index_list[model_set.p_days-1], CLOSE_COLUMN]
  classify_list = selected_df[CLOSE_COLUMN].values.tolist()[model_set.p_days:model_set.p_days+model_set.t_days] 
  # Classify as target passed (positive case)
  rc = -1
  if model_set.mode == 'L':
    target_price = entry_price*(1+model_set.percent/100)
    if max(classify_list) >= target_price:
      rc = 1 
  else:
    target_price = entry_price*(1-model_set.percent/100)
    if min(classify_list) <= target_price:
      rc = 1 

  if rc == 1:
    if model_set.first_day == True:
      rc = -1
      if model_set.mode == 'L':
        if classify_list[0] >= target_price:
          rc = 1
        elif classify_list[0] < entry_price:
          return 0
      else:
        if classify_list[0] >= target_price:
          rc = 1
        elif classify_list[0] >= entry_price:
          return 0

  if rc == 1:
    if CHECK_TENDENCY == True:
      # Check if all elements in tail more/less than previous
      if model_set.mode == 'L':
        if classify_list[0] < entry_price:
          return -1
      else:
        if classify_list[0] > entry_price:
          return -1
        
      for j in range(len(classify_list)-1):
        if model_set.mode == 'L':
          if classify_list[j+1] < classify_list[j]:
            return -1
        else:
          if classify_list[j+1] > classify_list[j]:
            return -1

    print(index_list[0], "Entry price:", entry_price, "Target price:", target_price, "classify_list:", classify_list)
    return rc

  # Check the last element: if we loose classify it is as 0 (negative case)
  if model_set.mode == 'L':
    neg_k = 1 - model_set.percent*model_set.neg_delta/10000
    if classify_list[model_set.t_days-1] < entry_price*neg_k:
      return 0
  else:
    neg_k = 1 + model_set.percent*model_set.neg_delta/10000
    if classify_list[model_set.t_days-1] > entry_price*neg_k:
      return 0

  return -1


# 1 - target is positive, 0 - target is negative, -1 NOT looses (NOT used in classification)
def classify_ds_old(selected_df, model_set):
  index_list = list(selected_df.index.values)
  entry_price  = selected_df.at[index_list[model_set.p_days-1], CLOSE_COLUMN]
  classify_list = selected_df[CLOSE_COLUMN].values.tolist()[model_set.p_days:model_set.p_days+model_set.t_days] 
  # Classify as target passed (positive case)
  rc = -1
  if model_set.mode == 'L':
    target_price = entry_price*(1+model_set.percent/100)
    if max(classify_list) >= target_price:
      rc = 1 
  else:
    target_price = entry_price*(1-model_set.percent/100)
    if min(classify_list) <= target_price:
      rc = 1 

  if rc == 1:
    #print("Entry price:", entry_price, "Target price:", target_price, "classify_list:", classify_list)
    return rc

  # Check the last element: if we loose classify it is as 0 (negative case)
  if model_set.mode == 'L':
    neg_k = 1 - model_set.percent*model_set.neg_delta/10000
    if classify_list[model_set.t_days-1] < entry_price*neg_k:
      return 0
  else:
    neg_k = 1 + model_set.percent*model_set.neg_delta/10000
    if classify_list[model_set.t_days-1] > entry_price*neg_k:
      return 0

  return -1

def check_seria(selected_df, index_sel_df, MODEL, check_intraday):
  #Check first and last elements of ticker and index DFs: is data is  synced up
  exp_length = MODEL.p_days + MODEL.t_days
  if len(selected_df) != exp_length or len(index_sel_df) != exp_length:
    #print("ERROR: ticker and index length incorrect", "\nTicker date:", selected_df, "\nIndx:", index_sel_df)
    return RC.SYNCH_ERROR

  # Check first and last datetime
  last_elem = len(selected_df)-1
  if selected_df.index[last_elem] != index_sel_df.index[last_elem] or selected_df.index[0] != index_sel_df.index[0]:
    #print("ERROR: ticker and index length is NOT OK", "\nTicker date:", selected_df, "\nIndx:", index_sel_df)
    return RC.SYNCH_ERROR

  if MODEL.time_i == '1d':
    return RC.OK     

  # Format 2023-02-02
  if check_intraday == True:
    if selected_df.index[0][0:10] != selected_df.index[last_elem][0:10]:
      #print("INFO: seria NOT within one day", selected_df.index[0], selected_df.index[last_elem])
      return RC.SKIPPED

    for elem in FIRST_TIME_LIST:
      if selected_df.index[0].find(elem) != -1:
        #print("INFO: first elements skipped", selected_df.index[0], selected_df.index[last_elem])
        return RC.SKIPPED

    for elem in LAST_TIME_LIST:
      if selected_df.index[last_elem].find(elem) != -1:
        #print("INFO: last elements skipped", selected_df.index[0], selected_df.index[last_elem])
        return RC.SKIPPED
  
  # TODO to think if it is need for 5m and 10m
  for id in [0, last_elem]:
    if selected_df.index[id] != index_sel_df.index[id]:
      #print("ERROR: ticker and index data NOT synced up.", "Ticker date:", selected_df.index[id], "Indx:", index_sel_df.index[id])
      return RC.SYNCH_ERROR
  return RC.OK

def put_df_into_list(selected_df, index_sel_df, norm_days, row_list):
  #Check first and last elements of ticker and index DFs: is data is  synced up

  high_list = selected_df[HIGH_COLUMN].values.tolist() 
  max_val = max(high_list[0: norm_days])
  #print("Max high:", max_val)
  
  for column in MODEL_TICKER_COLUMNS:
    column_list = selected_df[column].values.tolist()
    # Normalize data by separate column
    if column in NORM_BY_COLUMN:
      predict_part = column_list[0:norm_days]
      max_val = max(predict_part)
      if max_val == 0:
        print("WARNING!!! Index", max_val)
        return False

    if max_val == 0:
      print("WARNING!!! Index", max_val)
      return False
    #print(column, max_val)
    norm_list = [float(i)/max_val for i in column_list]

    #print(column_list)
    #print(norm_list)
    row_list.extend(norm_list)

  # Index (SP500, MOEX...) part
  index_list = index_sel_df[CLOSE_COLUMN].values.tolist() 
  max_val = max(index_list[0: norm_days])
  if max_val == 0:
    print("WARNING!!! Index", max_val)
    return False
  norm_list = [float(i)/max_val for i in index_list]
  #print(index_list)
  #print(norm_list)

  row_list.extend(norm_list)
  return True

def prepare_ticker_df_old(ticker_file, ticker_data, index_data, MODEL, dataset_df):
  COMMON_LENGTH = MODEL.p_days+MODEL.t_days
  syncup_error = 0
  skipped = 0
  classify_stat = ClassifyStat(0,0,0,0,0)
  check_intraday = True
  j = -1
  for i in range(len(ticker_data)):
    j+=1
    selected_df  = ticker_data.iloc[i:COMMON_LENGTH + i, ]
    index_sel_df = index_data.iloc[j:COMMON_LENGTH + j, ]
    if len(selected_df) < COMMON_LENGTH:
      break
    if i == 0: 
      print("DEBUG: ticker_df(0)\n", selected_df.to_string())
      print("DEBUG: index_df\n", index_sel_df.to_string())
    
    # Verify seria if it is ok
    rc = check_seria(selected_df, index_sel_df, MODEL, check_intraday)
    if rc == RC.SYNCH_ERROR:
      #print(selected_df.to_string())
      #print(index_sel_df.to_string())

      classify_stat.syncup_error+=1
      # Tune the index for the index dataset
      try_i = 0
      while True:
        if try_i>=3:
          break
        if selected_df.index[try_i] in index_data.index:
          j = index_data.index.get_loc(selected_df.index[try_i])
          #print("Index file switched into", selected_df.index[try_i], "Try:", try_i)
          break
        try_i+=1
      continue
    elif rc == RC.SKIPPED:
      classify_stat.skipped+=1
      continue
    
    # Calculate classification flag
    rc = classify_ds(selected_df, MODEL)
    #print("classify_ds() returns", rc)
    if rc == 1:
      classify_stat.positive_num+=1
    elif rc == 0:
      classify_stat.negative_num+=1
    else:
      # No looses: case is NOT interested in
      classify_stat.not_looses_num+=1
      continue
    
    classify_flag = rc
    row_list = []
    row_list.append(ticker_file)
    row_list.append(classify_flag)
    row_list.append(selected_df.index[0])
    rc = put_df_into_list(selected_df, index_sel_df, MODEL.p_days, row_list)
    
    if rc == False:
      print("WARNING!!! prepare_ticker_df_old() failed")
      return classify_stat

    #print("DS length:", len(row_list), "\nDS row:", row_list)
    dataset_df.loc[len(dataset_df)] = row_list
    #print(dataset_df)
  return classify_stat

def prepare_ticker_df(ticker_file, ticker_data, index_data, MODEL, dataset_df):
  COMMON_LENGTH = MODEL.p_days+MODEL.t_days
  syncup_error = 0
  skipped = 0
  classify_stat = ClassifyStat(0,0,0,0,0)
  check_intraday = True
  j = -1
  for i in range(len(ticker_data)):
    j+=1
    selected_df  = ticker_data.iloc[i:COMMON_LENGTH + i, ]
    index_sel_df = index_data.iloc[j:COMMON_LENGTH + j, ]
    if MODEL.time_i == '1d' and selected_df.index[0] != index_sel_df.index[0]:
      if selected_df.index[0] in index_data.index:
        j = index_data.index.get_loc(selected_df.index[0])
        #print("Index file switched into", selected_df.index[0], "j=", j)
        index_sel_df = index_data.iloc[j:COMMON_LENGTH + j, ]
      else:
        print(selected_df.index[0], "Not found in index_data")
        continue

    if len(selected_df) < COMMON_LENGTH:
      break
    if i == 0 or i == len(ticker_data) - 1: 
      print("DEBUG: ticker_df(", i , ")\n", selected_df.to_string())
      print("DEBUG: index_df\n", index_sel_df.to_string())
    
    # Verify seria if it is ok
    rc = check_seria(selected_df, index_sel_df, MODEL, check_intraday)

    # TODO check if it is NEEDED for 5m and 10m, probaly tunning above is enough
    if MODEL.time_i != '1d' and rc == RC.SYNCH_ERROR:
      # Tune the index for the index dataset
      try_i = 0
      while True:
        if try_i>=3:
          break
        if selected_df.index[try_i] in index_data.index:
          j = index_data.index.get_loc(selected_df.index[try_i])
          #print("Index file switched into", selected_df.index[try_i], "Try:", try_i)
          break
        try_i+=1

    if rc == RC.SYNCH_ERROR:
      #print(selected_df.to_string())
      #print(index_sel_df.to_string())
      classify_stat.syncup_error+=1
      continue
    elif rc == RC.SKIPPED:
      classify_stat.skipped+=1
      continue
    
    # Calculate classification flag
    rc = classify_ds(selected_df, MODEL)
    #print("classify_ds() returns", rc)
    if rc == 1:
      classify_stat.positive_num+=1
    elif rc == 0:
      classify_stat.negative_num+=1
    else:
      # No looses: case is NOT interested in
      classify_stat.not_looses_num+=1
      continue
    
    classify_flag = rc
    row_list = []
    row_list.append(ticker_file)          # Ticker
    row_list.append(classify_flag)        # Classification flag
    row_list.append(selected_df.index[0]) # Datetime
    rc = put_df_into_list(selected_df, index_sel_df, MODEL.p_days, row_list)
    if rc == False:
      print("WARNING!!! Something wrong")
      return classify_stat
         
    #print("DS length:", len(row_list), "\nDS row:", row_list)
    dataset_df.loc[len(dataset_df)] = row_list
    #print(dataset_df)
  return classify_stat

def check_ticker_id_file(MODEL):
  print("check_ticker_id_file: ID flag: ", MODEL.id_flag)
  if MODEL.id_flag == True:
    ticker_id_fn = get_ticket_id_file_name(MODEL)
    if os.path.isfile(ticker_id_fn):
      print("Ticker id file is configured")
      return True
  return False

def post_proc_df_add_clmns(dataset_df, MODEL):
  if check_ticker_id_file(MODEL) == True:
    print(TICKER_ID_CLMN, "added")
    dataset_df[TICKER_ID_CLMN] = 0 
    fill_ticker_id_clmn(dataset_df, MODEL)
    # Check if ticker_id value is OK
    if(len(dataset_df[(dataset_df['Ticker_id'] == 0)]) > 0):
      print("!!!Ahtung: Ticker_id configured incorrectly")
      exit(0)
    else:
      print("GOOD: Ticker_id values: ", dataset_df.Ticker_id.unique())  
    return True

  return False

def get_dataset_file(DATASET_DIR, MODEL):
  POSTFIX = ""
  if check_ticker_id_file(MODEL) == True:
    POSTFIX = "_ID"

  DS_OUT_FILE = DATASET_DIR + "_"+ MODEL.mode + '{:1.1f}'.format(MODEL.percent) + "_P" + str(MODEL.p_days).zfill(2) + "_T" + str(MODEL.t_days).zfill(2) + "_N" + str(MODEL.p_days).zfill(2) + POSTFIX + ".csv"
  return DS_OUT_FILE

# Return dataset file name
def prepare_dataset(DATASET_DIR, MODEL): 
  DS_OUT_FILE = get_dataset_file(DATASET_DIR, MODEL)
  print("DS_OUT_FILE:", DS_OUT_FILE)
  TICKERS_STAT_FILE_OUT = TICKERS_STAT_FILE + DATASET_DIR + "_" + MODEL.mode + '{:1.1f}'.format(MODEL.percent) + "_P" + str(MODEL.p_days).zfill(2) + "_T" + str(MODEL.t_days).zfill(2) + ".csv" #TODO for ID
  print("TICKERS_STAT_FILE_OUT:", TICKERS_STAT_FILE_OUT)
  
  COMMON_LENGTH = MODEL.p_days+MODEL.t_days
  DATASET_COLUMNS = get_dataset_columns(MODEL)  
  print(DATASET_COLUMNS)
  print("DS length:", len(DATASET_COLUMNS))
  LEARN_COLUMNS_NUM = MODEL.p_days*len(MODEL_COLUMNS)
  ALL_COLUMNS_NUM = COMMON_LENGTH*len(MODEL_COLUMNS)
  print("Learn columns number:", LEARN_COLUMNS_NUM )
  print("All columns number:", ALL_COLUMNS_NUM )                   
  
  dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
  tickers_stat_df = pd.DataFrame(columns=TICKERS_STAT_COLUMNS)  

  index_file = DATASET_DIR + '/'+ INDEX_FILE

  print("INDEX file:", index_file)
  index_data = pd.read_csv( index_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
  print("Index file:\n", index_data)
  
  classify_stat = ClassifyStat(0,0,0,0,0)
  
  counter = 0
  entries = os.listdir(DATASET_DIR)
  for entry in entries:
    # Skip Idex file
    skip_flag = False
    for file_x in SKIPPED_FILES:   
      if entry.find(file_x) != -1:
        skip_flag = True
        break

    if skip_flag == True:
      print(entry, "skipped")
      continue

    if os.path.isdir(entry) or entry.find(".csv") == -1:
      print(entry, "skipped")
      continue

    ticker_file = DATASET_DIR + '/'+ entry
    counter+=1
    print(counter, "!!!", ticker_file, "!!!")
    ticker_data = pd.read_csv( ticker_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)

    ticker_stat = prepare_ticker_df(ticker_file, ticker_data, index_data, MODEL, dataset_df)

    tickers_stat_list = [ticker_file, ticker_stat.positive_num, ticker_stat.hl_num, ticker_stat.negative_num, ticker_stat.not_looses_num, ticker_stat.syncup_error, ticker_stat.skipped]
    tickers_stat_df.loc[len(tickers_stat_df)] = tickers_stat_list
   
    classify_stat.positive_num = classify_stat.positive_num + ticker_stat.positive_num
    classify_stat.hl_num = classify_stat.hl_num + ticker_stat.hl_num
    classify_stat.negative_num = classify_stat.negative_num + ticker_stat.negative_num
    classify_stat.not_looses_num = classify_stat.not_looses_num + ticker_stat.not_looses_num
    classify_stat.syncup_error = classify_stat.syncup_error + ticker_stat.syncup_error
    classify_stat.skipped = classify_stat.skipped + ticker_stat.skipped
    
    print(entry, "Length DS:", len(dataset_df))
    print("Classify stat:", classify_stat)
  print("Sum check:", (classify_stat.positive_num + classify_stat.hl_num + classify_stat.negative_num + classify_stat.not_looses_num))  
  ### Here we can add additional columns like Ticker id ###
  post_proc_df_add_clmns(dataset_df, MODEL)

  dataset_df.to_csv(DS_OUT_FILE, index=False)

  tickers_stat_df = tickers_stat_df.sort_index().sort_values('POS', kind='mergesort', ascending=False)
  print(tickers_stat_df.to_string())
  tickers_stat_df.to_csv(TICKERS_STAT_FILE_OUT, index=False)
  return DS_OUT_FILE

def get_ticket_id_file_name(MODEL):
  file_name = TICKER_ID_FILE
  
  if MODEL.ds_name.find("ML3") >= 0 or MODEL.ds_name.find("MS3") >= 0:
    file_name = "MODELS\\ML3_ticker_id.csv" # MOEX index ticker list
  elif MODEL.ds_name.find("ML4") >= 0 or MODEL.ds_name.find("MS4") >= 0:
    file_name = "MODELS\\ML4_ticker_id.csv" # MOEX index ticker list

  print("Ticker id file name: ", file_name)
  return file_name

    
def fill_ticker_id_clmn(dataset_df, MODEL=None):
  print("fill_ticker_id_clmn")

  ticker_id_file_name = get_ticket_id_file_name(MODEL)
  ticker_id_df = pd.read_csv( ticker_id_file_name, names=TICKER_ID_FILE_CLMNS, index_col = "Ticker")
  if len(ticker_id_df) <= 0:
    print("!!!Ahtung: during reading", TICKER_ID_FILE)
  #print(ticker_id_df)
  #print(dataset_df.head(5))

  for id, row_id in dataset_df.iterrows():
    ticker = dataset_df.at[id, 'File']
    #print("Ticker:", ticker)
    # Tune format "ds_01d_CRY7T/1INCH-USD.csv"
    offset = ticker.find("/")
    #offset2 = ticker.find(".csv")
    if offset > 0:
       ticker = ticker[offset+1:-4]

    if ticker not in ticker_id_df.index:
      print("!!!AHTUNG fill_ticker_id_clmn()", ticker, " is absent in ticker id file" )
      return False
    
    dataset_df.at[id, TICKER_ID_CLMN] = ticker_id_df.at[ticker, TICKER_ID_CLMN]    

  print("fill_ticker_id_clmn() executed successfully")
  print(dataset_df.head(5))  
  print(dataset_df.tail(5))
  return True

def only_txt_into_page(pdf, title, text, title_sz=9, text_sz=6):
  print("only_txt_into_page() called")
  NUM_OF_LINES = 160 # it is for font 3
  tuned_num_of_lines = NUM_OF_LINES/(text_sz/3)

  plt.rcParams['font.family'] = 'Serif'

  fig = plt.figure(figsize=(12, 9))
  fig.clf()
  y_shift = 0.925
  fig.text(0.05, 0.95, title, transform=fig.transFigure, size=title_sz)

  page_txt = ""
  counter = 0
  for line in text.splitlines():
    page_txt = page_txt + line + "\n"
    counter += 1
    if counter%tuned_num_of_lines == 0:
      if counter > tuned_num_of_lines:
        y_shift = 0.95

      fig.text(0.05, y_shift, page_txt, horizontalalignment='left', verticalalignment='top', transform=fig.transFigure, size=text_sz)    
      pdf.savefig()
      plt.close()
      
      # Open new page
      plt.rcParams['font.family'] = 'Serif'
      fig = plt.figure(figsize=(12, 9))
      fig.clf()
      page_txt = ""

  # Close the last page
  fig.text(0.05, y_shift, page_txt, horizontalalignment='left', verticalalignment='top', transform=fig.transFigure, size=text_sz)    
  pdf.savefig()
  plt.close()

def get_ticker_list_by_model(model_name):
  if model_name.find("_S") >= 0:
    if model_name.find("model_ds_01d_MNS") >= 0:
      TICKERS = "MODELS\\short_new.csv"
    elif model_name.find("model_ds_01d_ML1") >= 0:
      TICKERS = "MODELS\\MS2.csv"
    elif model_name.find("model_ds_01d_ML3") >= 0:
      TICKERS = "MODELS\\ML3.csv" # MOEX index ticker list     
    elif model_name.find("model_ds_01d_ML4") >= 0:
      TICKERS = "MODELS\\ML4.csv" # MOEX index ticker list             
    else:
      TICKERS = "MODELS\\MS2.csv"

  if model_name.find("_L") >= 0:
    if model_name.find("_MLI_") >= 0:
      TICKERS = "MODELS\\moex_ind_long.csv"
    elif model_name.find("model_ds_01d_ML1") >= 0:
      TICKERS = "MODELS\\ML1.csv"
    elif model_name.find("model_ds_01d_ML3") >= 0:
      TICKERS = "MODELS\\ML3.csv" # MOEX index ticker list
    elif model_name.find("model_ds_01d_ML4") >= 0:
      TICKERS = "MODELS\\ML4.csv" # MOEX index ticker list              
    else:
      TICKERS = "MODELS\\TCS_MOEX_long.csv"

  print("Ticker file: ", TICKERS)
  tickers_df = pd.read_csv(TICKERS, names=["Ticker"], index_col=False)
  tickers_list = tickers_df['Ticker'].unique()
  print("Tickers list:", tickers_list, "Length:", len(tickers_list)) 
  return tickers_list
  