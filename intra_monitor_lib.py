import sys
import shutil
import os

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import date
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from statistics import mean 


#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *
from loader_lib import *
from draw_lib import *
from report_final_lib import *

MODELS_DIR = "MODELS"
RESULT_FILE = "result_file.txt"
GENERAL_TICKER_PREFIX = "XXX"
GENERAL_MODEL_PREFIX_LIST = ["model_ds_10m_MX3_L", "model_ds_5m_L"]

RUNTIME_PRED_STAT_COLUMNS = ['Ticker', 'Model', 'Pred', 'Price', 'Target', 'Entry_date', 'Vol_avg', 'Vol_pre', 'Vol_MLN_RUB']

TICKERS_FILE="tickers.csv"

def handle_ticker_file(start_date, INTRA_MODE, pred_stat_data, stock=NY_STOCK, TICKERS_FLAG=0):
  SLEEP_TM = 5
  #if stock == MOEX_STOCK:
  #  SLEEP_TM = 5
  BUNCH_SZ = 30 # TODO
 
  print(pred_stat_data.to_string()) 
  print("Number of entries: ", len(pred_stat_data))
  if TICKERS_FLAG == 0:                                       
    tickers_list = pred_stat_data['Ticker'].unique()
    print("Tickers list:", tickers_list, "Length:", len(tickers_list))
 
  counter = 0
  while True:
    # Get ticker list from the file on each iteration
    if TICKERS_FLAG == 1:
      df_tickers = pd.read_csv(TICKERS_FILE, names=['Ticker'], index_col=False)
      tickers_list = df_tickers['Ticker'].unique()
      print("Tickers list:", tickers_list, "Length:", len(tickers_list))

    #for index, row in df_tickers.iterrows():
    #  ticker = df_tickers.at[index, 'Symbol']
    #  ticker_list.append(ticker)
    #  counter+=1
    #  if counter >= BUNCH_SZ:
    #    break
    #print("Bunch includes:", ticker_list)
    print("Start: iteration", counter, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    load_tickers_bunch(tickers_list, start_date, INTRA_MODE, pred_stat_data, stock)
    print("Finish: iteration", counter, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 

    print("Sleeping", SLEEP_TM, "sec...")
    time.sleep(SLEEP_TM)
    counter+=1

def write_into_result_file(text):
  # Check if the text already exists in the file
  text_tmp = text + "\n"
  if os.path.isfile(RESULT_FILE):
    with open(RESULT_FILE, 'r') as f:
      if text_tmp in f.read():
        print("The line already in the file")
        return False

  f = open(RESULT_FILE,"a")
  f.write(text) 
  f.close() 
  return True

def get_index_from_pred_stat_data(ticker_in, model_name, pred_stat_data):
  for index, row in pred_stat_data.iterrows():
    model  = pred_stat_data.at[index, 'Model']
    ticker = pred_stat_data.at[index, 'Ticker']
    if model_name.find(model) != -1 and ticker_in.find(ticker) == 0 and len(ticker_in) == len(ticker):
      print(index, ticker, model, "to be checked")
      return index,row
  
  return -1, None
 
# Return Severity level - nimber of Pred that is achived
def check_pred_in_range(ticker_in, model_name, pred, pred_stat_data):
  print("check_pred_in_range() called with", ticker_in, model_name, pred)

  index, row = get_index_from_pred_stat_data(ticker_in, model_name, pred_stat_data)
  if index == -1:
   return -1

  for i in [4,3,2,1]:
    clmn = "Pred" + str(i)
    if float(pred_stat_data.at[index, clmn]) > 0 and pred >= float(pred_stat_data.at[index, clmn]):
      print("!!!", ticker_in, pred, "achived", "\nDetails:\n", row.to_frame().T, "!!!", "Severity:", i)
      return i
  print(pred, "is NOT in range. Pred1 = ", pred_stat_data.at[index, 'Pred1'] )
  return 0 

  print("!!!WARNING: something wrong", ticker_in, model_name, "NOT found")
  
  return 0 

def get_entry_price(ticker_data, MODEL):
  if MODEL.time_i == "1d" or MODEL.time_i == "24":
    entry_price = ticker_data.Close.iloc[-1]
  else:
    # Entry price is located in previous before last one
    entry_price = ticker_data.Close.iloc[-2]

  return entry_price

def get_target_price(entry_price, MODEL):
  if MODEL.mode == "L":
    target_price = round((entry_price * (1 + MODEL.percent/100) ), 3)
  else:
    target_price = round((entry_price * (1 - MODEL.percent/100) ), 3)
  
  return target_price

def get_end_date(ticker_data, MODEL, stock):
  return str(ticker_data.index[-1])[0:10]

def get_start_date(ticker_data, MODEL):
  if MODEL.time_i == "1d" or MODEL.time_i == "24":
    start_date = str(ticker_data.index[-10])[0:10]
  else:
    start_date=str(date.today())

  return start_date

def wtite_text_for_result(ticker_in, ticker_data, model_name, pred, pred_stat_data, stock=MOEX_STOCK):            
  #FORMAT = "LONG"
  FORMAT="SHORT"
  index, row = get_index_from_pred_stat_data(ticker_in, model_name, pred_stat_data)
  if index == -1:
    return 0 

  for i in [4,3,2,1]:
    clmn = "Pred" + str(i)
    if float(pred_stat_data.at[index, clmn]) > 0 and pred >= float(pred_stat_data.at[index, clmn]):
      MODEL = get_MODEL_by_model_name(model_name)
      end_date = ""

      entry_price  = get_entry_price(ticker_data, MODEL)
      target_price = get_target_price(entry_price, MODEL)
      start_date   = get_start_date(ticker_data, MODEL)

      end_date = ""
      if MODEL.time_i == "1d" or MODEL.time_i == "24":
        if stock == MOEX_STOCK:
          end_date = str(ticker_data.index[-1])[0:10]
        else:
          end_date = str(ticker_data.index[-1] + timedelta(days=1))[0:10]          

      print("Period to show:", start_date, end_date, "entry price:", entry_price, "target price:", target_price)    

      if FORMAT == "LONG":
        txt = "##### " + model_name + " achived " + str(pred) + " Severity: " + str(i) + " #####" + "\nPred details: \n" + str(pred_stat_data.loc[[index]]) + "\n"
        txt = txt + str(ticker_data.tail(1)) + "\n" + "Price entry=" + str( round(ticker_data.Close.iloc[-1],3) ) + "; Target price=" + str(target_price) + "; Percent=" + str(MODEL.percent) + "\n\n"
      else:
        ticker = get_ticker_name_from_model(model_name)
        # Check if it is general model
        if len(ticker) == 0:
          ticker = ticker_in
        if MODEL.time_i == "1d" or MODEL.time_i == "24":
          time_str1 = str(ticker_data.index[-10])[0:10]
          time_str2 = str(ticker_data.index[-1])[0:10]       
        else:      
          time_str1 = str(ticker_data.index[-2])
          time_str2 = str(ticker_data.index[-1])[11:]
        # For MOEX data Date and time is stored in pred_stat_data.Datetime.loc[index] but not in index
        #if len(time_str1) < 10:
        #  time_str1 = str(pred_stat_data.Datetime.loc[index-1])
        #  time_str2 = str(pred_stat_data.Datetime.loc[index])       
        txt = "\n" + time_str1 + "/" + time_str2 +  "  " + ticker.rjust(4, ' ') + "  " + MODEL.mode + str(MODEL.percent) + "  Price: " + \
              str( round(entry_price,3) ).ljust(8, ' ') + "->" + str(target_price).ljust(8, ' ') + "  Severity=" + str(i) + "  " + str(round(pred,3)) + "  Pred_list:" + \
              str(round(pred_stat_data.Pred1.loc[index], 3)) + "/" + str(round(pred_stat_data.Pred2.loc[index], 3)) + "/" + str(round(pred_stat_data.Pred3.loc[index], 3)) + "/" + \
              str(round(pred_stat_data.Pred4.loc[index], 3)) + " " + model_name + "   " + str(MODEL.t_days) + "  " + MODEL.time_i  
      print(txt)        
      write_into_result_file(txt)
      pdf_file = time_str2 + "_" + ticker + "_"+ MODEL.mode + str(MODEL.percent) + ".pdf"
      pdf_file = pdf_file.replace(":", "_")        
      draw_ticker_charts(ticker, stock, txt, start_date, pdf_file, model_name, pred_stat_data.Pred1.loc[index], pred, end_date)
      return i

  return 0 

def get_last_volume_per_average(ticker_data):
  vol_list = ticker_data['Volume'].tolist()
  length = len(vol_list)
  last_elem = vol_list[length-1]
  vol_list.pop() # Drop last element
  avg = mean(vol_list)
  if avg == 0:
    return 0
  rc = last_elem/avg
  #print("PROD: get_last_volume_per_average() returns", last_elem, "/", avg, "=", rc)
  return rc

def get_previous_volume(ticker_data):
  vol_list = ticker_data['Volume'].tolist()
  length = len(vol_list)

  if length < 2 or vol_list[length-2] == 0:
    print("get_previous_volume() incorrect data:", vol_list)  
    return -1

  return (vol_list[length-1]/vol_list[length-2])

def get_last_volume_mln(ticker_data):
  vol_list = ticker_data['Volume'].tolist()
  length = len(vol_list)

  return round((vol_list[length-1]/1000000), 1)

def show_filtered_df(runtime_df):
  counter = 6 
  runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
  filtered_df = runtime_df[(runtime_df['Pred'] >= 0.2) & (runtime_df['Vol_avg'] > 2) & (runtime_df['Vol_pre'] > 2)] 
  print(counter, "Filtered (Sorted by Pred, Pred>=0.2, Vol_avg > 2, Vol_pre > 2):\n", filtered_df.head(20).to_string()) 

  pred_list = [0.15, 0.2, 0.25, 0.3]
  for i in pred_list: 
    counter += 1
    runtime_df = runtime_df.sort_index().sort_values('Vol_pre', kind='mergesort', ascending=False)
    filtered_df = runtime_df[(runtime_df['Pred'] >= i) & (runtime_df['Vol_avg'] > 1) & (runtime_df['Vol_pre'] > 1)] 
    print(counter, "Filtered (Sorted by Vol_pre, Pred>=", i, ":\n", filtered_df.head(20).to_string())

# return dataset for prediction processing
def load_tickers_bunch(ticker_list, start_date, INTRA_MODE, pred_stat_data, stock, end_period=""):
  MOEX_MODEL = "model_ds_01d_MXL_L3.0_P10_T01_N10_E5000_T.zip"
  #MOEX_MODEL = "model_ds_01d_MXL_L8.0_P20_T01_N20_E10000_T.zip"
  #MOEX_MODEL = "model_ds_01d_MXL_L8.0_P15_T01_N15_E10000_T.zip" # BEST
  #MOEX_MODEL = "model_ds_01d_MXL_L8.0_P10_T01_N10_E15000_T.zip"

  print("load_tickers_bunch() caled with:", start_date, INTRA_MODE, stock, end_period)
  print(pred_stat_data)
  runtime_df = pd.DataFrame(columns=RUNTIME_PRED_STAT_COLUMNS)

  # TODO FIX ME: structure 10 and 0 hardcoded 
  if pred_stat_data is not None:
    model_name_inst = pred_stat_data.at[0, 'Model']
  else:
    if stock == MOEX_STOCK:
       model_name_inst = MOEX_MODEL
    else: 
      model_name_inst = "model_ds_01d_MYL_L4.0_P10_T01_N10_E5000_T.zip"   
  print("Model example:", model_name_inst)
  MODEL = get_MODEL_by_model_name(model_name_inst)
  MODEL.t_days = 0
  MODEL.time_i = INTRA_MODE

  # Now separate tickers list provided
  #ticker_list = pred_stat_data['Ticker'].unique()
  DATASET_COLUMNS = get_dataset_columns(MODEL)  
  print(DATASET_COLUMNS)
  print("DS length:", len(DATASET_COLUMNS))

  if stock == NY_STOCK:
    index_ticker = SP500_INDEX
  elif stock == CRYPT_STOCK:
    index_ticker = CRYPT_INDEX
  elif stock == MOEX_STOCK:
    index_ticker = MOEX_INDEX
  else:
    print("Error: undefined stock: ", stock)  
    return None
  print("Index ticker:", index_ticker)  

  while(True):
    index_data = load_ticker_data(index_ticker, start_date, INTRA_MODE, stock, end_period)
    if(index_data is None):
      print("ERROR:", index_ticker, "NOT loaded")   
      print("Sleeping", 9, "sec...")
      time.sleep(9)
    else:
      break

  print(index_ticker, "loaded")
  print(index_data.tail(10), "\nSize:", len(index_data))
  ticker_counter = 0
  for ticker in ticker_list:
    ticker_counter+=1

    if index_data is None or (MODEL.time_i != '1d' and ticker_counter%20 == 0):
      print("Reloading index ticker:", index_ticker)  
      index_data = load_ticker_data(index_ticker, start_date, INTRA_MODE, stock, end_period)
      if index_data is None:
        print("ERROR:", index_ticker, "NOT loaded")   
        print("Sleeping", 10, "sec...")
        time.sleep(10)
        continue
 
    dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
    print(ticker_counter, ticker, "loaded")
    ticker_data = load_ticker_data(ticker, start_date, INTRA_MODE, stock, end_period)
    if(ticker_data is None or len(ticker_data) == 0):
      print("ERROR:", ticker, "NOT loaded")   
      continue
    elif INTRA_MODE == "1d":
      print(ticker_data.tail(10))

    if pred_stat_data is not None:
      model_list = get_model_by_ticker_from_stat(ticker, pred_stat_data)
    else:
      if stock == MOEX_STOCK:
        model_name_inst = "MODELS\\" + MOEX_MODEL
      else: 
        model_name_inst = "MODELS\\model_ds_01d_MYL_L4.0_P10_T01_N10_E5000_T.zip"   
      model_list = [model_name_inst]

    if(len(model_list) == 0):
      print("Model NOT found")
      continue 
    rc = prepare_ticker_df_monitor(ticker, ticker_data, index_data, MODEL, dataset_df)
    if rc == False:
      print("ERROR: prepare_ticker_df_monitor()")
      continue

    print("Prepared dataset:\n", dataset_df)
    for model_name in model_list:
      print("##### ", ticker, model_name, " #####")
      
      pred = predict_monitor_df(ticker, dataset_df, model_name)
      ### Collect statistics ###
      row_list = []
      row_list.append(ticker)
      row_list.append(model_name)
      row_list.append(pred)
      entry_price  = get_entry_price(ticker_data, MODEL)
      target_price = get_target_price(entry_price, MODEL)
      end_date_in  = get_end_date(ticker_data, MODEL, stock)
      row_list.append(entry_price)
      row_list.append(target_price)
      row_list.append(end_date_in)
      vol_avg = get_last_volume_per_average(ticker_data)    
      row_list.append(vol_avg)
      vol_pre = get_previous_volume(ticker_data)
      row_list.append(vol_pre)
      vol_mln = get_last_volume_mln(ticker_data)
      row_list.append(vol_mln)
      runtime_df.loc[len(runtime_df)] = row_list
      ### Collect statistics ###

      if pred_stat_data is not None:
        rc = check_pred_in_range(ticker, model_name, pred, pred_stat_data)
        if rc > 0:
          wtite_text_for_result(ticker, ticker_data, model_name, pred, pred_stat_data, stock)

  print("load_tickers_bunch() end")

def load_ticker_data(ticker, start_date, INTRA_MODE, stock, end_period=""):
  if end_period == "":
    end_date = str(date.today() + timedelta(1)) 
  else:
    end_date = end_period
  # For debugging purpose
  #start_date = "2023-02-15"
  #end_date   = "2023-02-15 16:40:00"
  ######################
  return load_df_ext(ticker, start_date, end_date, INTRA_MODE, stock)

def load_ticker_data_old(ticker, start_date, INTRA_MODE, stock):
  CHECK_ENABLE = False
  #print(ticker, start_date, INTRA_MODE)

  try: 
    ticker_data_df = yf.download(ticker, start=start_date, interval=INTRA_MODE, progress=False, show_errors=False, threads=False, timeout=1)
    #ticker_data_df = yf.download(ticker, start="2023-02-15", end="2023-02-16", interval=INTRA_MODE, progress=False, show_errors=False, threads=False, timeout=1) # Simulate for debugging 
  except ValueError: 
    print("load_ticker_data_old: Exception catched on ticker", ticker)
    return Null

  if len(ticker_data_df) == 0:
    return "";
  if CHECK_ENABLE == True:
    bad_entry_counter = 0
    for id, row_id in ticker_data_df.iterrows():
      if pd.isna(ticker_data_df.at[id,'Open']):
        bad_entry_counter = bad_entry_counter + 1
      ticker_data_df = ticker_data_df.drop(labels=id, axis=0)
    print("Bad entries", bad_entry_counter)

  #print(ticker, ":")
  #print(ticker_data_df)
  return ticker_data_df

def prepare_ticker_df_monitor(ticker_file, ticker_data, index_data, MODEL, dataset_df):
  COMMON_LENGTH = MODEL.p_days
  #print("PROD: prepare_ticker_df_monitor()" )
  if COMMON_LENGTH > len(ticker_data) or COMMON_LENGTH > len(index_data):
    print("Not enough data")
    return False
  #print(ticker_data.tail(10), "\nVolume:", ticker_data.Volume.iloc[-1])
  #print(index_data.tail(10), "\nVolume:", index_data.Volume.iloc[-1])
  if ticker_data.Volume.iloc[-1] == 0:
    #print("Dropping last elements with volume 0 from ticker", ticker_data.Volume.iloc[-1])
    ticker_data = ticker_data.iloc[:-1] 
  if index_data.Volume.iloc[-1] == 0:
    #print("Dropping last elements with volume 0 from index", index_data.Volume.iloc[-1])
    index_data = index_data.iloc[:-1] 
  
  i = len(ticker_data) - COMMON_LENGTH
  j = len(index_data) - COMMON_LENGTH  
  selected_df  = ticker_data.iloc[i:COMMON_LENGTH + i, ]
  index_sel_df = index_data.iloc[j:COMMON_LENGTH + j, ]
  # Preliminary check
  if selected_df.index[0] != index_sel_df.index[0]:
    print("Syncup warning: ticker start:", selected_df.index[0], "index start:", index_sel_df.index[0])  
    if selected_df.index[0] in index_data.index:
       j = index_data.index.get_loc(selected_df.index[0])
       index_sel_df = index_data.iloc[j:COMMON_LENGTH + j, ]
       print("Index start moved into:", index_sel_df.index[0]) 

  #print("Selected Index\n", index_sel_df, "\nSelected DS:\n", selected_df )

  # The same check is located into check_seria()   
  #if len(selected_df) < COMMON_LENGTH:
  #  print("ERROR: NOT enough Ticker data len:", len(selected_df), "expected:", len(selected_df))
  #  return False

  #if len(index_sel_df) < COMMON_LENGTH:
  #  print("ERROR: NOT enough Index data len:", len(index_sel_df), "expected:", len(index_sel_df))
  #  return False

  check_intraday = False
  rc = check_seria(selected_df, index_sel_df, MODEL, check_intraday)
  if rc == RC.SYNCH_ERROR:
    print("ERROR: SYNCH_ERROR", "\nIndex\n", index_sel_df, "\nTicker\n", selected_df)
    return False
  elif rc != RC.OK:
    print("ERROR: RC = ", rc, "\n", "\nIndex\n", index_sel_df, "\nTicker\n", selected_df)
    return False

  classify_flag = '-1'
  row_list = []
  row_list.append(ticker_file)
  row_list.append(classify_flag)
  row_list.append(selected_df.index[0])
  rc = put_df_into_list(selected_df, index_sel_df, MODEL.p_days, row_list)
  
  if rc == False:
    print("WARNING!!! prepare_ticker_df_monitor(): put_df_into_list() failed")
    return False

  #print("DS length:", len(row_list), "\nDS row:", row_list)
  dataset_df.loc[len(dataset_df)] = row_list
  #print(dataset_df)
  return True


def get_model_by_ticker_from_stat(ticker, pred_stat_data):
  print("get_model_by_ticker_from_stat() called with", ticker)
  m_list = []
  for index, row in pred_stat_data.iterrows():
    model = pred_stat_data.at[index, 'Model']
    ticker_in_file = pred_stat_data.at[index, 'Ticker']
    #print("Model:", model)
    f = os.path.join(MODELS_DIR, model)
    # Check general models
    if ticker_in_file.find(GENERAL_TICKER_PREFIX) == 0 and os.path.isfile(f) and f.find(".zip") != -1:
      #print(ticker_in_file, model)
      m_list.append(f)
      continue

    if ticker_in_file.find(ticker) == 0 and len(ticker_in_file) == len(ticker) and os.path.isfile(f) and f.find(".zip") != -1:
      #print(ticker_in_file, model)
      m_list.append(f)
      continue

  print("Models for", ticker, ":", m_list)
  return m_list 

def get_model_by_ticker(ticker):
  m_list = []
  for filename in os.listdir(MODELS_DIR):
    f = os.path.join(MODELS_DIR, filename)
    if os.path.isfile(f) and f.find(".zip") != -1 and f.find(ticker) != -1:
      #print("Model found:", f)
      m_list.append(f)

  print("Models for", ticker, ":", m_list)
  return m_list 

def predict_monitor_df(ticker, dataset_df, model_name):
  MODEL_MAX = MODELS_DIR + "/" + "model_max_full"
  MODEL_MIN = MODELS_DIR + "/" + "model_min_full"

  MODEL = get_MODEL_by_model_name(model_name)
  print("Model name:", model_name, "Model:", MODEL)
  shutil.rmtree(MODEL_MAX, ignore_errors=True)
  shutil.rmtree(MODEL_MIN, ignore_errors=True)
               
  shutil.unpack_archive(model_name, MODELS_DIR, 'zip')

  if MODEL.mode == "L":
    #LONG models use this name
    load_model = MODEL_MIN
  else:
    #SHORT models use this name
    load_model = MODEL_MAX

  model = tf.keras.models.load_model(load_model)
  #model.summary() 

  LEARN_COLUMNS = get_learn_columns_clean_full(MODEL)
  print("LEARN_COLUMNS:\n", LEARN_COLUMNS, "\nLen=", len(LEARN_COLUMNS))  

  # Check the prediction
  new_cleaned_df  = dataset_df[LEARN_COLUMNS].copy()
  new_test_features = np.array(new_cleaned_df)
  new_test_predictions = model.predict(new_test_features, batch_size=BATCH_SIZE)
  print("PREDICTION:\n", new_test_predictions)
  dataset_df[['pred']] = new_test_predictions
  return dataset_df.pred[0]



# return dataset for prediction processing
def load_tickers_bunch_1d(ticker_list, start_date, INTRA_MODE, pred_stat_data, stock, end_period, model_name, DRAW_FLAG=False, volume_min = -1 ):
  print("load_tickers_bunch_1d() caled with:", start_date, INTRA_MODE, stock, end_period, model_name, DRAW_FLAG)
  print("Start:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
  #print(pred_stat_data)
  runtime_df = pd.DataFrame(columns=RUNTIME_PRED_STAT_COLUMNS)

  model_name_inst = "MODELS\\" + model_name
  print("Model example:", model_name_inst)
  MODEL = get_MODEL_by_model_name(model_name)
  MODEL.t_days = 0
  MODEL.time_i = INTRA_MODE

  # Now separate tickers list provided
  #ticker_list = pred_stat_data['Ticker'].unique()
  DATASET_COLUMNS = get_dataset_columns(MODEL)  
  print(DATASET_COLUMNS)
  print("DS length:", len(DATASET_COLUMNS))

  if stock == NY_STOCK:
    index_ticker = SP500_INDEX
  elif stock == CRYPT_STOCK:
    index_ticker = CRYPT_INDEX
  elif stock == MOEX_STOCK:
    index_ticker = MOEX_INDEX
  else:
    print("Error: undefined stock: ", stock)  
    return None

  print("Index ticker:", index_ticker)  

  while(True):
    index_data = load_ticker_data(index_ticker, start_date, INTRA_MODE, stock, end_period)
    if(index_data is None):
      print("ERROR:", index_ticker, "NOT loaded")   
      print("Sleeping", 9, "sec...")
      time.sleep(9)
    else:
      break

  print(index_ticker, "loaded")
  print(index_data.tail(10), "\nSize:", len(index_data))
  ticker_counter = 0
  dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
  for ticker in ticker_list:
    ticker_counter+=1

    if index_data is None or (MODEL.time_i != '1d' and ticker_counter%20 == 0):
      print("Reloading index ticker:", index_ticker)  
      index_data = load_ticker_data(index_ticker, start_date, INTRA_MODE, stock, end_period)
      if index_data is None:
        print("ERROR:", index_ticker, "NOT loaded")   
        print("Sleeping", 10, "sec...")
        time.sleep(10)
        continue
 
    ticker_df = pd.DataFrame(columns=DATASET_COLUMNS)
    print(ticker_counter, ticker, "loaded")
    ticker_data = load_ticker_data(ticker, start_date, INTRA_MODE, stock, end_period)
    if(ticker_data is None or len(ticker_data) == 0):
      print("ERROR:", ticker, "NOT loaded")   
      continue
    #elif INTRA_MODE == "1d":
      #print(ticker_data.tail(10))

    rc = prepare_ticker_df_monitor(ticker, ticker_data, index_data, MODEL, ticker_df)
    if rc == False:
      print("ERROR: prepare_ticker_df_monitor()")
      continue

    #print("PROD: ##### ", ticker, model_name_inst, " #####")   
    pred = 0 # Prediction will be setup after full dataset is prepared
    #pred = predict_monitor_df(ticker, dataset_df, model_name)

    ### Collect statistics ###
    row_list = []
    row_list.append(ticker)
    row_list.append(model_name_inst)
    row_list.append(pred)
    entry_price  = get_entry_price(ticker_data, MODEL)
    target_price = get_target_price(entry_price, MODEL)
    end_date_in  = get_end_date(ticker_data, MODEL, stock)
    row_list.append(entry_price)
    row_list.append(target_price)
    row_list.append(end_date_in)
    vol_avg = get_last_volume_per_average(ticker_data)    
    row_list.append(vol_avg)
    vol_pre = get_previous_volume(ticker_data)
    row_list.append(vol_pre)
    vol_mln = get_last_volume_mln(ticker_data)
    row_list.append(vol_mln)
    runtime_df.loc[len(runtime_df)] = row_list
    ### Collect statistics ###
    #print("PROD: Ticker dataset:\n", ticker_df)
    #dataset_df = dataset_df.append(ticker_df)
    dataset_df = pd.concat([dataset_df, ticker_df], axis=0)    

  ### Here we can add additional columns like Ticker id ###
  dataset_df.reset_index(inplace=True, drop=True) 
  post_proc_df_add_clmns(dataset_df, MODEL)

  print("Prepared df:\n", dataset_df)
  predict_monitor_df(ticker, dataset_df, model_name_inst)
  #print("PROD: Tickers dataset:\n", dataset_df)

  runtime_df = pd.merge(runtime_df, dataset_df[['File', 'pred']], left_on='Ticker', right_on='File')
  runtime_df['Pred'] = runtime_df['pred']
  runtime_df = runtime_df.drop('File', axis=1)
  runtime_df = runtime_df.drop('pred', axis=1)
  #print("Runtime dataset:\n", runtime_df)

  if MODEL.time_i == '1d' or MODEL.time_i == '24':
    # Draw data
    if DRAW_FLAG == True and ( model_name.find("model_ds_01d_MNL_L2.0_P13_T01_N13_E1500_T.zip") >= 0 or model_name.find("model_ds_01d_MXS_S3.0_P10_T01_N10_E5000_T.zip") >= 0 ):
      PRED_TO_SHOW = 0.22
      if model_name.find("model_ds_01d_MNL_L2.0_P13_T01_N13_E1500_T.zip") >= 0:
        PRED_TO_SHOW = 0.28
      if model_name.find("model_ds_01d_MXS_S3.0_P10_T01_N10_E5000_T.zip") >= 0:
        PRED_TO_SHOW = 0.15

      if stock == NY_STOCK:
        PRED_TO_SHOW = 0.25
      filtered_df = runtime_df[(runtime_df['Pred'] >= PRED_TO_SHOW)]
      end_date = str(end_date_in)
      for id, row_id in filtered_df.iterrows():
        ticker = filtered_df.at[id,'Ticker']
        pred = filtered_df.at[id,'Pred']
        pdf_file = str(end_date) + "_" + str(pred) + "_" + ticker + "_"+ MODEL.mode + str(MODEL.percent) + ".pdf"
        pdf_file = pdf_file.replace(":", "_")
        txt = ticker + str(pred)
        draw_ticker_charts(ticker, stock, txt, start_date, pdf_file, model_name, PRED_TO_SHOW, pred, end_date)  

    runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)

    if volume_min == -1:
      print("Volume is ignored\n")
    else:
      runtime_df = runtime_df[(runtime_df['Vol_MLN_RUB'] >= volume_min)]
      print("Volume MIN:", volume_min)

    print("Final stat:\n", runtime_df.to_string())

    RUNTIME_STAT_FILE = stock + "_" + end_date_in.replace(":", "_") + ".csv"
    runtime_df.to_csv(RUNTIME_STAT_FILE, index=False)

    #filtered_df = runtime_df[(runtime_df['Pred'] >= 0.15) & (runtime_df['Vol_avg'] > 1) & (runtime_df['Vol_pre'] > 1)] 
    #print("Filtered:\n", filtered_df.to_string()) 
 
    #runtime_df = runtime_df.sort_index().sort_values('Vol_avg', kind='mergesort', ascending=False)
    #print("TOP30 Last per average volume:\n", runtime_df.head(30).to_string())
    
    #runtime_df = runtime_df.sort_index().sort_values('Vol_pre', kind='mergesort', ascending=False)
    #print("TOP30 previous volume:\n", runtime_df.head(30).to_string())

    #show_filtered_df(runtime_df)

    out_str = ""
    if MODEL.mode == "L":

      if model_name.find("model_ds_01d_MNL_L2.0_P13_T01_N13_E1500_T.zip") >= 0: 
        out_str = "\n=== " + model_name + " ===\n" 
        # Vol_avg ordered
        #out_str+= "\n== Filtered by Volume ==\n"
        #filtered_df = runtime_df.sort_index().sort_values('Vol_avg', kind='mergesort', ascending=False).head(20)
        #out_str+= filtered_df.to_string()

        out_str+= "\n???OLD AlgConf(result=151.07; sort_clmn=Pred, filt_clmn=Pred, min_val=0.3, max_val=0.95, PRED_MIN=0.3, PRED_MAX=0.95, topX=5, mode=PART)\n"
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.3) & (runtime_df['Pred'] <= 0.95)].head(7) 
        out_str+= filtered_df.to_string()

        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.48) & (runtime_df['Pred'] <= 1)].head(5) 
        out_str+= "\nAlgoConf(result= 25.76; sort_clmn=Pred, filt_clmn=Pred, min_val=0.48, max_val=1, PRED=0.32-1, cases=23, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MNL_L6.0_P10_T01_N10_E1500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.3) & (runtime_df['Pred'] <= 1)].head(7)  
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+= "\nAlgoConf(result= 23.56; sort_clmn=Pred, filt_clmn=Pred, min_val=0.3, max_val=1.0, PRED=0.3-1.0, cases=16, topX=5, mode=PART)\n"
        out_str+= filtered_df.to_string()


      if model_name.find("model_ds_01d_MXL_L3.0_P10_T01_N10_E5000_T_E20000.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.22) & (runtime_df['Pred'] <= 0.91)].head(7)  
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+="AlgoConf(result= 20.46; sort_clmn=Pred, filt_clmn=Pred, min_val=0.22, max_val=0.91, PRED=0.22-0.91, cases=96, topX=5, mode=PART)\n"
        out_str+= filtered_df.to_string()

        #runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        #filtered_df = runtime_df[(runtime_df['Pred'] >= 0.23) & (runtime_df['Pred'] <= 1) & (runtime_df['Vol_avg'] >= 0.5) & (runtime_df['Vol_avg'] <= 50) ].head(7)  
        #out_str+="\nAlgoConf(result=139.06; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.5, max_val=50.0, PRED_MIN=0.23, PRED_MAX=1.0, topX=5, mode=ALL_DEP)\n"
        #out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MXL_L4.0_P10_T01_N10_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False).head(7)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+= runtime_df.to_string()
        out_str+="\nAlgoConf(result= 43.8; sort_clmn=Pred, filt_clmn=Vol_pre, min_val=1.41, max_val=45.85, PRED=0.22-0.315, cases=8, topX=5, mode=ALL_DEP)\n"
        runtime_df = runtime_df.sort_index().sort_values('Vol_pre', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.22) & (runtime_df['Pred'] < 0.315) & (runtime_df['Vol_pre'] >= 1.41) & (runtime_df['Vol_pre'] <= 45.85)].head(5)  
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MNL_L5.0_P10_T01_N10_E1500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Vol_avg', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.1) & (runtime_df['Pred'] <= 0.47) & (runtime_df['Vol_avg'] >= 0.31) & (runtime_df['Vol_avg'] <= 17.4)] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+= "\nAlgoConf(sort_clmn='Pred', filt_clmn='Vol_avg', min_val=0.31, max_val=17.400000000000816, PRED_MIN=0.1, PRED_MAX=0.46999999999999953, topX=5, result=123.71711950939269, mode=<ACCUM_DELTA_MODE.ALL_DEP: 2>)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MXL_L3.0_P13_T01_N13_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.35) & (runtime_df['Pred'] < 0.645)] 
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result= 95.21; sort_clmn=Pred, filt_clmn=Pred, min_val=0.35, max_val=0.645, PRED=0.35-0.645, cases=128, topX=5, mode=ALL_DEP)"        
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.35) & (runtime_df['Pred'] < 0.645) & (runtime_df['Vol_avg'] >= 0.66) & (runtime_df['Vol_avg'] <= 2.85) ] 
        out_str += "\nAlgoConf(result=164.66; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.66, max_val=2.85, PRED=0.35-0.645, cases=78, topX=5, mode=ALL_DEP)"
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_MNL_L7.0_P10_T01_N10_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.145) & (runtime_df['Pred'] <= 0.815) & (runtime_df['Vol_pre'] >= 0.86) & (runtime_df['Vol_pre'] <= 9.05)] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+= "\nAlgoConf(result=173.15; sort_clmn=Pred, filt_clmn=Vol_pre, min_val=0.86, max_val=9.05, PRED_MIN=0.145, PRED_MAX=0.815, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MLI_L3.0_P20_T01_N20_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.125) & (runtime_df['Vol_avg'] >= 0.11) & (runtime_df['Vol_avg'] <= 9.15)] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+= "\nAlgoConf(result= 34.69; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.11, max_val=9.15, PRED_MIN=0.125, PRED_MAX=0.965, topX=5, mode=PART)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MLI_L2.0_P20_T01_N20_E40000_T_E20000.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(7).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.1) & (runtime_df['Vol_avg'] >= 2.61) & (runtime_df['Vol_avg'] <= 14.6)] 
        out_str+= "\nAlgoConf(result= 58.93; sort_clmn=Vol_avg, filt_clmn=Vol_avg, min_val=2.61, max_val=14.6, PRED=0.1-1, cases=27, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MXL_L8.0_P10_T01_N10_E15000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(7).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.16) & (runtime_df['Pred'] <= 0.47) & (runtime_df['Vol_pre'] >= 0.56) & (runtime_df['Vol_pre'] <= 40.6)] 
        out_str+= "\nAlgoConf(result=105.49; sort_clmn=Vol_pre, filt_clmn=Vol_pre, min_val=0.56, max_val=40.6, PRED=0.16-0.47, cases=15, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L2.0_P10_T01_N10_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.29) & (runtime_df['Pred'] <= 1)].head(5) 
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=9169.54; sort_clmn=Pred, filt_clmn=Pred, min_val=0.29, max_val=1, PRED=0.135-1, cases=1005, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.335) & (runtime_df['Pred'] < 0.995) ].head(5) 
        out_str += "\nAlgoConf(result=516.48; sort_clmn=Pred, filt_clmn=Pred, min_val=0.335, max_val=0.995, PRED=0.335-0.995, cases=649, topX=5, mode=ALL_DEP) - Max 715.958 at 11:00\n"        
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L3.0_P13_T01_N13_E3500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.24) & (runtime_df['Pred'] <= 1)].head(5) 
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=8674.63; sort_clmn=Pred, filt_clmn=Pred, min_val=0.24, max_val=1, PRED=0.1-1, cases=873, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.14) & (runtime_df['Pred'] <= 0.995) & (runtime_df['Vol_avg'] >= 4.26) & (runtime_df['Vol_avg'] <= 8.15) ].head(5) 
        out_str += "\nAlgoConf(result=637.49; sort_clmn=Vol_avg, filt_clmn=Vol_avg, min_val=4.26, max_val=8.15, PRED=0.14-0.995, cases=347, topX=5, mode=ALL_DEP) - Max 704.161 at 16:50\n"        
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L3.0_P10_T01_N10_E3500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.17) & (runtime_df['Pred'] <= 1)].head(5) 
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=3441.31; sort_clmn=Pred, filt_clmn=Pred, min_val=0.17, max_val=1, PRED=0.1-1, cases=1044, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.22) & (runtime_df['Pred'] <= 0.975) ].head(5) 
        out_str += "\nAlgoConf(result=441.36; sort_clmn=Pred, filt_clmn=Pred, min_val=0.22, max_val=0.975, PRED=0.22-0.975, cases=925, topX=5, mode=PART) - Max 789.601 at 10:50\n"        
        out_str += filtered_df.to_string()

      # Active models #
      if model_name.find("model_ds_01d_ML12306_L3.0_P13_T01_N13_ID_E1500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.25) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(PRED=0.25-1)\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.32) & (runtime_df['Pred'] <= 0.745) & (runtime_df['Vol_avg'] >= 0.36) & (runtime_df['Vol_avg'] <= 8.15)].head(5)
        out_str += "\nAlgoConf(result=1234.64; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.36, max_val=8.15, PRED=0.32-0.745, cases=268, topX=5, mode=ALL_DEP) - Max 1716.432 at 13:40\n"
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L6.0_P13_T01_N13_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.1) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=46837.72; sort_clmn=Pred, filt_clmn=Pred, min_val=0.1, max_val=1, PRED=0.1-1, cases=891, topX=5, mode=ALL_DEP\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.205) & (runtime_df['Pred'] <= 0.995) & (runtime_df['Vol_pre'] >= 0.21) & (runtime_df['Vol_pre'] <= 37.55)].head(5)
        out_str += "\nAlgoConf(result=731.84; sort_clmn=Pred, filt_clmn=Vol_pre, min_val=0.21, max_val=37.55, PRED=0.205-0.995, cases=302, topX=5, mode=ALL_DEP) - Max 875.541 at 16:50\n"
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L3.0_P13_T01_N13_ID_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.19) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=84542.02; sort_clmn=Pred, filt_clmn=Pred, min_val=0.19, max_val=1, PRED=0.1-1, cases=1010, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.3) & (runtime_df['Pred'] <= 1)].head(7)
        out_str += "\nAlgoConf(result=134.46; sort_clmn=Pred, filt_clmn=Pred, min_val=0.3, max_val=1, PRED=0.3-1, cases=199, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_L3.0_P09_T01_N09_ID_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.24) & (runtime_df['Pred'] <= 0.92)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\nAlgoConf(result=1185.02; sort_clmn=Pred, filt_clmn=Pred, min_val=0.24, max_val=0.92, PRED=0.24-0.92, cases=770, topX=5, mode=ALL_DEP) - Max 1786.971 at 16:50\n"
        out_str += filtered_df.to_string()

      if(model_name.find("model_ds_01d_ML12405_L3.0_P13_T01_N13_ID_E10000_T.zip") >= 0 or 
         model_name.find("model_ds_01d_ML12406_L3.0_P13_T01_N13_ID_E10000_T.zip") >=0 ): 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.19) & (runtime_df['Pred'] <= 1)].head(7)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\n!!!AlgoConf(result=???; sort_clmn=Pred, filt_clmn=Pred, min_val=0.19, max_val=1, PRED=0.1-1, cases=1010, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()

      if(model_name.find("model_ds_01d_ML1240816_L3.0_P13_T01_N13_ID_E10000_T.zip") >= 0): 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.225) & (runtime_df['Pred'] <= 1)].head(7)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += "\n!!!94.85435267531177 - TestStat(pred=0.225, cl_counter=61, hl_counter=60, lo_counter=104, nl_counter=124, all_cases=228, cl_lo=0.587, clhl_lo=1.163, not_looses_lo=1.192, delta=287.34999999999997)\n"
        out_str += filtered_df.to_string()


      if model_name.find("model_ds_01d_ML3_2204_240816_L3.0_P10_T01_N10_ID_E1000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.22) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += runtime_df.head(5).to_string()
        out_str += "\n AlgoConf(result= 19.14; sort_clmn=Pred, filt_clmn=Pred, min_val=0.22, max_val=1, PRED=0.22-1, cases=28, topX=5, mode=ALL_DEP) - LONG 19.14/12.31\n"
        out_str += filtered_df.to_string()

      if model_name.find("model_ds_01d_ML3240816TR_L2.0_P10_T01_N10_ID_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.4) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += runtime_df.head(5).to_string()
        out_str += "\nAlgoConf(result= 13.57; sort_clmn=Pred, filt_clmn=Pred, min_val=0.4, max_val=1, PRED=0.4-1, cases=11, topX=5, mode=ALL_DEP)\n"
        out_str += filtered_df.to_string()
        
      if model_name.find("model_ds_01d_ML4_240816_L2.0_P10_T01_N10_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.35) & (runtime_df['Pred'] <= 1)].head(5)     
        out_str = "\n=== " + model_name + " ===\n"
        out_str += runtime_df.head(5).to_string()
        out_str += "\n!!!! 0.35 AlgoConf(result= 25.49; sort_clmn=Pred, filt_clmn=Pred, min_val=0.38, max_val=1, PRED=0.35-1, cases=50, topX=5, mode=PART);ALL_DEP=28.59)\n"
        out_str += filtered_df.to_string()

    else:

      if model_name.find("model_ds_01d_MNS_S1.5_P10_T01_N10_E1500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.46)] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+="AlgoConf(result= 14.22; sort_clmn=Pred, filt_clmn=Pred, min_val=0.46, max_val=1, PRED_MIN=0.24, PRED_MAX=1, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MNS_S2.0_P10_T01_N10_E1500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.445) & (runtime_df['Pred'] <= 1)].head(7) 
        out_str+="\nAlgoConf(result= 14.62; sort_clmn=Pred, filt_clmn=Pred, min_val=0.445, max_val=1, PRED=0.1-1, cases=22, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MNS_S4.0_P07_T01_N07_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.2) & (runtime_df['Pred'] <= 0.9) & (runtime_df['Vol_avg'] >= 0.66) & (runtime_df['Vol_avg'] <= 50)] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str+="AlgoConf(result= 14.87; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.66, max_val=50, PRED_MIN=0.2, PRED_MAX=0.9, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MXS_S3.0_P10_T01_N10_E5000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.2) & (runtime_df['Pred'] <= 0.92)] 
        out_str+="\nAlgoConf(result= 32.54; sort_clmn=Pred, filt_clmn=Pred, min_val=0.2, max_val=0.92, PRED=0.2-0.92, cases=60, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MNS_S3.0_P10_T01_N10_E1500_T.zip") >= 0: 
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.32) ] 
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        out_str+= "\nAlgoConf(result= 9.13; sort_clmn=Pred, filt_clmn=Pred, min_val=0.32, max_val=1, PRED_MIN=0.1, PRED_MAX=1, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_S3.0_P13_T01_N13_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.1) & (runtime_df['Pred'] <= 1) & (runtime_df['Vol_avg'] >= 7.46) ] 
        out_str+="\nAlgoConf(result=126.72; sort_clmn=Vol_avg, filt_clmn=Vol_avg, min_val=7.46, max_val=50, PRED=0.1-1, cases=42, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_S2.0_P13_T01_N13_ID_E3500_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.655) & (runtime_df['Pred'] <= 0.995)] 
        out_str+="\nAlgoConf(result=268.42; sort_clmn=Pred, filt_clmn=Pred, min_val=0.655, max_val=0.995, PRED=0.655-0.995, cases=234, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_ML12306_S3.0_P10_T01_N10_ID_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.3) & (runtime_df['Pred'] <= 1)] 
        out_str+="\nAlgoConf(result= 26.66; sort_clmn=Pred, filt_clmn=Pred, min_val=0.3, max_val=1, PRED=0.3-1, cases=40, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MS22407_S3.0_P13_T01_N13_ID_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.25) & (runtime_df['Pred'] <= 1)] 
        out_str+="\n!!! 0.25 AlgoConf(result= 8.05; sort_clmn=Pred, filt_clmn=Pred, min_val=0.25, max_val=1, PRED=0.25-1, cases=29, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_MS2240816_S3.0_P13_T01_N13_ID_E10000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.25) & (runtime_df['Pred'] <= 1)] 
        out_str+="\n???AlgoConf(result= 8.05; sort_clmn=Pred, filt_clmn=Pred, min_val=0.25, max_val=1, PRED=0.25-1, cases=29, topX=5, mode=ALL_DEP)\n"
        out_str+= filtered_df.to_string()

      if model_name.find("model_ds_01d_ML3240816_S2.0_P10_T01_N10_ID_E1000_T.zip") >= 0: 
        runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
        out_str = "\n=== " + model_name + " ===\n" 
        #out_str += runtime_df.head(5).to_string()
        filtered_df = runtime_df[(runtime_df['Pred'] >= 0.15) & (runtime_df['Pred'] <= 1)].head(7) 
        out_str+="AlgoConf(result= 19.44; sort_clmn=Pred, filt_clmn=Pred, min_val=0.15, max_val=1, PRED=0.15-1, cases=110, topX=5, mode=PART)\n"
        out_str+= filtered_df.to_string()



###### CRYPTO #######
    if model_name.find("model_ds_01d_CRY2_L4.5_P15_T01_N15_E10000_T.zip") >= 0: 
      runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
      out_str = "\n=== " + model_name + " ===\n" 
      out_str+= "\n== TOP20 ==\n"
      out_str+= runtime_df.head(20).to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.285 ) & (runtime_df['Vol_avg'] >= 0.9) & (runtime_df['Vol_avg'] <= 50)] 
      out_str+="\nAlgoConf(result=153.73; sort_clmn=Pred, filt_clmn=Vol_avg, min_val=0.9, max_val=16.15, PRED=0.285-1.0, cases=110, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.head(10).to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.285 ) & (runtime_df['Vol_pre'] >= 2.76) & (runtime_df['Vol_avg'] <= 25.5)] 
      out_str+="\nAlgoConf(result=300.17; sort_clmn=Vol_pre, filt_clmn=Vol_pre, min_val=2.76, max_val=25.05, PRED=0.1-1, cases=134, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.head(10).to_string()


    if model_name.find("model_ds_01d_CRY2_S4.5_P15_T01_N15_E10000_T.zip") >= 0: 
      runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
      out_str = "\n=== " + model_name + " ===\n" 
      out_str+= "\n== TOP20 ==\n"
      out_str+= runtime_df.head(20).to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.825 ) & (runtime_df['Pred'] <= 0.995 )] 
      out_str+="\nAlgoConf(result= 99.18; sort_clmn=Pred, filt_clmn=Pred, min_val=0.825, max_val=0.995, PRED=0.825-0.995, cases=79, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.to_string()


    if model_name.find("model_ds_01d_CRY6_L9.5_P15_T01_N15_E10000_T.zip") >= 0: 
      runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
      out_str = "\n=== " + model_name + " ===\n" 
      out_str+= "\n== TOP10 ==\n"
      out_str+= runtime_df.head(10).to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.255 ) & (runtime_df['Pred'] <= 0.76)] 
      out_str+="\nAlgoConf(result=121.84; sort_clmn=Pred, filt_clmn=Pred, min_val=0.255, max_val=0.76, PRED=0.255-0.76, cases=109, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.255 ) & (runtime_df['Pred'] <= 0.76) & (runtime_df['Vol_pre'] >= 0.86) & (runtime_df['Vol_pre'] <= 50)] 
      out_str+="\nAlgoConf(result=270.15; sort_clmn=Vol_pre, filt_clmn=Vol_pre, min_val=0.86, max_val=50, PRED=0.255-0.76, cases=79, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.to_string()

    if model_name.find("model_ds_01d_CRY6_S4.5_P15_T01_N15_E10000_T.zip") >= 0: 
      runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
      out_str = "\n=== " + model_name + " ===\n" 
      out_str+= "\n== TOP10 ==\n"
      out_str+= runtime_df.head(10).to_string()

      filtered_df = runtime_df[(runtime_df['Pred'] >= 0.895 ) & (runtime_df['Pred'] <= 0.985) ] 
      out_str+="\nAlgoConf(result=148.21; sort_clmn=Pred, filt_clmn=Pred, min_val=0.895, max_val=0.985, PRED=0.895-0.985, cases=51, topX=5, mode=ALL_DEP)\n"
      out_str+= filtered_df.to_string()

    # Handle common stat file #
    hadle_stat_data_for_ticker_df(runtime_df)
    ###########################
 
    print("RUNTIME_STAT_FILE:", RUNTIME_STAT_FILE)

  print(out_str)
  print("load_tickers_bunch_1d() end")
  print("Finish:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  return out_str
