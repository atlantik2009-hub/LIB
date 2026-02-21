import sys
import pandas as pd
import numpy as np
import time
import random

from datetime import date
from datetime import datetime, timedelta

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.dates as mdates

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shutil
import yfinance as yf
from yahoofinancials import YahooFinancials
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF

from csv import writer

SP500_FILE = "sp500_new.csv"


COLUMNS_LEARN_LIST = ['Open0','Open1','Open2','Open3','Open4','Open5','Open6',
'High0','High1','High2','High3','High4','High5','High6',
'Low0','Low1','Low2','Low3','Low4','Low5','Low6',
'Close0', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5', 'Close6', 
'Volume0', 'Volume1','Volume2','Volume3','Volume4','Volume5','Volume6',
'Sp0', 'Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6']

COLUMNS_YAHOO_LIST = ['Date','Open','High','Low','Close','Adj Close','Volume']

def get_short_model_name(str):
  id = str.find("_L")
  if id == -1:
    id = str.find("_S")
    if id == -1:
      print("!!!Warning: short model name NOT found!!!")
      return ""
  
  return str[id+1:-4]

def update_sp500_file():
  #print("S&P500 read from the file.")
  # Read SP500 data
  sp500_data = pd.read_csv( SP500_FILE, names=COLUMNS_YAHOO_LIST, index_col ='Date')
  #print(sp500_data.tail(3))

  st_date = sp500_data.index[len(sp500_data)-1]
  en_date = date.today()
  #print("S&P500 read online:", st_date, en_date)
  sp500_data_new = yf.download("^GSPC", start=st_date, end=en_date, progress=False, show_errors=False, threads=False)  
  if len(sp500_data_new) <= 1:
    return
 
  sp500_data_new = sp500_data_new.drop(st_date)
  sp500_data_new = sp500_data_new.round(decimals = 6)
  print(sp500_data_new)
  sp500_data_new.to_csv(SP500_FILE, mode='a', header=False)
  print("S&P500 index file updated.")  
             
def get_start_pred(TEST_MODE, PERCENTAGE):
  if TEST_MODE == "L":
    START_PRED = 0.1
  else: 
    START_PRED = 0.1

  if PERCENTAGE == 6 or PERCENTAGE == 7: 
    START_PRED = 0.1
  if PERCENTAGE >= 8:
    START_PRED = 0.05   

  return START_PRED

# Return 1 in case of success in rc[0], rc[1] stores mode "S"/"L", rc[2] stores percent
# Obsolete to be removed
def get_model_params_old(str):
  print("Model:", str)
  short_tmplates = ["_S", "_short"]
  long_tmplates = ["_L", "_long"]
  found = False
  mode = ""
  for elem in short_tmplates:
    val = str.find(elem)
    if val != -1:
      start_id = val+len(elem)
      end_id = str[start_id:].find("_")
      #print(str[start_id:], start_id, end_id)
      percent = int(str[start_id: start_id+end_id])      
      if percent < 3:
        print("Impossible to specify model parameters", str)
        return (-1, 0, 0, 0)       
      mode = "S"
      start_pred = get_start_pred(mode, percent)
      return (1, mode, percent, start_pred)

  for elem in long_tmplates:
    val = str.find(elem)
    if val != -1:    
      start_id = val+len(elem)
      end_id = str[start_id:].find("_")
      #print(str[start_id:], start_id, end_id)
      percent = int(str[start_id: start_id+end_id])
      if percent < 3:
        print("Impossible to specify model parameters", str)
        return (-1, 0, 0, 0)       
      mode = "L"
      start_pred = get_start_pred(mode, percent)
      return (1, mode, percent, start_pred)
        
  print("Impossible to specify model parameters", str)
  return (-1, 0, 0, 0)



def find_between(str, start_tag, end_tag):
  id = -1
  rc = ""
  val = str.find(start_tag)
  if val != -1:
    start_id = val+len(start_tag)
    end_id = str[start_id:].find(end_tag)
    if end_id != -1:
      rc = str[start_id: start_id+end_id]
      id = start_id 

  return id, rc

# Return 1 in case of success in rc[0], rc[1] stores mode "S"/"L", rc[2] stores percent
# For example model_ds3_LASR_L4_T7_N7_E1500.zip
def get_model_params(str):
  print("Model:", str)
  short_tmplates = ["_S", "_short"]
  long_tmplates = ["_L", "_long"]
  ext_tmplt = "_T7"
  found = False
  mode = ""
  percent = 0
  for elem in short_tmplates:
    rc = find_between(str, elem, ext_tmplt)
    start_id = rc[0]
    percent_str = rc[1]
    if len(percent_str) > 0:
      if percent_str.isdigit():
        percent = int(percent_str)
      else:
        rc = find_between(str[start_id:], elem, ext_tmplt)
        start_id = rc[0]
        percent_str = rc[1]
        if percent_str.isdigit():
          percent = int(percent_str)
   
      if percent >= 3:
        mode = "S"
        start_pred = get_start_pred(mode, percent)
        return (1, mode, percent, start_pred)

  for elem in long_tmplates:
    rc = find_between(str, elem, ext_tmplt)
    start_id = rc[0]
    percent_str = rc[1]
    if len(percent_str) > 0:
      if percent_str.isdigit():
        percent = int(percent_str)
      else:
        rc = find_between(str[start_id:], elem, ext_tmplt)
        start_id = rc[0]
        percent_str = rc[1]
        if percent_str.isdigit():
          percent = int(percent_str)
   
      if percent >= 3:
        mode = "L"
        start_pred = get_start_pred(mode, percent)
        return (1, mode, percent, start_pred)
        
  print("Impossible to specify model parameters", str)
  return (-1, 0, 0, 0)

def hist_data_loader(ticker_list):

  ### Constant section ###
  #start_date = '2010-01-01'
  start_date = '2021-11-01'
  end_date = date.today()
  test_archive = 'history_data'
  ########################

  print(ticker_list)
  print("Number of tickers:", len(ticker_list))

  if os.path.exists(test_archive):
    shutil.rmtree(test_archive, ignore_errors=True) 
  os.mkdir(test_archive)

  counter = 0
  for ticker in ticker_list:

    if counter % 1 == 0:
      print(counter, ticker)

    try:
      ticker_data_df = yf.download(ticker, start=start_date, end=end_date, progress=False, show_errors=False, threads=False)
    except ValueError: 
      print("Exception catched on ticker", ticker)
      continue;
    #else:
      # Do nothing

    if len(ticker_data_df) < 7:
      print("Incorrect data: Not enough. Ticker", ticker)
      continue;

    bad_entry_counter = 0
    for id, row_id in ticker_data_df.iterrows():
      if pd.isna(ticker_data_df.at[id,'Open']):
        bad_entry_counter = bad_entry_counter + 1
        ticker_data_df = ticker_data_df.drop(labels=id, axis=0)

    if bad_entry_counter > 0:  
      print("Bad entries", bad_entry_counter)

    file_name = test_archive + '/' + ticker + '.csv'
    ticker_data_df.to_csv(file_name) 

    counter = counter + 1
    # For debug purpose
    if counter == 0:
      print("LEN:", len(ticker_data_df))
      print(ticker_data_df)

#################################################
stat_test_df = pd.DataFrame(columns=['File', 'Date', 'Success_flag','Pred', 'First_day', 'Test_flag', 'CL1', 'HL1', 'CL2', 'HL2', 'CL3', 'HL3'])
result_stat_df = pd.DataFrame(columns=['Model', 'Percentage', 'Ticker', 'Price', 'Pred', 'Start_date', 'Close_EV', 'Close_HV', 'High_Low_EV', 'Looses', 'Cases', 'Accuracy', 'Looses_per', 
                                       'HL_suc1', 'HL_suc2', 'HL_suc3', 'CL_suc1', 'CL_suc2', 'CL_suc3', 'CL_los1', 'CL_los2', 'CL_los3'])
failed_test_df = pd.DataFrame(columns=['File', 'Type', 'Date',
                            "Open0","Open1","Open2","Open3","Open4","Open5","Open6","Open7","Open8","Open9",
                            "High0","High1","High2","High3","High4","High5","High6","High7","High8","High9",
                            "Low0","Low1","Low2","Low3","Low4","Low5","Low6","Low7","Low8","Low9",
                            'Close0', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5', 'Close6', 'Close7', 'Close8', 'Close9',
                            'Volume0', 'Volume1','Volume2','Volume3','Volume4','Volume5','Volume6', 'Volume7','Volume8','Volume9',
                            'Sp0', 'Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6', 'Sp7', 'Sp8', 'Sp9'
                            , 'Test_flag', 'Pred'
                                        ])

def test_ticker(model_name, ticker, test_value, predict_low, predict_high, entry_price, sp500_data, test_archive):
  shutil.rmtree("model_max_full", ignore_errors=True)
  shutil.rmtree("model_min_full", ignore_errors=True)
               
  shutil.unpack_archive(model_name, './', 'zip')

##### Load S&P500 data #####
  if len(sp500_data) == 0:
    print("S&P500 read from the file.")
    # Read SP500 data
    sp500_data = pd.read_csv( SP500_FILE, names=COLUMNS_YAHOO_LIST, index_col ='Date')
  #print(sp500_data.tail(10).to_string())
  #date_val = '2022-11-15'
  #print(date_val, "SP500 price:", sp500_data.loc[date_val]['Close'])
################################

  # TO THINK OF THIS
  AUTO_MODE = True
  if AUTO_MODE == True:
    print("AUTO_MODE enabled")
    rc = get_model_params(model_name)
    if rc[0] != 1:
      sys.exit("Impossible to specify parameters of the model.") 
  
    TEST_MODE = rc[1] 
    #PERCENTAGE = rc[2]

  if TEST_MODE == "L":
    #LONG models use this name
    load_model = "model_min_full"
  else:
    #SHORT models use this name
    load_model = "model_max_full"

  model = tf.keras.models.load_model(load_model)
  model.summary() 

  global stat_test_df
  global result_stat_df
  global failed_test_df
  ETALON_VALUE = 1 + test_value/100 # Should be like 1.06
  HALF_VALUE   = 1 + test_value/200 # Should be like 1.03
  # predict_value Should be like 0.3

  predict_range = str(predict_low) + '-' + str(predict_high)
  print("Test_mode:", TEST_MODE, "Model:", model_name, "Ticker:", ticker, ": ETALON_VALUE:", ETALON_VALUE, "PRED:", predict_range)

  FAILED_TC = False

  DEBUG_INDEX = 12505
  DEBUG_TICKER= ticker
  BATCH_SIZE = 2048

  file_name = test_archive + '/' + ticker + '.csv'

  if os.path.isfile(file_name):
    print("\n!!!!!!TIKER!!!!!", ticker)
  else:
    print('File doent exist', file_name)
    return

  hist_data = pd.read_csv( file_name, names=COLUMNS_YAHOO_LIST, skiprows=1)

  new_test_df = pd.DataFrame(columns=['File', 'Date',
                            "Open0","Open1","Open2","Open3","Open4","Open5","Open6","Open7","Open8","Open9",
                            "High0","High1","High2","High3","High4","High5","High6","High7","High8","High9",
                            "Low0","Low1","Low2","Low3","Low4","Low5","Low6","Low7","Low8","Low9",
                            'Close0', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5', 'Close6', 'Close7', 'Close8', 'Close9',
                            'Volume0', 'Volume1','Volume2','Volume3','Volume4','Volume5','Volume6', 'Volume7','Volume8','Volume9',
                            'Sp0', 'Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6', 'Sp7', 'Sp8', 'Sp9',
                            'Test_flag', 'Day_flag'
                                    ])
  
  print(ticker, "test set shape:", hist_data.shape)
  if(len(hist_data)<10):
    print("Ticker skipped due to not enough data")
    return

  #print( hist_data.head(5).to_string())
  counter =0
  xx = [1,2,3,4,5,6,7,8,9,10]

  date_list   = np.array(hist_data['Date'].tolist())
  open_list   = np.array(hist_data['Open'].tolist())
  high_list   = np.array(hist_data['High'].tolist())
  low_list    = np.array(hist_data['Low'].tolist())
  close_list  = np.array(hist_data['Close'].tolist())
  volume_list = np.array(hist_data['Volume'].tolist())

  print("LEN: ", len(close_list))
  #print("Close list")
  #print(close_list)

  # Select series by 10 elements 
  list_c = [None] * 10;   
  list_v = [None] * 10;
  list_d = [None] * 10;
  list_o = [0] * 10;
  list_h = [0] * 10;
  list_l = [0] * 10;
  for index in range(0, len(date_list) - 10):
    for i in range(0, 10):    
        list_c[i] = close_list[i + index]
        list_v[i] = volume_list[i + index]
        list_d[i] = date_list[i + index]
        list_o[i] = open_list[i + index]
        list_h[i] = high_list[i + index]
        list_l[i] = low_list[i + index]
    date_start = date_list[index]
    
    close_max  = max(list_h[0:7])    
    volume_max = max(list_v[0:7])

    # Setup real test flag
    real_flag = 0
    day_flag = 0 # store the first day when the target was atchived
    if TEST_MODE == "S":
      min_val = min(list_c[7:]) # Equals to list_c[7:10]
      low_min_val = min(list_l[7:])

      # 2 means that the target pattern happened by min value; otherwise it stores Close[6]/Close[9]. 
      # If it is less than 1 it means LOOSES. If it is more than 1 it means some pluse but the target price is NOT atchived 
      if list_c[6]/min_val >= ETALON_VALUE:
        real_flag = 2
      else:
        real_flag = list_c[6]/list_c[9]

      # Put the number of day/week after entry in which the target pattern happens(1,2,3 or 0 if pattern didn't happen)
      if list_c[6]/low_min_val >= ETALON_VALUE:
        if list_c[6]/list_l[7] >= ETALON_VALUE:
          day_flag = 1
        elif list_c[6]/list_l[8] >= ETALON_VALUE:
          day_flag = 2
        elif list_c[6]/list_l[9] >= ETALON_VALUE:
          day_flag = 3
        if index == DEBUG_INDEX and ticker == DEBUG_TICKER:
          print(index, "Entry price:", list_c[6], "Low min:", low_min_val, "Delta:", (list_c[6]/low_min_val))
          print(list_l[7:])
      else:
        day_flag = 0
    else:
      max_val = max(list_c[7:10])
      high_max_val = max(list_h[7:10])

      if max_val/list_c[6] >= ETALON_VALUE:
        real_flag = 2
      else:
        real_flag = list_c[9]/list_c[6]

      if high_max_val/list_c[6] >= ETALON_VALUE:
        if list_h[7]/list_c[6] >= ETALON_VALUE:
          day_flag = 1
        elif list_h[8]/list_c[6] >= ETALON_VALUE:
          day_flag = 2
        elif list_h[9]/list_c[6] >= ETALON_VALUE:
          day_flag = 3
        if index == DEBUG_INDEX and ticker == DEBUG_TICKER:
          print(index, "Entry price:", list_c[6], "High max:", high_max_val, "Delta:", (high_max_val/list_c[6]), "Day:", day_flag)
          print(list_h[7:])
      else:
        day_flag = 0
    
    # Format "2010-01-04 00:00:00-05:00" to be changed into "2010-01-04"
    if (len(list_d[0])>len("2010-01-04")):
      #print("Date format to be updated", list_d[0], "into", list_d[0][0:10])
      for elem_id in range(len(list_d)):
        list_d[elem_id] = list_d[elem_id][0:10]

    sp500_list = [sp500_data.loc[list_d[0]]['Close'], sp500_data.loc[list_d[1]]['Close'], sp500_data.loc[list_d[2]]['Close'], sp500_data.loc[list_d[3]]['Close'], sp500_data.loc[list_d[4]]['Close'],
                  sp500_data.loc[list_d[5]]['Close'], sp500_data.loc[list_d[6]]['Close']]
    sp500_max = float(max(sp500_list))

    sp500_list.append(sp500_data.loc[list_d[7]]['Close'])
    sp500_list.append(sp500_data.loc[list_d[8]]['Close'])
    sp500_list.append(sp500_data.loc[list_d[9]]['Close'])
    # Conver all values into float
    sp500_list = np.array(sp500_list,dtype=float)
    #print(sp500_list)
    new_entry = pd.DataFrame({'File': ticker, 'Date': date_start,
                                        'Open0': list_o[0]/close_max, 'Open1': list_o[1]/close_max,
                                        'Open2': list_o[2]/close_max, 'Open3': list_o[3]/close_max,
                                        'Open4': list_o[4]/close_max, 'Open5': list_o[5]/close_max,
                                        'Open6': list_o[6]/close_max, 'Open7': list_o[7]/close_max, 'Open8': list_o[8]/close_max, 'Open9': list_o[9]/close_max,
                                        'High0': list_h[0]/close_max, 'High1': list_h[1]/close_max,
                                        'High2': list_h[2]/close_max, 'High3': list_h[3]/close_max,
                                        'High4': list_h[4]/close_max, 'High5': list_h[5]/close_max,
                                        'High6': list_h[6]/close_max, 'High7': list_h[7]/close_max, 'High8': list_h[8]/close_max, 'High9': list_h[9]/close_max,
                                        'Low0': list_l[0]/close_max, 'Low1': list_l[1]/close_max,
                                        'Low2': list_l[2]/close_max, 'Low3': list_l[3]/close_max,
                                        'Low4': list_l[4]/close_max, 'Low5': list_l[5]/close_max,
                                        'Low6': list_l[6]/close_max, 'Low7': list_l[7]/close_max, 'Low8': list_l[8]/close_max, 'Low9': list_l[9]/close_max,                                      
                                        'Close0': list_c[0]/close_max, 'Close1': list_c[1]/close_max,
                                        'Close2': list_c[2]/close_max, 'Close3': list_c[3]/close_max,
                                        'Close4': list_c[4]/close_max, 'Close5': list_c[5]/close_max,
                                        'Close6': list_c[6]/close_max, 'Close7': list_c[7]/close_max, 'Close8': list_c[8]/close_max, 'Close9': list_c[9]/close_max,                                    
                                        'Volume0': list_v[0]/volume_max, 'Volume1': list_v[1]/volume_max,
                                        'Volume2': list_v[2]/volume_max, 'Volume3': list_v[3]/volume_max,
                                        'Volume4': list_v[4]/volume_max, 'Volume5': list_v[5]/volume_max,
                                        'Volume6': list_v[6]/volume_max, 'Volume7': list_v[7]/volume_max, 'Volume8': list_v[8]/volume_max, 'Volume9': list_v[9]/volume_max,
                                        'Sp0': sp500_list[0]/sp500_max, 'Sp1': sp500_list[1]/sp500_max, 'Sp2': sp500_list[2]/sp500_max,
                                        'Sp3': sp500_list[3]/sp500_max, 'Sp4': sp500_list[4]/sp500_max, 'Sp5': sp500_list[5]/sp500_max,
                                        'Sp6': sp500_list[6]/sp500_max, 'Sp7': sp500_list[7]/sp500_max, 'Sp8': sp500_list[8]/sp500_max,
                                        'Sp9': sp500_list[9]/sp500_max,
                                        'Test_flag': real_flag, 'Day_flag': day_flag
                                      }, index=[0])
    new_test_df = pd.concat([new_test_df, new_entry], ignore_index=True)
    
    if index == DEBUG_INDEX and ticker == DEBUG_TICKER:
      print("+++++ DEBUG INFO: Index", index, " +++++")
      print("SP500 list for date:", date_start)
      print(sp500_list)
      print("sp500_max", sp500_max)
      print("Close list:", list_c)
      print("Volume list:", list_v, "Volume max[0,7]", volume_max)
      print("High list:", list_h, "High max[0:7]:", close_max)
      print("Low list:", list_l, "High max[0:7]:", close_max)
      print(new_test_df.iloc[index].to_string())
      print("+++++ DEBUG INFO END +++++")

    counter = counter + 1
    if index % 2000 == 0:
      print(index,", ", end = '')
  print("Done")
  print("Entries", counter)
  #print( new_test_df.head(5).to_string())

  # Check the prediction
  new_cleaned_df  = new_test_df[COLUMNS_LEARN_LIST].copy()

  new_test_features = np.array(new_cleaned_df)
  new_test_predictions = model.predict(new_test_features, batch_size=BATCH_SIZE)

  new_test_df[['pred']] = new_test_predictions
  sortd_predict = new_test_df.sort_index().sort_values('pred', kind='mergesort', ascending=False)
  #print("Predicted set:")
  pd.options.display.float_format = '{:,.3f}'.format
  print(sortd_predict[['File', 'Date', 'Low6', 'Low7' ,'Low8', 'Low9', 'High6', 'High7' ,'High8', 'High9', 'Close6', 'Close7', 'Close8',  'Close9', 'Test_flag', 'Day_flag', 'pred']].head(10).to_string())

  passed = 0
  all_examples = 0
  hv_pers = 0
  looses = 0
  high_low_counter = 0 # In case of SHORT it is low counter, in case of LONG it is high counter

  hl_suc1 = 0
  cl_suc1 = 0
  cl_los1 = 0
  hl_suc2 = 0
  cl_suc2 = 0
  cl_los2 = 0
  hl_suc3 = 0
  cl_suc3 = 0
  cl_los3 = 0
  for index, row in sortd_predict.iterrows():
    #print(sortd_predict.at[index, 'pred'])
    if sortd_predict.at[index, 'pred'] > predict_high:
      continue

    if sortd_predict.at[index, 'pred'] < predict_low:
      break

    round_pred = round(sortd_predict.at[index, 'pred'], 3)
    passed_flag = 0
    all_examples = all_examples + 1
    if new_test_df.at[index, 'Test_flag'] == 2:
      passed = passed + 1
      passed_flag = 1

    if new_test_df.at[index, 'Day_flag'] >= 1:
      high_low_counter = high_low_counter + 1
      passed_flag = 1

    if new_test_df.at[index, 'Test_flag'] >= HALF_VALUE or passed_flag == 1:
      hv_pers = hv_pers + 1

    if passed_flag == 0 and new_test_df.at[index, 'Test_flag'] < 1:
      looses = looses + 1
      # Save failed/loose cases in order to learn the model
      if FAILED_TC == True:
        failed_test_df = failed_test_df.append({'File': ticker, 'Type':'0', 'Date': date_start,
                                        'Open0': new_test_df.at[index, 'Open0'], 'Open1': new_test_df.at[index, 'Open1'],
                                        'Open2': new_test_df.at[index, 'Open2'], 'Open3': new_test_df.at[index, 'Open3'],
                                        'Open4': new_test_df.at[index, 'Open4'], 'Open5': new_test_df.at[index, 'Open5'],
                                        'Open6': new_test_df.at[index, 'Open6'], 'Open7': new_test_df.at[index, 'Open7'],
                                        'Open8': new_test_df.at[index, 'Open8'], 'Open9': new_test_df.at[index, 'Open9'],
                                        'High0': new_test_df.at[index, 'High0'], 'High1': new_test_df.at[index, 'High1'],
                                        'High2': new_test_df.at[index, 'High2'], 'High3': new_test_df.at[index, 'High3'],
                                        'High4': new_test_df.at[index, 'High4'], 'High5': new_test_df.at[index, 'High5'],
                                        'High6': new_test_df.at[index, 'High6'], 'High7': new_test_df.at[index, 'High7'], 
                                        'High8': new_test_df.at[index, 'High8'], 'High9': new_test_df.at[index, 'High9'],
                                        'Low0': new_test_df.at[index, 'Low0'], 'Low1': new_test_df.at[index, 'Low1'],
                                        'Low2': new_test_df.at[index, 'Low2'], 'Low3': new_test_df.at[index, 'Low3'],
                                        'Low4': new_test_df.at[index, 'Low4'], 'Low5': new_test_df.at[index, 'Low5'],
                                        'Low6': new_test_df.at[index, 'Low6'], 'Low7': new_test_df.at[index, 'Low7'], 
                                        'Low8': new_test_df.at[index, 'Low8'], 'Low9': new_test_df.at[index, 'Low9'],                                      
                                        'Close0': new_test_df.at[index, 'Close0'], 'Close1': new_test_df.at[index, 'Close1'],
                                        'Close2': new_test_df.at[index, 'Close2'], 'Close3': new_test_df.at[index, 'Close3'],
                                        'Close4': new_test_df.at[index, 'Close4'], 'Close5': new_test_df.at[index, 'Close5'],
                                        'Close6': new_test_df.at[index, 'Close6'], 'Close7': new_test_df.at[index, 'Close7'],
                                        'Close8': new_test_df.at[index, 'Close8'], 'Close9': new_test_df.at[index, 'Close9'],                                    
                                        'Volume0': new_test_df.at[index, 'Volume0'], 'Volume1': new_test_df.at[index, 'Volume1'],
                                        'Volume2': new_test_df.at[index, 'Volume2'], 'Volume3': new_test_df.at[index, 'Volume3'],
                                        'Volume4': new_test_df.at[index, 'Volume4'], 'Volume5': new_test_df.at[index, 'Volume5'],
                                        'Volume6': new_test_df.at[index, 'Volume6'], 'Volume7': new_test_df.at[index, 'Volume7'], 
                                        'Volume8': new_test_df.at[index, 'Volume8'], 'Volume9': new_test_df.at[index, 'Volume9'],
                                        'Sp0': new_test_df.at[index, 'Sp0'], 'Sp1': new_test_df.at[index, 'Sp1'], 'Sp2': new_test_df.at[index, 'Sp2'],
                                        'Sp3': new_test_df.at[index, 'Sp3'], 'Sp4': new_test_df.at[index, 'Sp4'], 'Sp5': new_test_df.at[index, 'Sp5'],
                                        'Sp6': new_test_df.at[index, 'Sp6'], 'Sp7': new_test_df.at[index, 'Sp7'], 'Sp8': new_test_df.at[index, 'Sp8'],
                                        'Sp9': new_test_df.at[index, 'Sp9']
                                        , 'Test_flag': new_test_df.at[index, 'Test_flag'], 'Pred': sortd_predict.at[index, 'pred'] 
                                      }, ignore_index=True)
    
    if TEST_MODE == "S":
      hl1 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Low7'], 3)
      cl1 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Close7'], 3)
      hl2 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Low8'], 3)
      cl2 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Close8'], 3)
      hl3 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Low9'], 3)
      cl3 = round(new_test_df.at[index, 'Close6']/new_test_df.at[index, 'Close9'], 3)
    else:
      hl1 = round(new_test_df.at[index, 'High7']/new_test_df.at[index, 'Close6'], 3)
      cl1 = round(new_test_df.at[index, 'Close7']/new_test_df.at[index, 'Close6'], 3)
      hl2 = round(new_test_df.at[index, 'High8']/new_test_df.at[index, 'Close6'], 3)
      cl2 = round(new_test_df.at[index, 'Close8']/new_test_df.at[index, 'Close6'], 3)
      hl3 = round(new_test_df.at[index, 'High9']/new_test_df.at[index, 'Close6'], 3)
      cl3 = round(new_test_df.at[index, 'Close9']/new_test_df.at[index, 'Close6'], 3)

    if cl1 >= ETALON_VALUE:
      cl_suc1 = cl_suc1 + 1
    elif cl1 < 1:
      cl_los1 = cl_los1 + 1
      # Calculate the 2-nd day
      if cl2 >= ETALON_VALUE:
        cl_suc2 = cl_suc2 + 1
      elif cl2 < 1:
        cl_los2 = cl_los2 + 1
        # Calculate the 3-rd day
        if cl3 >= ETALON_VALUE:
          cl_suc3 = cl_suc3 + 1
        elif cl3 < 1:
          cl_los3 = cl_los3 + 1

    if hl1 >= ETALON_VALUE:
      hl_suc1 = hl_suc1 + 1
    else:
      #Calculate the 2-nd day
      if hl2 >= ETALON_VALUE:
        hl_suc2 = hl_suc2 + 1
      else:
        #Calculate the 3-rd day
        if hl3 >= ETALON_VALUE:
          hl_suc3 = hl_suc3 + 1

    new_entry = pd.DataFrame({'File': new_test_df.at[index, 'File'], 'Date': new_test_df.at[index, 'Date'], 'Success_flag': passed_flag ,
                                        'Pred': round_pred,
                                        'First_day': new_test_df.at[index, 'Day_flag'], 'Test_flag': new_test_df.at[index, 'Test_flag'],
                                        'CL1': cl1, 'HL1': hl1, 'CL2': cl2, 'HL2': hl2, 'CL3': cl3, 'HL3': hl3}, index=[0])
    stat_test_df = pd.concat([stat_test_df, new_entry], ignore_index=True)

    if index == DEBUG_INDEX and ticker == DEBUG_TICKER:
      print("close_day1:", cl1, "HL_day1:", hl1)

  # To avoid devision by 0
  if all_examples == 0:
    all_examples = 1
    #passed = 1
  print("Passed:", passed, "of", all_examples)
  print("HALF_VALUE:", hv_pers, "of", all_examples)
  print("High/Low passed:", high_low_counter, "of", all_examples, "Accuracy:", high_low_counter/all_examples)
  print("Looses:", looses, "of", all_examples, "Delta:", (looses/all_examples))

  delta_hl_suc1 = hl_suc1/all_examples
  delta_cl_suc1 = cl_suc1/all_examples
  delta_cl_los1 = cl_los1/all_examples
  delta_hl_suc2 = hl_suc2/all_examples
  delta_cl_suc2 = cl_suc2/all_examples
  delta_cl_los2 = cl_los2/all_examples
  delta_hl_suc3 = hl_suc3/all_examples
  delta_cl_suc3 = cl_suc3/all_examples
  # Take into consideration only looses in accordance with successful CLOSES of previous days while high_low_counter(Accuracy) takes into considerations both High/Low condition + CLOSES
  delta_cl_los3 = cl_los3/all_examples
  print("Success by day 1 High/Low:", hl_suc1, "of", all_examples, "Delta:", delta_hl_suc1)
  print("Success by day 1 close:", cl_suc1, "of", all_examples, "Delta:", delta_cl_suc1)
  print("Looses by day 1 close:", cl_los1, "of", all_examples, "Delta:", delta_cl_los1)
 
  price_str = str(round(entry_price, 3)) + '/' + str(round((entry_price*(2 - ETALON_VALUE)), 2)) + '/' + str(round((ETALON_VALUE*entry_price), 2))
  # Substruct the DIR name
  index = model_name.find('/')
  if index > 0:
    model_name_fixed = model_name[index+1:]
  else:
    model_name_fixed = model_name

  new_entry = pd.DataFrame({ 'Model': model_name_fixed, 'Percentage': test_value,
                                'Ticker': ticker, 'Price': price_str, 'Pred': predict_range, 'Start_date': date_list[0], 'Close_EV': passed, 
                                'Close_HV': hv_pers, 'High_Low_EV':high_low_counter, 'Looses': looses, 'Cases':all_examples,
                                'Accuracy': (high_low_counter/all_examples), "Looses_per": (looses/all_examples),
                                'HL_suc1': delta_hl_suc1, 'HL_suc2': delta_hl_suc2, 'HL_suc3': delta_hl_suc3, 
                                'CL_suc1': delta_cl_suc1, 'CL_suc2': delta_cl_suc2, 'CL_suc3': delta_cl_suc3, 
                                'CL_los1': delta_cl_los1, 'CL_los2': delta_cl_los2, 'CL_los3': delta_cl_los3}, index=[0])
  result_stat_df = pd.concat([result_stat_df, new_entry], ignore_index=True)
  #print(result_stat_df.to_string())

#################################################

def draw_chart_intraday(ticker, start_date, title_str, entry_price, pdf):
  end_date = pd.to_datetime(start_date) + pd.DateOffset(days=1)
  print("draw_chart_intraday() started")
  for tick in [ticker, "^GSPC"]:
    try:
      ticker_data_df = yf.download(tick, start=start_date, end=end_date, interval = "15m", progress=False, show_errors=False, threads=False)
    except ValueError: 
      print("Exception catched on ticker", ticker)
      return False
    except Exception as ex:
      print("Exception catched on ticker", ticker, ex)
      return False

    #else:
      # Do nothing
  
    #print(ticker_data_df.tail(10))
    if tick == "^GSPC":
      sp500_df = ticker_data_df
    else:
      ticker_df = ticker_data_df

  if len(sp500_df) == 0 or len(ticker_df) == 0:
    return False
  #print(sp500_df.tail(10))
  #print(ticker_df.tail(10))
  sp500_x = np.array(sp500_df.index.tolist())
  sp500_close = np.array(sp500_df['Close'].tolist())
  sp500_open = np.array(sp500_df['Open'].tolist())
  sp500_levels = [sp500_open[0], sp500_open[0]*1.01, sp500_open[0]*0.99 ]
  
  ticker_x = np.array(ticker_df.index.tolist())
  ticker_close = np.array(ticker_df['Close'].tolist())
  ticker_open = np.array(ticker_df['Open'].tolist())
  ticker_hi = np.array(ticker_df['High'].tolist())
  ticker_lo = np.array(ticker_df['Low'].tolist())

  tick_levels = [ticker_open[0], ticker_open[0]*1.04, ticker_open[0]*0.96 ]
  
  text_str = title_str + "\n" + ticker + ":" + start_date
  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.95, text_str, transform=fig.transFigure, size=9)    
  plt.title('Ticker intraday close price, 15m')
  plt.plot(ticker_x, ticker_close, color='k', label = 'Close')
  plt.plot(ticker_x, ticker_open, color='b', label = 'Open')
  plt.plot(ticker_x, ticker_hi, color='g', label = 'High')
  plt.plot(ticker_x, ticker_lo, color='r', label = 'Low')

  plt.hlines(tick_levels, min(ticker_x), max(ticker_x), linestyle='--', color='r', label='+-4%')
  if entry_price > 0:
    tick_levels2 = [entry_price, entry_price*1.04, entry_price*0.96] 
    plt.hlines(tick_levels2, min(ticker_x), max(ticker_x), linestyle='--', color='g', label='+-4% entry price')
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()
  
  text_str = "S&P500" + ":" + start_date
  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.95, text_str, transform=fig.transFigure, size=7)    
  plt.title('S&P500 intraday close price, 15m')
  plt.plot(sp500_x, sp500_close, color='k', label = 'Sp500')
  plt.hlines(sp500_levels, min(sp500_x), max(sp500_x), linestyle='--', color='r', label='+-1%')
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()




def get_prediction_stat(stat_data, START_PRED, PERCENTAGE):
  MAX_PRED   = 0.995
  STEP       = 0.005
  TARGET_VAL = 1 + PERCENTAGE/100
  x_pred = START_PRED
  final_stat = pd.DataFrame(columns=['Pred', "All", 'CL1', 'CL2','CL3', 'HL1', 'HL2', 'HL3', 'LO1', 'LO2', 'LO3', 'CL123', 'HL123', 'CL1_LO1', 'HL1_LO1', 'CL123_LO3', 'HL123_LO3'])

  delta_c1 = 0
  delta_h1 = 0
  delta_cl123 = 0
  delta_hl123 = 0

  while x_pred <= MAX_PRED:
    success_counter = 0
    failure_counter = 0
    all_cases       = 0
    cl1_counter     = 0
    lo1_counter     = 0
    hl1_counter     = 0
    cl2_counter     = 0
    lo2_counter     = 0
    hl2_counter     = 0
    cl3_counter     = 0
    lo3_counter     = 0
    hl3_counter     = 0
  
    cl123_counter     = 0 # Aggregated CL1, CL2, CL3
    hl123_counter     = 0 # Aggregated HL1, HL2, HL3
    lo3_counter     = 0 # By Close3
   
    for index_dt,row in stat_data.iterrows():
      if float(stat_data.at[index_dt, 'Pred']) >= x_pred:
        all_cases = all_cases + 1
   
        for i in [1,2,3]:
          column_CL = "CL" + str(i);
          cl = stat_data.at[index_dt, column_CL]
          if cl >= TARGET_VAL:
            cl123_counter = cl123_counter + 1
            break
        for i in [1,2,3]:
          column_HL = "HL" + str(i); 
          hl = stat_data.at[index_dt, column_HL]
          if hl >= TARGET_VAL:
            hl123_counter = hl123_counter + 1
            break
     
        cl1 = stat_data.at[index_dt, 'CL1']
        if cl1 >= TARGET_VAL:
          cl1_counter = cl1_counter + 1
        if cl1 < 1:
          lo1_counter = lo1_counter + 1
        hl1 = stat_data.at[index_dt, 'HL1']
        if hl1 >= TARGET_VAL:
          hl1_counter = hl1_counter + 1
     
        cl2 = stat_data.at[index_dt, 'CL2']
        if cl2 >= TARGET_VAL:
          cl2_counter = cl2_counter + 1
        if cl2 < 1:
          lo2_counter = lo2_counter + 1
        hl2 = stat_data.at[index_dt, 'HL2']
        if hl2 >= TARGET_VAL:
          hl2_counter = hl2_counter + 1
     
        cl3 = stat_data.at[index_dt, 'CL3']
        if cl3 >= TARGET_VAL:
          cl3_counter = cl3_counter + 1
        if cl3 < 1:
          lo3_counter = lo3_counter + 1
        hl3 = stat_data.at[index_dt, 'HL3']
        if hl3 >= TARGET_VAL:
          hl3_counter = hl3_counter + 1
     
    if all_cases == 0:
      break
    
    if lo1_counter != 0:
      delta_c1 = round(cl1_counter/lo1_counter,2)
      delta_h1 = round(hl1_counter/lo1_counter,2)

    if lo3_counter != 0:
      delta_cl123 = round(cl123_counter/lo3_counter,2)
      delta_hl123 = round(hl123_counter/lo3_counter,2)
    #else:
      # Leave as it was on the previous step 
      # delta_c1...
    
      #delta_c = cl1_counter # Leave as it was on the previous step
      #delta_h = hl1_counter # Leave as it was on the previous step
    
    new_entry = pd.DataFrame({ 'Pred': round(x_pred,3), 'All':all_cases, 'CL1':cl1_counter, 'CL2':cl2_counter,'CL3':cl3_counter, 'HL1':hl1_counter, 'HL2':hl2_counter, 'HL3':hl3_counter, 
                               'LO1':lo1_counter, 'LO2':lo2_counter, 'LO3':lo3_counter, 'CL123':cl123_counter, 'HL123':hl123_counter, 
                               'CL1_LO1':delta_c1, 'HL1_LO1':delta_h1, 'CL123_LO3':delta_cl123, 'HL123_LO3':delta_hl123 }, index=[0])
    final_stat = pd.concat([final_stat, new_entry], ignore_index=True)
    
    x_pred = x_pred + STEP
    
  #print(final_stat.to_string())
  return final_stat


def handle_ticker_data(stat_data, START_PRED, PERCENTAGE, pdf, PREDICTION):
  HL1_THRESHOLD = 0
  print("HL1_THRESHOLD=", HL1_THRESHOLD)
  ticker_str = stat_data.at[stat_data.index[0], 'File']

  final_stat = get_prediction_stat(stat_data, START_PRED, PERCENTAGE) 
  print(final_stat.to_string())

  x_pred_list = np.array(final_stat['Pred'].tolist())
  cl1_list    = np.array(final_stat['CL1'].tolist())
  lo1_list    = np.array(final_stat['LO1'].tolist())
  hl1_list    = np.array(final_stat['HL1'].tolist())
  cl123_list = np.array(final_stat['CL123'].tolist())
  hl123_list = np.array(final_stat['HL123'].tolist())
  lo3_list = np.array(final_stat['LO3'].tolist())
  
  delta_cl1_lo1 = np.array(final_stat['CL1_LO1'].tolist())
  delta_hl1_lo1 = np.array(final_stat['HL1_LO1'].tolist())
  delta_cl123_lo3 = np.array(final_stat['CL123_LO3'].tolist())
  delta_hl123_lo3 = np.array(final_stat['HL123_LO3'].tolist())

  delta_hl1_max = max(delta_hl1_lo1)
  delta_cl1_max = max(delta_cl1_lo1)
  delta_cl123_max = max(delta_cl123_lo3)
  delta_hl123_max = max(delta_hl123_lo3)
  # Show only "good" examples
  if delta_hl1_max < HL1_THRESHOLD:
    return False

  first_day_list = np.array(stat_data['First_day'].tolist())
  first_day_loses = np.array(stat_data['CL1'].tolist())
  pred_list = np.array(stat_data['Pred'].tolist())

  text_str = ticker_str + ":" + "HL1_MAX:" + str(delta_hl1_max) + ". CL1_MAX:" + str(delta_cl1_max) + ". CL123_MAX:" + str(delta_cl123_max) + ". HL123_MAX:" + str(delta_hl123_max)

  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.9, text_str, transform=fig.transFigure, size=7)    
  plt.title('CL1, HL1 and LO1 cases')
  plt.plot(x_pred_list, cl1_list, color='g', label = 'Close 1 success cases')
  plt.plot(x_pred_list, lo1_list, color='r', label = 'Close 1 looses cases')
  plt.plot(x_pred_list, hl1_list, color='y', label = 'High/Low 1 success cases')
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()
  
  fig = plt.figure(figsize=(12, 8))    
  plt.title('CL123, HL123 and LO3 cases')
  plt.plot(x_pred_list, cl123_list, color='g', label = 'Close 123 success cases')
  plt.plot(x_pred_list, hl123_list, color='y', label = 'High/Low 123 success cases')
  plt.plot(x_pred_list, lo3_list, color='r', label = 'Close 3 looses cases')
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()

  
  fig = plt.figure(figsize=(12, 8))    
  plt.title('CL1,HL1 and LO1 delta')
  plt.plot(x_pred_list, delta_cl1_lo1, color='g', label = 'CL1/LO1')
  plt.plot(x_pred_list, delta_hl1_lo1, color='y', label = 'HL1/LO1')
  for i in range(len(first_day_list)):
    val = first_day_list[i]/3
    l_styles = "solid"    
    if first_day_loses[i] < 1: # Looses case for the first day
      val = -0.3
    elif val == 0: # NOT looses case for the first day, but target is NOT executed
      l_styles ="dashed"
      val = 0.15

    plt.vlines(pred_list[i], 0, val, linestyles=l_styles, colors ="k")
  if float(PREDICTION) > 0:
    plt.vlines(PREDICTION, 0, max(delta_hl1_lo1), linestyles="solid", colors ="r")    

  plt.hlines([0,1,1.5,2,3], min(x_pred_list), max(x_pred_list), linestyle='--', color='r')
  plt.legend()
  pdf.savefig()
  plt.close()

  
  fig = plt.figure(figsize=(12, 8))    
  plt.title('CL123, HL123 and LO3 delta')
  plt.plot(x_pred_list, delta_cl123_lo3, color='g', label = 'CL123/LO3')
  plt.plot(x_pred_list, delta_hl123_lo3, color='y', label = 'HL123/LO3')
  for i in range(len(first_day_list)):
    val = first_day_list[i]/3
    if first_day_list[i] == 0:
      val = -0.3
    plt.vlines(pred_list[i], 0, val, linestyles ="solid", colors ="k")
  #if float(PREDICTION) > 0:
  #  plt.vlines(PREDICTION, 0, max(delta_hl1_lo1), linestyles ="solid", colors ="r")    

  plt.hlines([0,1,1.5,2,3], min(x_pred_list), max(x_pred_list), linestyle='--', color='r')
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()
  
  return True


def draw_chart_by_df(ticker_df, title_str, pdf):  
  ticker_x = np.array(ticker_df.index.tolist())
  ticker_close = np.array(ticker_df['Close'].tolist())
  ticker_hi = np.array(ticker_df['High'].tolist())
  ticker_lo = np.array(ticker_df['Low'].tolist())
  
  CL1 = round(ticker_close[6]/ticker_close[7], 3)
  LO1 = round(ticker_close[6]/ticker_lo[7], 3)
  CL1_REV = round(ticker_close[7]/ticker_close[6], 3)
  HI1 = round(ticker_hi[7]/ticker_close[6], 3)
 
 
  title_str = "Close:" + str(ticker_close) + "\n" + "High:" + str(ticker_hi) + "\n" + "Low:" + str(ticker_lo) + "\n"
  title_str = title_str + "SHORT: CL1:" + str(CL1) + " LO1:" + str(LO1) + " LONG: CL1: " + str(CL1_REV) + " HI1:" + str(HI1)
  print(title_str)
  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.9, title_str, transform=fig.transFigure, size=9)    
  plt.title('Ticker Close, High, Low, 1d')
  plt.plot(ticker_x, ticker_close, color='b', label = 'Close')
  plt.plot(ticker_x, ticker_hi, color='g', label = 'High')
  plt.plot(ticker_x, ticker_lo, color='r', label = 'Low')
  plt.vlines([ticker_x[6], ticker_x[7]], min(ticker_lo), max(ticker_hi), linestyles ="dotted", colors ="k", label="7th (entry) and 8-th (first) day")
  plt.legend()
  #plt.show()
  pdf.savefig()
  plt.close()
  

def get_Nth_day(start_seria_date, target_day_id):
  #TARGET_DAY starts from 0. for example 7 means the 8-th day
  end_date = pd.to_datetime(start_seria_date) + pd.DateOffset(days=20)
  print("S&P500 read online:", start_seria_date, end_date)
  sp500_data = yf.download("^GSPC", start=start_seria_date, end=end_date, progress=False, show_errors=False, threads=False)  
  if( target_day_id >= len(sp500_data) ):
    print("WARNING: something wrong: target_day_id >= len(sp500_data)", target_day_id, len(sp500_data))
    return ""
  #print(sp500_data.head(10).to_string())

  target_date = str(sp500_data.index[target_day_id].date())
  print("\nHistory date range:", start_seria_date, "-", target_date, ":")
  return target_date

def get_seria_by_ticker(start_seria_date, ticker, seria_size):
  #seria_size starts from 1. 10 means 10 elements
  print(ticker, "read online.")
  end_date = pd.to_datetime(start_seria_date) + pd.DateOffset(days=20)
  data_df = yf.download(ticker, start=start_seria_date, end=end_date, progress=False, show_errors=False, threads=False)  

  result_df = data_df.head(seria_size)
  #print(result_df.to_string())

  return result_df

def draw_title_page(str_title, prediction, pdf):
  predict_str = "Prediction: " + str(prediction)
  fig = plt.figure(figsize=(16, 12))
  fig.clf()
  fig.text(0.1, 0.9, predict_str, transform=fig.transFigure, size=12)
  fig.text(0.5, 0.1, str_title, transform=fig.transFigure, size=10, ha="center")    
  pdf.savefig()
  plt.close()


def do_ticker_stat(stat_file, ticker, prediction, verbose):
  short_name = get_short_model_name(stat_file)
  LOG_FILE = "log_" + ticker + "_" + short_name + "_" + str(datetime.now().strftime('%Y%m%d_%H%M%S')) + ".txt"
  sys.stdout = open(LOG_FILE, 'w', buffering=1)


  print("Started:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  stat_data = pd.read_csv( stat_file,
                           names=['id', 'File', 'Date', 'Success_flag', 'Pred', 'First_day', 'Test_flag', 'CL1', 'HL1', 'CL2', 'HL2', 'CL3', 'HL3'], header=None, skiprows=[0]
                         )
  print(stat_file, "loaded")
  v_postfix = ""
  if verbose == True:
    v_postfix = "_v"
  OUTPUT_PDF_FILE = "PDF_" + ticker + "_" + short_name + "_intraday" + v_postfix + ".pdf"

  TARGET_DAY = 7 # Starting from 0. it means the 8 the day
  SERIA_SZ   = 10
   
  stat_ticker_data = stat_data[(stat_data['File'] == ticker)]
  first_page = "Model: " + short_name + " Ticker: " + ticker + "\n" + \
               stat_ticker_data.to_string() + "\n" + \
               "Size: " + str(len(stat_ticker_data))
           
  print(first_page)
  if len(stat_ticker_data) == 0: 
    return

  pdf = PdfPages(OUTPUT_PDF_FILE)
  draw_title_page(first_page, prediction, pdf)

  rc = get_model_params(stat_file)
  if rc[0] != 1:
    sys.exit("Impossible to specify parameters of the model.") 
  #TEST_MODE = rc[1] 
  PERCENTAGE = rc[2]
  START_PRED = rc[3]
  handle_ticker_data(stat_ticker_data, START_PRED, PERCENTAGE, pdf, prediction)

  if verbose == False:
    pdf.close()
    return

  counter = 0 
  for index_dt,row in stat_ticker_data.iterrows():
    ticker_str = stat_ticker_data.at[index_dt, 'File']
    str_title = str(counter) + str(row.to_frame().T)
    print(str_title)
  
    # It is start of the seria but we need to get the 8th day is the day after the entry day
    start_date = stat_data.at[index_dt, 'Date']
    target_date = get_Nth_day(start_date, TARGET_DAY)
    print("Target date:", target_date)
    if len(target_date) == 0:
      continue

    # Only last 60 days available for intraday  
    last_day = date.today() - timedelta(days=60)
    print(str(pd.to_datetime(target_date).date()), str(last_day))
    if pd.to_datetime(target_date).date() <= last_day :
      print(target_date, "is too old. Intraday is NOT available!!!")
      continue

    seria_df = get_seria_by_ticker(start_date, ticker, SERIA_SZ)
    ticker_close = np.array(seria_df['Close'].tolist())
    if(len(ticker_close) < 7):
      continue
    entry_price = ticker_close[6]     
    draw_chart_intraday(ticker, target_date, str_title, entry_price, pdf)
    draw_chart_by_df(seria_df, "TODO", pdf)

  counter += 1
  pdf.close()

def create_pdf_file(OUTPUT_PDF_FILE, header_title, txt_result):
  ## Create a new PDF file
  ## Orientation: P = Portrait, L = Landscape
  ## Unit = mm, cm, in
  ## Format = 'A3', 'A4' (default), 'A5', 'Letter', 'Legal', custom size with (width, height)
  pdf = FPDF(orientation="L", unit="mm", format="A4")

  ## Add a page
  pdf.add_page()

  ## Specify Font
  ## Font Family: Arial, Courier, Helvetica, Times, Symbol
  ## Font Style: B = Bold, I = Italic, U = Underline, combinations (i.e., BI, BU, etc.)
  pdf.set_font("Courier", style="B", size=12)
  pdf.set_text_color(0, 0, 0) 
  #pdf.set_text_color(0, 0, 255) # Blue 

  ## Add text
  ## Cell(w, h, txt, border, ln, align)
  ## w = width, h = height
  ## txt = your text
  ## ln = (0 or False; 1 or True - move cursor to next line)
  ## border = (0 or False; 1 or True - draw border around the cell)
  pdf.cell(200, 10, txt=header_title, ln=1, align="C")

  pdf.set_font("Courier", size=7)
  pdf.set_text_color(0, 0, 0)
  for line in txt_result.splitlines():
    if len(line) > 0 and (line[0] == "=" or line[0] == "#"):
      pdf.set_font("Courier", style="B", size=7)
    else:
      pdf.set_font("Courier", size=7)
    pdf.cell(0, 5, txt=line, ln=1, align=0)

  ## Save the PDF file
  pdf.output(OUTPUT_PDF_FILE)
  pdf.close()

def tune_details(details, pred):
  id = 0
  #print(details)  
  while id < len(details):
    base = id
    #print("id=", id)
    if id > 0:
      id = details[id:].find(";")
      if id < 0:
        break
      id += 1     
    id = base + id

    id_end = details[id:].find(":")
    if id_end < 0:
      break

    id_end = id + id_end
    #print("id=", id, "id_end=", id_end)
    pred_in = details[id:id_end]
   
    #print("pred_in", pred_in, id, id_end) 
    if float(pred_in) < pred:
      break
    if len(details) > id_end + 1: 
      id = id_end + 1
    else:
      break
 
  str_out = details[0:id] + "_X_" + details[id:]
  print(str_out)
  return str_out

def predict_cases(stat_df, zip_model_archive, mode):
  BATCH_SIZE = 2048
  print(stat_df)
  if mode == "S":
    main_model_dir = "model_max_full"
  else:
    main_model_dir = "model_min_full"    

  shutil.unpack_archive(zip_model_archive, './', 'zip')
             
  model = tf.keras.models.load_model(main_model_dir)
  model.summary()

  new_cleaned_df  = stat_df[COLUMNS_LEARN_LIST].copy()

  new_cleaned_df = new_cleaned_df.astype(float)
  print(new_cleaned_df.to_string())
  new_test_features = np.array(new_cleaned_df)

  new_test_predictions = model.predict(new_test_features, batch_size=BATCH_SIZE)

  #print("NEW test features: ", new_test_features)
  print ("NEW predictions max: ", new_test_predictions.max())

  stat_df[['pred']] = new_test_predictions
  stat_df['pred'] = stat_df['pred'].round(decimals = 3)
  return stat_df

def find_latest_file_by_template(template):
  dir_in = "."
  latest_date = 0 
  rc = ""
  date_sz = 8
  for f in os.listdir(dir_in):
    #full_path = os.path.join(dir_in,f)
    id = f.find(template)
    if id >= 0 and f.find(".csv") >= 0: 
      #print(f, id)
      offset = id + len(template)
      curr_date = int(f[offset:offset+date_sz])
      #print(curr_date, offset)
      if curr_date > latest_date:
        latest_date = curr_date   
        rc = f
  print("rc =", rc)
  return rc


###### Ticker stat analytics #########
def calc_profit_col(row):
  if float(row['CL1']) >= 0.995:
    return 1

  return 0

                     
def calc_prof_targ_col(row):
  if float(row['Profit']) == 1 or float(row['First_day']) == 1:
    return 1
  return 0


def calc_cl1_hl1_targ_col(stat_data, percent):
  stat_data['CL1_targ'] = 0  
  stat_data['HL1_targ'] = 0
  for index, row in stat_data.iterrows():
    if float(row['CL1']) >= 1 + percent/100:
      stat_data.loc[index, 'CL1_targ'] = 1

    if float(row['HL1']) >= 1 + percent/100:
      stat_data.loc[index, 'HL1_targ'] = 1
  
  return stat_data

def calc_model_types_stat(stat_data):
  MODEL_TYPES = ["T7_N7", "T78_N7", "T789_N7", "T7_N9", "T7_N10"]
  MODEL_NUMS = [0]* len(MODEL_TYPES)
  for index, row in stat_data.iterrows():
    for i in range(len(MODEL_TYPES)):
      if row['Model'].find(MODEL_TYPES[i]) != -1:
        MODEL_NUMS[i] += 1
  for i in range(len(MODEL_TYPES)):
    print( MODEL_TYPES[i], ":", MODEL_NUMS[i] )


def copy_stat_model_files(final_stat, MODE):
  OUT_DIR = "FINAL_DIR_" + MODE
  shutil.rmtree(OUT_DIR, ignore_errors=True)
  os.makedirs(OUT_DIR, exist_ok=True)

  for index, row in final_stat.iterrows():
    csv_file = row['Csv_file']
    model_file = row['Model_file']
    if os.path.isfile(csv_file) and os.path.isfile(model_file):
      shutil.copy(csv_file, OUT_DIR)
      shutil.copy(model_file, OUT_DIR)
  
  final_file = "final_tickers_stat_" + str(datetime.now().strftime('%Y%m%d_%H%M%S')) + ".csv"
  final_stat.to_csv(final_file)
  shutil.move(final_file, OUT_DIR)

def prepare_seria(stat_data, seria_sz):
  str_out = ""
  cl_details = "" 
  for index, row in stat_data.iterrows():
    if seria_sz <= 0:
      break

    val = "LO"
    if stat_data.loc[index, 'CL1_targ'] == 1:
      val = "CL"
    elif stat_data.loc[index, 'HL1_targ'] == 1: 
      val = "HL"
    elif stat_data.loc[index,'Profit'] == 1:
      #val = "PR"
      val = str(float(row['CL1']))

    str_out = str_out + str(row['Pred'])+ ":" + val + ";"
    cl_details = cl_details + str(row['Pred']) + ":" + str(row['CL1']) + ";"
    seria_sz = seria_sz - 1 
  return str_out, cl_details

def check_stat_file(stat_file, percent):
  print("Input file: ", stat_file)
  stat_data = pd.read_csv( stat_file,
                          names=['id', 'File', 'Date', 'Success_flag', 'Pred', 'First_day', 'Test_flag', 'CL1', 'HL1', 'CL2', 'HL2', 'CL3', 'HL3'], header=None, skiprows=[0]
                          )
  if len(stat_data) < 3:
    print("WARNING: Something wrong with", stat_file)
    return 0,0,0,0,0,0,0
  stat_data['Profit'] = stat_data.apply(calc_profit_col, axis=1)
  stat_data['Prof_targ'] = stat_data.apply(calc_prof_targ_col, axis=1) 
  stat_data = calc_cl1_hl1_targ_col(stat_data, percent)

  # Check the first 10 elements
  #target_list = [10, 5, 4, 3, 2]
  target_list = [10, 5, 4, 3]
  for seria_sz in target_list:
    succ_cases = sum(stat_data['Prof_targ'].head(seria_sz))
    cl1_cases = sum(stat_data['CL1_targ'].head(seria_sz))
    hl1_cases = sum(stat_data['HL1_targ'].head(seria_sz))
    THRESHOLD = 0
    RC = 0
    if seria_sz == 10:
      THRESHOLD = 9
      RC = 5
    elif seria_sz == 5:
      THRESHOLD = 5
      RC = 4
    elif seria_sz == 4:
      THRESHOLD = 4
      RC = 3
    elif seria_sz == 3:
      THRESHOLD = 3
      RC = 2
    #elif seria_sz == 2:
    #  THRESHOLD = 2
    #  RC = 1


    # At least one HL1 case is required    
    if succ_cases >= THRESHOLD and len(stat_data) >= seria_sz and hl1_cases >= 1:
      print(stat_data.head(seria_sz).to_string())
      seria_rc = prepare_seria(stat_data, seria_sz)
      return RC, seria_sz, succ_cases, cl1_cases, hl1_cases, stat_data.loc[seria_sz-1,'Pred'], seria_rc[0], seria_rc[1] 
   
  return 0,0,0,0,0,0,0

##########################################################

def prepare_test_archive(TEST_MODE, ticker, test_src_dir):
  TEST_POSTFIX = "_dataset_"
  # Create test archive
  test_archive = "TEST_" + TEST_MODE + TEST_POSTFIX + ticker
  os.makedirs(test_archive, exist_ok=True)

  # Create ticker list file
  if TEST_MODE == "L": 
    ticker_file = test_archive + "/SPB_ALL_tickers.csv"
  else:
    ticker_file = test_archive + "/short_tickers_NY.csv"
  with open(ticker_file, 'w') as f:
    f.write(ticker)
  f.close()
  
  # Create dataset file
  dataset_file = ticker + ".csv"
  dataset_file = os.path.join(test_src_dir, dataset_file)
  if os.path.isfile(dataset_file):
    shutil.copy(dataset_file, test_archive)
  else:
    print(dataset_file, "NOT found")
    return ""
  # Create s&p500 file
  shutil.copy(SP500_FILE, test_archive)
  shutil.make_archive(test_archive, 'zip', ".", test_archive)
  shutil.rmtree(test_archive, ignore_errors=True)
  return (test_archive + ".zip")


def get_between(details, sep1, sep2):
  #print("get_between", details)
  rc = ""
  sep1_id = details.find(sep1)
  sep2_id = details[sep1_id:].find(sep2)
  if sep1_id < 0:
    print("sep1", sep1, "NOT found")
    return rc

  if sep2_id < 0:
    print("sep2", sep2, "NOT found")
    return rc
  substr = details[sep1_id+len(sep1):sep1_id+sep2_id] 
  #print(substr, sep1_id+len(sep1), sep1_id+sep2_id)  
  return substr

def estimate_details(details, pred):
  target_prev = ""
  id = 0 
  print("===", details, "Pred:", pred, "===")  
  while id < len(details):
    base = id
    #print("id=", id)
    if id > 0:
      id = details[id:].find(";")
      if id < 0:
        break
      id += 1     
    id = base + id

    id_end = details[id:].find(":")
    if id_end < 0:
      break

    target_in = get_between(details[id:], ":", ";")
    if len(target_in) < 0:
      break

    id_end = id + id_end
    #print("id=", id, "id_end=", id_end)
    pred_in = details[id:id_end]
   
    #print("pred_in", pred_in, id, id_end) 
    #print("pred_in:", pred_in, "target_in:", target_in)
    if float(pred_in) < pred:
      break
    if len(details) > id_end + 1: 
      id = id_end + 1
    else:
      break
    target_prev = target_in

  rc = ""
  #print("target_prev:", target_prev, "target_in:", target_in)
  for i in [target_prev, target_in]:
    if i == "CL":
      rc += "4"
    elif i == "HL":
      rc += "3"
    elif i == "":
      rc += "2"  # In case PRED > estimated pred
    elif i == "LO":
      rc += "0"
    else:
      if(float(i) > 1):
        rc += "1"
      else:
        rc += "0"  
  print("estimate_details() returns", rc)
  return rc
