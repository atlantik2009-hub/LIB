import sys, os, time, tempfile, random, shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *
from intra_learn_lib import load_model

STAT_TEST_PREFIX="ST_"

RESULT_STAT_FILE="result_stat_file.txt"

DETAILED_STAT_COLUMNS = ['File', 'CL', 'HL', 'LO', 'NL', 'CL_LO', 'CLHL_LO', 'NOT_LOOS_LO']
DELTA_COLUMN = 'Delta'

BATCH_SIZE = 2048

DEBUG = True

### Test functions ###
# CLx - Target passed by CLose price in x-th timeframe
# HLx - Target passed by High/Low price in x-th timeframe
# LOx - LOoses by Close price in x-th timeframe
# NLx - No Looses by Close price in x-th timeframe
def classify_ds_ext(selected_df, model_set):
  index_list = list(selected_df.index.values)
  entry_price  = selected_df.at[index_list[model_set.p_days-1], CLOSE_COLUMN]
  classify_list = selected_df[CLOSE_COLUMN].values.tolist()[model_set.p_days:model_set.p_days+model_set.t_days] 
  # Classify as target passed (positive case)
  rc = ""
  if model_set.mode == 'L':
    target_price = entry_price*(1+model_set.percent/100)
    if max(classify_list) >= target_price:
      id = classify_list.index(max(classify_list))
      rc = "CL" + str(id)

    delta = (classify_list[model_set.t_days-1]/entry_price - 1)*100      
  else:
    target_price = entry_price*(1-model_set.percent/100)
    if min(classify_list) <= target_price:
      id = classify_list.index(min(classify_list))
      rc = "CL" + str(id)

    delta = (1 - classify_list[model_set.t_days-1]/entry_price)*100  
  delta = round(delta,2)

  if len(rc) > 0:
    print(rc, "1_Entry price:", entry_price, "Target price:", target_price, "close_list:", classify_list, "Delta:", delta)
    return rc, delta
  
  # Classify as target passed (positive case)
  if model_set.mode == 'L':
    hl_list = selected_df[HIGH_COLUMN].values.tolist()[model_set.p_days:model_set.p_days+model_set.t_days] 
    if max(hl_list) >= target_price:
      id = hl_list.index(max(hl_list))
      rc = "HL" + str(id)
  else:
    hl_list = selected_df[LOW_COLUMN].values.tolist()[model_set.p_days:model_set.p_days+model_set.t_days] 
    if min(hl_list) <= target_price:
      id = hl_list.index(min(hl_list))
      rc = "HL" + str(id)  

  if len(rc) > 0:
    print(rc, "2_Entry price:", entry_price, "Target price:", target_price, "HL_list:", hl_list, "Delta:", delta)
    return rc, delta

  # Check the last element of the CLOSE list: if we loose classify it is as 0 (negative case)
  if model_set.mode == 'L':
    neg_k = 1 - model_set.percent*model_set.neg_delta/10000
    #print("neg_k", neg_k)
    if classify_list[model_set.t_days-1] < entry_price*neg_k:
      rc =  "LO" + str(model_set.t_days-1)
  else:
    neg_k = 1 + model_set.percent*model_set.neg_delta/10000
    #print("neg_k", neg_k)
    if classify_list[model_set.t_days-1] > entry_price*neg_k:
      rc =  "LO" + str(model_set.t_days-1)

  if len(rc) > 0:
    print(rc, "3_Entry price:", entry_price, "Target price:", target_price, "close_list:", classify_list, "Delta:", delta)
    return rc, delta
  
  # NOT looses for this case in day model_set.t_days
  rc = "NL" + str(model_set.t_days-1)
  print(rc, "4_Entry price:", entry_price, "Target price:", target_price, "close_list:", classify_list, "Delta:", delta)
  return rc, delta


def prepare_ticker_df_test(ticker_file, ticker_data, index_data, MODEL, dataset_df):
  global DEBUG

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

    #For debug purpose 
    if DEBUG == True:
      print(selected_df.to_string())
      print(index_sel_df.to_string())
      DEBUG = False
    
    # Calculate classification flag
    rc_array = classify_ds_ext(selected_df, MODEL)
    rc    = rc_array[0]
    delta = rc_array[1]    
    #print("classify_ds_ext() returns", rc)
    if rc[0:2] == "CL": 
      classify_stat.positive_num+=1
    elif rc[0:2] == "HL":
      classify_stat.hl_num+=1

    print("Delta: ", delta)
    if delta >= 0:
      # No looses
      classify_stat.not_looses_num+=1
    else:
      classify_stat.negative_num+=1
      if rc[0:2] == "CL" or rc[0:2] == "NL":
        print("AHTUNG: incorrect status ", rc[0:2])
    
    classify_flag = rc
    row_list = []
    row_list.append(ticker_file)
    row_list.append(classify_flag)
    row_list.append(selected_df.index[0])
    rc = put_df_into_list(selected_df, index_sel_df, MODEL.p_days, row_list)
    if rc == False:
      print("WARNING!!! prepare_ticker_df_test(): Something wrong")
      return classify_stat

    row_list.append(delta)
    
    #print("DS length:", len(row_list), "\nDS row:", row_list)
    dataset_df.loc[len(dataset_df)] = row_list

  #print("DS:\n", dataset_df[get_close_target_columns(MODEL)])
  return classify_stat

# Return stat file name 
def test_dataset(DATASET_DIR, MODEL, model_name, predict_low=0.1):
  print(MODEL)
  predict_high = 1
  #predict_low  = 0.1
  TEST_STAT_OUT_FILE = STAT_TEST_PREFIX + model_name[0:-4] + "_T" + str(MODEL.t_days) + "_" + str(datetime.now().strftime('%H%M%S')) + ".csv"
  print("TEST_STAT_OUT_FILE:", TEST_STAT_OUT_FILE)

  exclude_file = get_base_name(model_name) + "_exclude.csv"
  if os.path.isfile(exclude_file):
    print("Exclude file exists:", exclude_file)
    exclude_df = pd.read_csv(exclude_file, names=["Ticker"], skiprows = 0)
    #print(exclude_df)
    tickers_list = exclude_df["Ticker"].values.tolist()
    print("Size:", len(tickers_list), "Exclude tickers:", tickers_list )  
  else:
    print("Exclude file does NOT exist:", exclude_file) 
    tickers_list = []

  test_stat_df = pd.DataFrame(columns=TEST_STAT_COLUMNS)
 
  print("Test dir:", DATASET_DIR, "Model name:", model_name)

  model = load_model(model_name, MODEL)
  
  COMMON_LENGTH = MODEL.p_days+MODEL.t_days
  #print("0DS length:", len(DATASET_COLUMNS))
  DATASET_COLUMNS = get_dataset_columns(MODEL)
  #print("1DS length:", len(DATASET_COLUMNS))
  DATASET_COLUMNS.append(DELTA_COLUMN)  
  print(DATASET_COLUMNS)
  print("DS length:", len(DATASET_COLUMNS))
  LEARN_COLUMNS_NUM = MODEL.p_days*len(MODEL_COLUMNS)
  ALL_COLUMNS_NUM = COMMON_LENGTH*len(MODEL_COLUMNS)
  LEARN_COLUMNS = get_learn_columns_clean_full(MODEL)
  print("LEARN_COLUMNS:\n", LEARN_COLUMNS, "\nLen=", len(LEARN_COLUMNS))  
  print("Learn columns number:", LEARN_COLUMNS_NUM )
  print("All columns number:", ALL_COLUMNS_NUM )                   

  SHOWN_COLUMNS = get_close_target_columns(MODEL)
  SHOWN_COLUMNS.append(PRED_COLUMN)
  SHOWN_COLUMNS.append(DELTA_COLUMN)
  print("SHOWN_COLUMNS:", SHOWN_COLUMNS)
  
  dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
  
  index_file = DATASET_DIR + '/'+ INDEX_FILE

  index_data = pd.read_csv( index_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
  print("Index file:\n", index_data)

  classify_stat = ClassifyStat(0,0,0,0,0)
  
  counter = 0
  entries = os.listdir(DATASET_DIR)
  for entry in entries:
    # Skip Idex file
    if entry.find(INDEX_FILE) != -1 or entry.find(INDEX_FILE2) != -1:
      print(entry, "skipped")
      continue
    if os.path.isdir(entry) or entry.find(".csv") == -1:
      print(entry, "skipped")
      continue

    # Check if the ticker in the exclude list
    if len(tickers_list) > 0 and entry in tickers_list:
      print(entry, "excluded")
      continue

    ticker_file = DATASET_DIR + '/'+ entry
    counter+=1
    print(counter, "!!!", ticker_file, "!!!")
    ticker_data = pd.read_csv( ticker_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)

    dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
    ticker_stat = prepare_ticker_df_test(ticker_file, ticker_data, index_data, MODEL, dataset_df)
 
    classify_stat.positive_num = classify_stat.positive_num + ticker_stat.positive_num
    classify_stat.hl_num = classify_stat.hl_num + ticker_stat.hl_num
    classify_stat.negative_num = classify_stat.negative_num + ticker_stat.negative_num
    classify_stat.not_looses_num = classify_stat.not_looses_num + ticker_stat.not_looses_num
    classify_stat.syncup_error = classify_stat.syncup_error + ticker_stat.syncup_error
    classify_stat.skipped = classify_stat.skipped + ticker_stat.skipped
    
    #print(entry, "Length DS:", len(dataset_df))
    print("Classify stat:", classify_stat)
    print("Sum check:", (classify_stat.positive_num + classify_stat.hl_num + classify_stat.negative_num + classify_stat.not_looses_num))  

    ### Here we can add additional columns like Ticker id ###
    post_proc_df_add_clmns(dataset_df, MODEL)

    # Check the prediction
    new_cleaned_df  = dataset_df[LEARN_COLUMNS].copy()
    new_test_features = np.array(new_cleaned_df)
    if len(new_test_features) == 0:
      print("Length of the dataset is 0!")
      continue

    try:
      new_test_predictions = model.predict(new_test_features, batch_size=BATCH_SIZE)
    except Exception as ex:
      print("Exception catched on ticker", ticker_file, ex)
      print(new_test_features)
      continue

    dataset_df[[PRED_COLUMN]] = new_test_predictions
    sortd_predict = dataset_df.sort_index().sort_values('pred', kind='mergesort', ascending=False)
    pd.options.display.float_format = '{:,.4f}'.format

    metrix_list = get_metrix(sortd_predict, MODEL, predict_low)

    for index, row in sortd_predict.iterrows():
      #print(sortd_predict.at[index, 'pred'])
      if sortd_predict.at[index, 'pred'] > predict_high:
        continue

      if sortd_predict.at[index, 'pred'] < predict_low:
        break
      # Collect data fror pred(0.1-1)
      out_list = [row.File, row.Type, row.Datetime, row.pred]     
      test_stat_df.loc[len(test_stat_df)] = out_list
      #print(out_list)

  #print("Sum check:", (classify_stat.positive_num + classify_stat.hl_num + classify_stat.negative_num + classify_stat.not_looses_num))  
  #dataset_df.to_csv(DS_OUT_FILE, index=False) 
  #print("TEST_DF:\n", test_stat_df)
  test_stat_df.to_csv(TEST_STAT_OUT_FILE)
  return TEST_STAT_OUT_FILE

def get_metrix(sortd_predict, MODEL, predict_low):
  SHOWN_COLUMNS = get_close_target_columns(MODEL)
  SHOWN_COLUMNS.append(PRED_COLUMN)
  SHOWN_COLUMNS.append(DELTA_COLUMN)
  print("SHOWN_COLUMNS:", SHOWN_COLUMNS)

  filtered_df = sortd_predict[(sortd_predict['pred'] >= predict_low)] 
  print(filtered_df[SHOWN_COLUMNS].head(20).to_string())
  top20_delte = sortd_predict[DELTA_COLUMN].head(20).sum() 
  top10_delte = sortd_predict[DELTA_COLUMN].head(10).sum()
  top05_delte = sortd_predict[DELTA_COLUMN].head(5).sum()
  filt_pos_df = filtered_df[(filtered_df[DELTA_COLUMN] > 0)]
  filt_neg_df = filtered_df[(filtered_df[DELTA_COLUMN] < 0)]
  avg_pos = filt_pos_df[DELTA_COLUMN].mean()
  avg_neg = filt_neg_df[DELTA_COLUMN].mean()
  pos_num = len(filt_pos_df)
  neg_num = len(filt_neg_df)
  print("TOP20_SUM=", top20_delte)    
  print("TOP10_SUM=", top10_delte)
  print("TOP05_SUM=", top05_delte)        
  print("Avrg(pred>0.1)_pos=", avg_pos, "Size:", pos_num)
  print("Avrg(pred>0.1)_neg=", avg_neg, "Size:", neg_num)
  return [top20_delte, top10_delte, top05_delte, avg_pos, avg_neg, pos_num, neg_num]

# TODO Extend stat file by adding Delta
def get_metrix_lite(sortd_predict, MODEL, predict_low):
  #SHOWN_COLUMNS = get_close_target_columns(MODEL)
  #SHOWN_COLUMNS.append(PRED_COLUMN)
  #SHOWN_COLUMNS.append(DELTA_COLUMN)
  #print("SHOWN_COLUMNS:", SHOWN_COLUMNS)

  filtered_df = sortd_predict[(sortd_predict['pred'] >= predict_low)] 
  #print(filtered_df[SHOWN_COLUMNS].to_string())
  top20_delte = sortd_predict[DELTA_COLUMN].head(20).sum() 
  top10_delte = sortd_predict[DELTA_COLUMN].head(10).sum()
  top05_delte = sortd_predict[DELTA_COLUMN].head(5).sum()
  filt_pos_df = filtered_df[(filtered_df[DELTA_COLUMN] > 0)]
  filt_neg_df = filtered_df[(filtered_df[DELTA_COLUMN] < 0)]
  avg_pos = filt_pos_df[DELTA_COLUMN].mean()
  avg_neg = filt_neg_df[DELTA_COLUMN].mean()
  pos_num = len(filt_pos_df)
  neg_num = len(filt_neg_df)
  print("TOP20_SUM=", top20_delte)    
  print("TOP10_SUM=", top10_delte)
  print("TOP05_SUM=", top05_delte)        
  print("Avrg(pred>0.1)_pos=", avg_pos, "Size:", pos_num)
  print("Avrg(pred>0.1)_neg=", avg_neg, "Size:", neg_num)
  return [top20_delte, top10_delte, top05_delte, avg_pos, avg_neg, pos_num, neg_num]

def test_dataset_ticker(DATASET_DIR, MODEL, model_name):
  predict_high = 1
  predict_low  = 0.1
  TEST_STAT_OUT_FILE = STAT_TEST_PREFIX + model_name[0:-4] + ".csv"
  print("TEST_STAT_OUT_FILE:", TEST_STAT_OUT_FILE)

  test_stat_df = pd.DataFrame(columns=TEST_STAT_COLUMNS)
 
  print("Test dir:", DATASET_DIR, "Model name:", model_name)
  model = load_model(model_name, MODEL)
  
  COMMON_LENGTH = MODEL.p_days+MODEL.t_days
  DATASET_COLUMNS = get_dataset_columns(MODEL)  
  DATASET_COLUMNS.append(DELTA_COLUMN)  
  print(DATASET_COLUMNS)
  print("DS length:", len(DATASET_COLUMNS))
  LEARN_COLUMNS_NUM = MODEL.p_days*len(MODEL_COLUMNS)
  ALL_COLUMNS_NUM = COMMON_LENGTH*len(MODEL_COLUMNS)
  LEARN_COLUMNS = get_learn_columns_clean(MODEL)
  print("LEARN_COLUMNS:\n", LEARN_COLUMNS)  
  print("Learn columns number:", LEARN_COLUMNS_NUM )
  print("All columns number:", ALL_COLUMNS_NUM )                   
 
  dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
  
  index_file = DATASET_DIR + '/'+ INDEX_FILE

  index_data = pd.read_csv( index_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
  print("Index file:\n", index_data)

  classify_stat = ClassifyStat(0,0,0,0,0)

  ticker_to_test = get_ticker_name_from_model(model_name)
  print("get_ticker_name_from_model returns", ticker_to_test, " for model", model_name)
  if len(ticker_to_test) == 0:
    return ""

  ticker_file = DATASET_DIR + '/'+ ticker_to_test + '.csv'
  if not os.path.isfile(ticker_file):
   print(ticker_file, "is absent")
   return ""

  print("!!!", ticker_file, "!!!")
  ticker_data = pd.read_csv( ticker_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)

  dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
  ticker_stat = prepare_ticker_df_test(ticker_file, ticker_data, index_data, MODEL, dataset_df)

  classify_stat.positive_num = classify_stat.positive_num + ticker_stat.positive_num
  classify_stat.hl_num = classify_stat.hl_num + ticker_stat.hl_num
  classify_stat.negative_num = classify_stat.negative_num + ticker_stat.negative_num
  classify_stat.not_looses_num = classify_stat.not_looses_num + ticker_stat.not_looses_num
  classify_stat.syncup_error = classify_stat.syncup_error + ticker_stat.syncup_error
  classify_stat.skipped = classify_stat.skipped + ticker_stat.skipped
  
  #print(entry, "Length DS:", len(dataset_df))
  print("Classify stat:", classify_stat)
  print("Sum check:", (classify_stat.positive_num + classify_stat.hl_num + classify_stat.negative_num + classify_stat.not_looses_num))  

  # Check the prediction
  new_cleaned_df  = dataset_df[LEARN_COLUMNS].copy()
  new_test_features = np.array(new_cleaned_df)
  new_test_predictions = model.predict(new_test_features, batch_size=BATCH_SIZE)

  dataset_df[[PRED_COLUMN]] = new_test_predictions
  sortd_predict = dataset_df.sort_index().sort_values('pred', kind='mergesort', ascending=False)
  pd.options.display.float_format = '{:,.4f}'.format

  metrix_list = get_metrix(sortd_predict, MODEL, predict_low)

  for index, row in sortd_predict.iterrows():
    #print(sortd_predict.at[index, 'pred'])
    if sortd_predict.at[index, 'pred'] > predict_high:
      continue

    if sortd_predict.at[index, 'pred'] < predict_low:
      break
    # Collect data fror pred(0.1-1)
    out_list = [row.File, row.Type, row.Datetime, row.pred]     
    test_stat_df.loc[len(test_stat_df)] = out_list
    #print(out_list)

  #print("Sum check:", (classify_stat.positive_num + classify_stat.hl_num + classify_stat.negative_num + classify_stat.not_looses_num))  
  #dataset_df.to_csv(DS_OUT_FILE, index=False) 
  #print("TEST_DF:\n", test_stat_df)
  test_stat_df.to_csv(TEST_STAT_OUT_FILE)
  return TEST_STAT_OUT_FILE, metrix_list

def get_detailed_stat(stat_data, detailed_pred):
  stat_df = pd.DataFrame(columns=DETAILED_STAT_COLUMNS)
  ticker_list = stat_data.File.unique()
  for ticker in ticker_list:
    ticker_df = stat_data[stat_data['File'] == ticker]
    stat = get_common_stat(ticker_df, detailed_pred)
    #print(ticker, stat) 
    row_list = [ticker, stat.cl_counter, stat.hl_counter, stat.lo_counter, stat.nl_counter, stat.cl_lo, stat.clhl_lo, stat.not_looses_lo]
    stat_df.loc[len(stat_df)] = row_list

  print("Ordered by CL - BEGIN")
  print(stat_df.sort_values(by='CL', ascending=False).head(50).to_string())
  print("Ordered by CL - END")

  print("Ordered by LO - BEGIN")
  print(stat_df.sort_values(by='LO', ascending=False).head(25).to_string())
  bad_df = stat_df[stat_df['CL_LO'] < 0.75]
  bad_df = bad_df[bad_df['LO'] > 0]
  print("Bad tickers:\n", bad_df.sort_values(by='LO', ascending=False).to_string())
  lo = bad_df['LO'].sum()
  cl = bad_df['CL'].sum()  
  print("LO numbers:", lo, "CL numbers:", cl, "Num of tickers:", len(bad_df))

  # Drop "bad" tickers
  in_len = len(stat_data)
  for index_dt, row in bad_df.iterrows():
    ticker = bad_df.at[index_dt, 'File']
    stat_data = stat_data[stat_data['File'] != ticker]

  print("IN len:", in_len, "OUT len", len(stat_data))
  return stat_data


def get_common_stat(stat_data, detailed_pred):
  test_stat = TestStat()
  test_stat.pred = detailed_pred 
  test_stat.cl_counter = 0
  test_stat.hl_counter = 0
  test_stat.lo_counter = 0
  test_stat.nl_counter = 0
  test_stat.all_cases  = 0
  test_stat.not_loses  = 0
  for index_dt, row in stat_data.iterrows():
    if float(stat_data.at[index_dt, 'pred']) >= test_stat.pred:
      test_stat.all_cases = test_stat.all_cases + 1

      if stat_data.at[index_dt, 'Type'][0:2] == "CL":
        test_stat.cl_counter = test_stat.cl_counter + 1
      elif stat_data.at[index_dt, 'Type'][0:2] == "HL":
        test_stat.hl_counter = test_stat.hl_counter + 1

      if stat_data.at[index_dt, 'Delta'] < 0:
        test_stat.lo_counter = test_stat.lo_counter + 1
      else:
        test_stat.nl_counter = test_stat.nl_counter + 1

  test_stat.not_loses = test_stat.cl_counter + test_stat.hl_counter + test_stat.nl_counter
  if test_stat.lo_counter > 0:
    test_stat.cl_lo   = round(test_stat.cl_counter/test_stat.lo_counter,3)
    test_stat.clhl_lo = round((test_stat.cl_counter+test_stat.hl_counter)/test_stat.lo_counter,3)
    test_stat.not_looses_lo = round((test_stat.nl_counter)/test_stat.lo_counter,3)
 
  return test_stat



def get_common_stat_ext(stat_data, detailed_pred, TYPE_CLMN="Type", PRED_CLMN="pred", reverse=False):
  print("get_common_stat_ext: ", detailed_pred, TYPE_CLMN, PRED_CLMN, reverse)
  #print("DEBUG:\n", stat_data) # For debug
  test_stat = TestStat()
  test_stat.pred = detailed_pred 
  test_stat.cl_counter = 0
  test_stat.hl_counter = 0
  test_stat.lo_counter = 0
  test_stat.nl_counter = 0
  test_stat.all_cases  = 0
  test_stat.not_loses  = 0
  for index_dt, row in stat_data.iterrows():
    if (reverse == False and float(stat_data.at[index_dt, PRED_CLMN]) >= test_stat.pred) or (reverse == True and float(stat_data.at[index_dt, PRED_CLMN]) <= test_stat.pred):
      #print("DEBUG ", test_stat.pred)
      test_stat.all_cases = test_stat.all_cases + 1
      test_stat.delta += stat_data.at[index_dt, "Delta"]
      #print("DEBUG: ", stat_data.at[index_dt, TYPE_CLMN])
      #if(stat_data.at[index_dt, TYPE_CLMN] is None or stat_data.at[index_dt, TYPE_CLMN] == "NaN"):
      #  print("AHTUNG: TYPE_CLMN is None: ", stat_data.at[index_dt, TYPE_CLMN])      
      if stat_data.at[index_dt, TYPE_CLMN][0:2] == "CL":
        test_stat.cl_counter = test_stat.cl_counter + 1
      elif stat_data.at[index_dt, TYPE_CLMN][0:2] == "HL":
        test_stat.hl_counter = test_stat.hl_counter + 1

      if stat_data.at[index_dt, "Delta"] < 0:
        test_stat.lo_counter = test_stat.lo_counter + 1
      else:
        test_stat.nl_counter = test_stat.nl_counter + 1

  #print("DEBUG ", test_stat)
  test_stat.not_loses = test_stat.cl_counter + test_stat.hl_counter + test_stat.nl_counter
  if test_stat.lo_counter > 0:
    test_stat.cl_lo   = round(test_stat.cl_counter/test_stat.lo_counter,3)
    test_stat.clhl_lo = round((test_stat.cl_counter+test_stat.hl_counter)/test_stat.lo_counter,3)
    test_stat.not_looses_lo = round((test_stat.nl_counter)/test_stat.lo_counter,3)
 
  return test_stat

def process_stat_df(stat_data, pdf, final_flag, START_PRED=0.2, ticker_in="", target_pred=0):
  ANALYT_MODE = "HL" # HL or CL
  
  LO_ZERO_INDEX = 3
  target_cl_lo = 0 # CL/LO for target_pred
  pred_val_list = [0,0,0,0] # 4-th position for LO = 0
  cl_hl_p_list  = [0,0,0,0] # Stores number of CL and HL cases for specific pred
  #cl_pred_list = [1, 1.5, 2]
  cl_pred_list = [1.25, 1.5, 1.75]
  tuned_stat = pd.DataFrame(columns=DETAILED_STAT_COLUMNS)

  print("\nprocess_stat_df() called with start pred:", START_PRED, " and target_pred:", target_pred, "Length of dataset:", len(stat_data))
  print("cl_pred_list=", cl_pred_list)
  print("ANALYT_MODE=", ANALYT_MODE)
  if (len(stat_data) == 0):
    return tuned_stat, pred_val_list, cl_hl_p_list, target_cl_lo

  MAX_PRED   = 0.995
  STEP       = 0.005
  x_pred = START_PRED

  if final_flag == True:
    etalon_list = [1.5, 2, 3, 4]
  else: 
    etalon_list = [1, 1.75, 2, 2.25]  

  x_pred_list  = []
  cl_list = []
  hl_list = []
  clhl_list = []
  nl_list  = []
  lo_list  = []
  not_loses_list = [] 
  cl_lo_list  = []
  clhl_lo_list  = []
  not_looses_lo_list = []

  prev_cl_lo = 0
  prev_clhl_lo = 0
  prev_not_looses_lo = 0  
  delta_h = 0
  details_shown = False
  while x_pred <= MAX_PRED:
    test_stat = get_common_stat(stat_data, x_pred)  
    x_pred_list.append(x_pred)
    cl_list.append(test_stat.cl_counter)
    hl_list.append(test_stat.hl_counter)
    clhl_list.append((test_stat.cl_counter+test_stat.hl_counter))
    lo_list.append(test_stat.lo_counter)
    nl_list.append(test_stat.nl_counter)
    not_loses_list.append(test_stat.not_loses)

    if test_stat.lo_counter > 0: 
      prev_cl_lo = test_stat.cl_lo
      prev_clhl_lo = test_stat.clhl_lo
      prev_not_looses_lo = test_stat.not_looses_lo
    else:
      test_stat.cl_lo = prev_cl_lo 
      test_stat.clhl_lo = prev_clhl_lo
      test_stat.not_looses_lo = prev_not_looses_lo 

    cl_lo_list.append(test_stat.cl_lo)
    clhl_lo_list.append(test_stat.clhl_lo) 
    not_looses_lo_list.append(test_stat.not_looses_lo) 

    print("pred", round(x_pred, 3), "cl_num:", test_stat.cl_counter, "hl_num:", test_stat.hl_counter, "nl_num:", test_stat.nl_counter, 
          "not_loose:", test_stat.not_loses, "lo_num:", test_stat.lo_counter, "all:", test_stat.all_cases, 
          "CL/LO:", test_stat.cl_lo, "CL+HL/LO:", test_stat.clhl_lo, "CL+HL+NL/LO:", test_stat.not_looses_lo )

    if target_pred > 0 and target_cl_lo == 0 and x_pred >= target_pred:
      target_cl_lo = test_stat.cl_lo
      print("target_cl_lo:", target_cl_lo)
  
    # Setup PRED array
    for i in range(len(cl_pred_list)):
      #print("i=", i)
      if ANALYT_MODE == "CL":
        if test_stat.cl_lo >= cl_pred_list[i] and pred_val_list[i] == 0:
          print("    CL/LO =", cl_pred_list[i], "achived")
          pred_val_list[i] = x_pred
          cl_hl_p_list[i] = test_stat.cl_counter + test_stat.hl_counter
      else:
        if test_stat.clhl_lo >= cl_pred_list[i] and pred_val_list[i] == 0:
          print("    CL+HL/LO =", cl_pred_list[i], "achived")
          pred_val_list[i] = x_pred
          cl_hl_p_list[i] = test_stat.cl_counter + test_stat.hl_counter

       
    if pred_val_list[LO_ZERO_INDEX] == 0 and test_stat.lo_counter == 0:
      print("    LO =", test_stat.lo_counter, "achived")
      pred_val_list[LO_ZERO_INDEX] = x_pred
    cl_hl_p_list[LO_ZERO_INDEX] = test_stat.cl_counter + test_stat.hl_counter     

    # Get details for CL/LO = 1  
    if test_stat.cl_lo >= 0.9 and details_shown == False:
    #if test_stat.cl_lo >= 0.75 and details_shown == False: 
      details_shown = True
      print("!!!CL/LO=", test_stat.cl_lo, "detailed_pred=", x_pred)
      tuned_stat = get_detailed_stat(stat_data, x_pred)

    if test_stat.all_cases == 0:
      break

    x_pred = x_pred + STEP

  #Check if we need to show this info
  if len(ticker_in) > 0 and pred_val_list[0] == 0:
    # Bad example and so no need to show
    print("Bad example and so no need to show further info")
    return tuned_stat, pred_val_list, cl_hl_p_list, target_cl_lo       
  
  fig = plt.figure(figsize=(12, 8))
  #plt.text(x=0.5, y=0.8, s=stat_file)
  str_title = "\n" + ticker_in + " Successful and loose cases"
  plt.title(str_title)
  plt.plot(x_pred_list, cl_list, color='g', label = 'Suc by close')
  plt.plot(x_pred_list, hl_list, color='y', label = 'Suc by high/low')
  plt.plot(x_pred_list, nl_list, color='b', label = 'Not looses')
  plt.plot(x_pred_list, lo_list, color='r', label = 'Looses')
  plt.plot(x_pred_list, not_loses_list, color='k', label = 'Not looses sum')
  plt.rcParams["figure.figsize"]=(12, 8)
  plt.legend()
  pdf.savefig()
  plt.close(fig)
  
  fig = plt.figure(figsize=(12, 8))       
  plt.title('CL1 and LO1 delta')
  plt.plot(x_pred_list, cl_lo_list, color='g', label = 'CL/LO')
  plt.plot(x_pred_list, clhl_lo_list, color='y', label = 'CL+HL/LO')
  plt.plot(x_pred_list, not_looses_lo_list, color='k', label = 'CL+HL+NL/LO')
  plt.hlines(etalon_list, min(x_pred_list), max(x_pred_list), linestyle='--', color='r')
  plt.hlines([0], min(x_pred_list), max(x_pred_list), linestyle='solid', color='k') # Zero horizont
  if target_pred > 0:
    plt.vlines(target_pred, 0, max(not_looses_lo_list), linestyles ="dotted", colors ="b")
    pred_str = "Pred: " + str(target_pred) + "; CL/LO: " + str(target_cl_lo)
    plt.text(0.01, 0.9, pred_str, transform=fig.transFigure, size=12)

  ### Draw details statistics ###
  type_list = np.array(stat_data['Type'].tolist())
  pred_list = np.array(stat_data['pred'].tolist()) 
  for i in range(len(pred_list)):
    if pred_list[i] < START_PRED:
      continue
    col = "k"
    l_styles = "solid"
    type_e = type_list[i][0:2]
    day_e  = float(type_list[i][2:3]) + 1
    #print(type_e, day_e)
    val = 2/day_e
    if type_e == "CL":
      l_styles = "solid"
    elif type_e == "HL":
      l_styles ="dashed"
    elif type_e == "NL":
      l_styles ="dotted"
      col = "y"
    elif type_e == "LO":
      l_styles ="solid"
      col = "r"
      val = -0.5

    plt.vlines(pred_list[i], 0, val, linestyles=l_styles, colors = col)
  ### Draw details statistics ###

  plt.legend()
  pdf.savefig()
  plt.close(fig)

  print("Ticker:", ticker_in)
  print("CL/LO list:", cl_pred_list, "and LO = 0")
  print("pred_val_list", pred_val_list)
  print("cl_hl_p_list", cl_hl_p_list)

  with open(RESULT_STAT_FILE, "a") as f:
    print("Ticker:", ticker_in, file=f)
    print("CL/LO list:", cl_pred_list, "and LO = 0", file=f)
    print("pred_val_list", pred_val_list, file=f)
    print("cl_hl_p_list", cl_hl_p_list, file=f)

  return tuned_stat, pred_val_list, cl_hl_p_list, target_cl_lo


def do_stat_analytics(stat_file, tune_flag, START_PRED):
  print(stat_file, START_PRED)
	
  OUTPUT_PDF_FILE = stat_file[0:-4] + ".pdf"
  stat_data = pd.read_csv( stat_file,  names=TEST_STAT_COLUMNS, header=None, skiprows=[0])

  pdf = PdfPages(OUTPUT_PDF_FILE)

  with open(RESULT_STAT_FILE, "a") as f:
    print("\nStat file:", stat_file, file=f)

  rc = process_stat_df(stat_data, pdf, False, START_PRED)
  stat_data = rc[0]
  pred_list = rc[1]
  cl_hl_p_list = rc[2]
  pred_list.extend(cl_hl_p_list) # Join both lists 
  # Used for aggregate data
  #if tune_flag == True:
  #  process_stat_df(stat_data, pdf, True, START_PRED)

  pdf.close() 
  return pred_list


def do_stat_analytics_separate(model_name, stat_file, tune_flag, START_PRED):
  predict_low = 0.05

  MODEL = get_MODEL_by_model_name(model_name)
  OUTPUT_PDF_FILE = stat_file[0:-4] + "_sep.pdf"
  pdf = PdfPages(OUTPUT_PDF_FILE)
  OUT_STAT_FILE = "final_stat_" + stat_file[0:-4] + ".csv"
  print("OUT_STAT_FILE=", OUT_STAT_FILE)
 
  stat_df = pd.DataFrame(columns=PRED_STAT_COLUMNS_OLD)

  with open(RESULT_STAT_FILE, "a") as f:
    print("\nStat file:", stat_file, file=f)

  stat_data = pd.read_csv( stat_file,  names=TEST_STAT_COLUMNS, header=None, skiprows=[0])
  print(stat_file, START_PRED, "LEN:", len(stat_data))  
  #print(stat_data.head(10))

  ticker_list = stat_data.File.unique()
  print(ticker_list) 
  counter = 0
  good_examples = 1
  for ticker in ticker_list:
    counter+=1
    #print(ticker, "\n", stat_data)   
    ticker_df = stat_data[stat_data['File'] == ticker]
    shift_id = ticker.find("/")
    if shift_id != -1:
      ticker_clean = ticker[shift_id+1:-4]
    else:      
      ticker_clean = ticker
    print(counter, ticker_clean, "LEN:", len(ticker_df))
    
    ticker_in = str(good_examples) + "_" + ticker
    rc = process_stat_df(ticker_df, pdf, False, START_PRED, ticker_in)
    st_data = rc[0]
    pred_list = rc[1]
    cl_hl_p_list = rc[2]
    pred_list.extend(cl_hl_p_list) # Join both lists
    if pred_list[0] > 0:
      good_examples+=1

      row_list = []
      row_list.append(ticker_clean)
      row_list.append(model_name) 
      row_list.extend(pred_list)  
      # TODO  
      #metrix_list = get_metrix_lite(ticker_df, MODEL, predict_low)
      #row_list.extend(metrix_list)
      stat_df.loc[len(stat_df)] = row_list

    # Used for aggregate data
    #if tune_flag == True:
    #  process_stat_df(stat_data, pdf, True, START_PRED)

  pdf.close() 

  #print(stat_df)
  stat_df = stat_df.sort_index().sort_values('CL_HL_P1', kind='mergesort', ascending=True)
  stat_df.loc['Total'] = stat_df[PRED_COLUMNS_LIST_OLD].sum()
  print("Final stat:\n", stat_df.to_string())
  stat_df.to_csv(OUT_STAT_FILE)

  return 0
