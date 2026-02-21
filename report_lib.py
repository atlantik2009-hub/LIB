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

from typing import List

#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *
from loader_lib import *
from intra_monitor_lib import *
from intra_test_lib import *
from intra_common_lib import * 

FIRST_DATE = "2025-09-01"
LAST_DATE  = "2026-01-01"  

TICKERS_FILE="tickers.csv"

#EXCEPTION_LIST = ["ARVL", "AMTI", "AWH", "BLI", "LYLT", "SIVB", "MSGE", "BBBY", "SLQT", "AI", "DLO", "TTCF", "VXRT", "VSMO"] # MSGE data absent for some dates

EXCEPTION_LIST = []

RESULT_PRED_STAT_COLUMNS = ['Ticker', 'Model', 'Pred', 'Price', 'Target', 'Entry_date', 'Vol_avg', 'Vol_pre', 'Vol_MLN_RUB', 'Classify', 'Delta']
TICKERS_STAT_DF_COLUMNS = ['Ticker', 'Cases', 'Delta', 'CL', 'NL', 'LO']


TOP5_STR = 'T5_'
TOP10_STR = 'T10_'
TOP15_STR = 'T15_'

TOP_ARRAY = [TOP5_STR, TOP10_STR, TOP15_STR]
LO_CLMN    = 'LO_F'
CL_CLMN    = 'CL_F'
DELTA_CLMN = 'DLT_F'
DESC_CLMN  = 'DSC_F'

# Columns types: LO, CL, Delta, Description
LO_REP_CLMN  = 0
CL_REP_CLMN  = 1
DLT_REP_CLMN = 2
DSC_REP_CLMN = 3

REPORT_FILTERS_NUM = 11

REPORT_STAT_COLUMNS = [LO_CLMN, CL_CLMN, DELTA_CLMN, DESC_CLMN]
REPORT_STAT_COLUMNS_MATH = [LO_CLMN, CL_CLMN, DELTA_CLMN]


def init_last_date(stock):
  if stock == MOEX_STOCK:
    index_file = "DS_MOEX\\INDEX.csv"
  elif stock == NY_STOCK or stock == CRYPT_STOCK:
    index_file = "DS_NY\\INDEX.csv"

  global LAST_DATE
  index_date = pd.read_csv( index_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
  last_date  = index_date.index[len(index_date)-1]
  LAST_DATE = str(last_date)[0:10]
  print("init_last_date(): LAST_DATE=", LAST_DATE)
  return LAST_DATE

def get_report_clmn_name_by_enum(clmn):
  clmn_str = "NA"
  if clmn == LO_REP_CLMN:
    clmn_str = LO_CLMN
  elif clmn == CL_REP_CLMN:
    clmn_str = CL_CLMN
  elif clmn == DLT_REP_CLMN:
    clmn_str = DELTA_CLMN
  elif clmn == DSC_REP_CLMN:
    clmn_str = DESC_CLMN

  return clmn_str

# Return column name in format T5_LO_F7 for example
def get_report_clmn_name(clmn, top, filter_i):
  clmn_str = get_report_clmn_name_by_enum(clmn)
  #return ('T' + str(top) + '_'+ clmn_str + str(filter_i))
  return (str(top) + clmn_str + str(filter_i))


def get_report_stat_columns_filter(filter_i, math=None):
  if math is None:
    clmn_list = REPORT_STAT_COLUMNS
  else:
    clmn_list = REPORT_STAT_COLUMNS_MATH

  filter_columns = [] # Index
  for top in TOP_ARRAY:
    for column in clmn_list:
      column_i = top + str(column) + str(filter_i)
      filter_columns.append(column_i)
  #print("get_report_stat_columns_filter() returns for ", filter_i, "filters:\n", filter_columns)
  return filter_columns

def get_report_stat_columns(filters_num, math=None):
  filter_columns = [INDEX_COL]
  for filt_i in range(filters_num):
    filt_i_array = get_report_stat_columns_filter(filt_i, math)
    filter_columns.extend(filt_i_array)
  #print("get_report_stat_columns() returns for", filters_num, "filters:\n", filter_columns, "\n")
  return filter_columns

def get_report_stat_columns_by_clmn(filters_num, rep_clmn, top_in=None):
  if top_in is None:
    top_array = TOP_ARRAY
  else:
    top_array = [top_in]

  filter_columns = [INDEX_COL]
  for top in top_array:
    for filt_i in range(filters_num):
      filt_i_array = get_report_clmn_name(rep_clmn, top, filt_i)
      #print(filt_i_array)
      filter_columns.append(filt_i_array)
  #print("get_report_stat_columns_by_clmn() returns for", filters_num, "filters:\n", filter_columns, "\n")
  return filter_columns


def check_price_target(price, target, ticker_df, MODEL):
  print("MODEL to test", MODEL)
  rc_array = classify_ds_ext(ticker_df, MODEL)
  rc    = rc_array[0]
  delta = rc_array[1]
  print("Result:", rc, delta, "\n" )    
  return rc_array

def show_top_df(runtime_df):
  rc_array = [] 
  for i in [5, 10, 15]:
    cl_p = 0
    hl_p = 0
    nl_p = 0
    lo_p = 0
    delta = 0
    desc_str = ""
    top_df = runtime_df.head(i)
    j = len(top_df)  
    if j > 0: 
      top_df_cl = len(top_df[(top_df['Classify'] == 'CL0')])
      top_df_hl = len(top_df[(top_df['Classify'] == 'HL0')])
      top_df_lo = len(top_df[(top_df['Classify'] == 'LO0')])
      top_df_nl = len(top_df[(top_df['Classify'] == 'NL0')])
      delta = sum(top_df['Delta'])
      cl_p = round(100*top_df_cl/j, 0)
      hl_p = round(100*top_df_hl/j, 0)
      nl_p = round(100*top_df_nl/j, 0)
      lo_p = round(100*top_df_lo/j, 0) 
      desc_str = "TOP" + str(i) + "/" + str(j) + ":" + "CL:" + str(top_df_cl) + "/" + str(cl_p) + "%. HL:" + str(top_df_hl) + "/" + str(hl_p) + "%. NL:" + str(top_df_nl) + "/" + str(nl_p) + "%. LO:" + str(top_df_lo) + "/" + str(lo_p) + "%"
      print( desc_str )
      print( "Delta:", delta)

    rc_array.append(lo_p)
    rc_array.append(cl_p)
    rc_array.append(delta)
    rc_array.append(desc_str)
 
  return rc_array

def show_report_df(stat_df):
  print("== show_report_df() BEGIN ==")
  print(stat_df.to_string())

  for clmns in [LO_REP_CLMN, CL_REP_CLMN, DLT_REP_CLMN, DSC_REP_CLMN]: 
    CLMNS_TO_SHOW = get_report_stat_columns_by_clmn(REPORT_FILTERS_NUM, clmns)
    print(stat_df[CLMNS_TO_SHOW].to_string())

  print("== TOP5 info ==")
  for clmns in [LO_REP_CLMN, CL_REP_CLMN, DLT_REP_CLMN, DSC_REP_CLMN]: 
    CLMNS_TO_SHOW = get_report_stat_columns_by_clmn(REPORT_FILTERS_NUM, clmns, TOP5_STR)
    print(stat_df[CLMNS_TO_SHOW].to_string())

  print("== TOP10 info ==")
  for clmns in [LO_REP_CLMN, CL_REP_CLMN, DLT_REP_CLMN, DSC_REP_CLMN]: 
    CLMNS_TO_SHOW = get_report_stat_columns_by_clmn(REPORT_FILTERS_NUM, clmns, TOP10_STR)
    print(stat_df[CLMNS_TO_SHOW].to_string())

  print("== show_report_df() END ==")

def show_monitor_df(runtime_df):
  print("show_monitor_df(): runtime_df\n", runtime_df.head())
  entry_date = runtime_df.iloc[0]['Entry_date']
  print("Entry_date:", entry_date)

  clmns = get_report_stat_columns(REPORT_FILTERS_NUM)

  stat_df = pd.DataFrame(columns=clmns)

  # Filter counter = 11
  counter = 0
  all_result = [entry_date]
  runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
  print(counter, "Final stat:\n", runtime_df.to_string())
  result = show_top_df(runtime_df)
  all_result.extend(result)

  counter += 1
  runtime_df = runtime_df.sort_index().sort_values('Vol_avg', kind='mergesort', ascending=False)
  print(counter, "TOP30 Last per average volume:\n", runtime_df.head(30).to_string())
  result = show_top_df(runtime_df)
  all_result.extend(result)
  
  counter += 1
  runtime_df = runtime_df.sort_index().sort_values('Vol_pre', kind='mergesort', ascending=False)
  print(counter, "TOP30 previous volume:\n", runtime_df.head(30).to_string())
  result = show_top_df(runtime_df)
  all_result.extend(result)

  counter += 1
  runtime_df = runtime_df.sort_index().sort_values('Delta', kind='mergesort', ascending=False)        
  print(counter, "TOP30 delta:\n", runtime_df.head(30).to_string())
  result = show_top_df(runtime_df)
  all_result.extend(result)

  counter += 1
  runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
  filtered_df = runtime_df[(runtime_df['Pred'] >= 0.1) & (runtime_df['Vol_avg'] >= 0.85)] 
  print(counter, "Filtered (Sorted by Pred, Pred>=0.1 & Vol_avg>=0.85):\n", filtered_df.to_string()) 
  result = show_top_df(filtered_df)
  all_result.extend(result)

  counter += 1 
  runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
  filtered_df = runtime_df[(runtime_df['Pred'] >= 0.2) & (runtime_df['Vol_avg'] > 1.5) & (runtime_df['Vol_pre'] > 1.5)] 
  print(counter, "Filtered (Sorted by Pred, Pred>=0.2, Vol_avg > 1.5, Vol_pre > 1.5):\n", filtered_df.to_string()) 
  result = show_top_df(filtered_df)
  all_result.extend(result)

  counter += 1
  runtime_df = runtime_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
  filtered_df = runtime_df[(runtime_df['Pred'] >= 0.2) & (runtime_df['Vol_avg'] > 2) & (runtime_df['Vol_pre'] > 2)] 
  print(counter, "Filtered (Sorted by Pred, Pred>=0.2, Vol_avg > 2, Vol_pre > 2):\n", filtered_df.to_string()) 
  result = show_top_df(filtered_df)
  all_result.extend(result)

  pred_list = [0.15, 0.2, 0.25, 0.3]
  for i in pred_list: 
    runtime_df = runtime_df.sort_index().sort_values('Vol_pre', kind='mergesort', ascending=False)
    filtered_df = runtime_df[(runtime_df['Pred'] >= i) & (runtime_df['Vol_avg'] > 1) & (runtime_df['Vol_pre'] > 1)] 
    counter += 1
    print(counter, "Filtered (Sorted by Vol_pre, Pred>=", i, ":\n", filtered_df.head(30).to_string())
    result = show_top_df(filtered_df)
    all_result.extend(result)

  print( "Final report:\n", clmns, "\n", all_result)
  stat_df.loc[len(stat_df)] = all_result

  show_report_df(stat_df)

  return stat_df

def filter_monitor_df(monitor_df):
  print("filter_monitor_df() len=", len(monitor_df))
  if os.path.isfile(TICKERS_FILE):
    df_tickers = pd.read_csv(TICKERS_FILE, names=['Ticker'], index_col=False)
    tickers_list = np.array(df_tickers['Ticker'].tolist())

    for id, row_id in monitor_df.iterrows():
      ticker = monitor_df.at[id, 'Ticker']
      if ticker not in tickers_list:
        monitor_df = monitor_df.drop(id)
        print(ticker, "dropped")       
  #else:
  print("= Bad tickers =")       
  #Drop bad tickers  
  for id, row_id in monitor_df.iterrows():
    ticker = monitor_df.at[id, 'Ticker']
    if ticker in EXCEPTION_LIST:
      monitor_df = monitor_df.drop(id)
      print(ticker, "dropped")       

  print("filter_monitor_df() len=", len(monitor_df))
  return monitor_df

def get_report(INPUT_FILE, PRED_THRESHOLD = 0.025):
  DAY_TO_CHECK = 1
  OUTPUT_FILE = INPUT_FILE[0:-4] + "_RES.csv" 
  
  if INPUT_FILE[0:2] == "MO":
    stock=MOEX_STOCK
    INTRA_MODE = "24"
    index = MOEX_INDEX 
  elif INPUT_FILE[0:2] == "CR":
    stock=NY_STOCK
    INTRA_MODE = "1d"
    index = CRYPT_INDEX 
  else:
    stock=NY_STOCK 
    INTRA_MODE = "1d"
    index = SP500_INDEX
  
  clmns = get_report_stat_columns(REPORT_FILTERS_NUM)
  
  test_df = pd.DataFrame(columns=RUNTIME_PRED_STAT_COLUMNS)
  monitor_df = pd.read_csv(INPUT_FILE, names=RUNTIME_PRED_STAT_COLUMNS, skiprows = 1, index_col=False)

  monitor_df = filter_monitor_df(monitor_df)

  len_1 = len(monitor_df)
  monitor_df = monitor_df[(monitor_df['Pred'] >= PRED_THRESHOLD)]
  len_2 = len(monitor_df)
  if len_2 == 0:
    return None
  
  monitor_df["Classify"] = ""
  monitor_df["Delta"] = 0.00
  
  print("PRED_THRESHOLD=", PRED_THRESHOLD)
  print("Input file data:\n", monitor_df.to_string(), "\nLength:", len_2, "Original length:", len_1)
  
  entry_date = monitor_df.at[monitor_df.index[0],'Entry_date']
  print("Input file:", INPUT_FILE, "Stock:", stock, "Entry_date:", entry_date)

  model_name = monitor_df.at[monitor_df.index[0],'Model']
  rc = get_params(model_name)
  if rc[0] == False:
    print(model_name, "is NOT ok")
  else:
    DAY_TO_CHECK = int(rc[4])

  print("Model:", model_name, "DAY_TO_CHECK=", DAY_TO_CHECK)

  #end_date = str(date.today() + timedelta(days=1))
  end_date = LAST_DATE
  index_data = load_df_ext(index, entry_date, end_date, INTRA_MODE, stock)
  if(index_data is None):
    print("ERROR:", index, "NOT loaded")   
    sys.exit("Index NOT loaded.")
   
  print(index, "loaded")
  print(index_data.tail(10), "\nSize:", len(index_data))
  if len(index_data) > DAY_TO_CHECK:
    day_check = str(index_data.index[DAY_TO_CHECK])[0:10]
    print("DAY_TO_CHECK:", DAY_TO_CHECK, "-", day_check)
    #row = index_data.iloc[DAY_TO_CHECK]
    #print(str(row.to_frame().T))
    #test_df.loc[day_check, :] = row
    #print("OUT:\n", test_df)
  if stock == MOEX_STOCK:
    if len(index_data) <= DAY_TO_CHECK:
      print("AHTUNG: Dirty hack for future date len(index_data)=", len(index_data), "DAY_TO_CHECK=", DAY_TO_CHECK)
      start_date = index_data.index[0]
      end_date = index_data.index[0]
    else:
      start_date = index_data.index[1]
      end_date = index_data.index[DAY_TO_CHECK]
  else:
    start_date = str(index_data.index[1])[0:10]
    end_date = str(index_data.index[DAY_TO_CHECK])[0:10]
  
  print("Entry date:", entry_date, "Check period:", start_date, "-", end_date)
  
  for id, row_id in monitor_df.iterrows():
    ticker     = monitor_df.at[id, 'Ticker']
    price      = monitor_df.at[id, 'Price']
    target     = monitor_df.at[id, 'Target']
    entry_date = monitor_df.at[id, 'Entry_date']
    model_name = monitor_df.at[id, 'Model']
    rc = get_params(model_name)
    if rc[0] == False:
      print(model_name, "is NOT ok")
      continue
    MODE    = rc[1]
    PERCENT = float(rc[2])
    MODEL   = ModelSettings(p_days=1, t_days=DAY_TO_CHECK, time_i='1d', mode = MODE, percent = PERCENT, neg_delta=0)
  
    #print("\n===", ticker, "===")
    try_i = 0
    while( try_i <= 4 ):
      ticker_df = load_df_ext(ticker, entry_date, end_date, INTRA_MODE, stock)
      if(ticker_df is None):
        print(ticker, "NOT loaded properly. Try:", try_i )
        try_i += 1
        time.sleep((try_i*3))
        continue
      else:
        break
    if( ticker_df is None or len(ticker_df) < 2 ):
      print("FIXME - ERROR:", ticker, "NOT loaded properly")
      # To think NOT GOOD but anyway     
      monitor_df.at[id, 'Classify'] = "NL0"
      monitor_df.at[id, 'Delta']    = 0
      continue
  
    print(ticker_df)
    print("Entry date:", entry_date, "Entry price:", price, "Target price:", target)
    rc = check_price_target(price, target, ticker_df, MODEL)
    monitor_df.at[id, 'Classify'] = rc[0]
    monitor_df.at[id, 'Delta']    = rc[1]
  
  stat_df = show_monitor_df(monitor_df)
  monitor_df.to_csv(OUTPUT_FILE, index=False)
  return stat_df
  

def get_index_df(start_date, end_date, stock):
  INTRA_MODE ="1d"
  end_d      = ""
  if stock == NY_STOCK:
    end_d = str(datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))[0:10]
    index_df = load_df_ext(SP500_INDEX, start_date, end_d, INTRA_MODE, NY_STOCK) 
  elif stock == CRYPT_STOCK:
    end_d = str(datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))[0:10]
    index_df = load_df_ext(CRYPT_INDEX, start_date, end_d, INTRA_MODE, NY_STOCK) 
  else:
    end_d = end_date
    index_df = load_df(MOEX_INDEX, start_date, end_d, INTRA_MODE, MOEX_STOCK)
  return index_df

def get_pairs_date(period_start, period_end, days, stock):
  print("get_pairs_date(", period_start, period_end, days, stock, ")")
  array_start = []
  array_end   = []
  index_df = get_index_df(FIRST_DATE, period_end, stock)
  index_period_df = get_index_df(period_start, period_end, stock)
  print(index_period_df.to_string(), "\nLen:", len(index_period_df))

  for id in range(len(index_period_df)):
    end_date = str(index_period_df.index[id])[0:10]
    rc = get_dates_by_end_date(end_date, days, stock, index_df)
    print(rc)
    if rc is None:
      continue
    array_start.append(rc[0])
    array_end.append(rc[1])
  return array_start, array_end

### FROM test_res_analytics_1d.py ###
TOP_X = 5

# Sum of All cases 
# If topX=5 only 20 percents are allocated for a position per day.
class ACCUM_DELTA_MODE(Enum):
  SIMPLE  = 0 
  PART    = 1
  ALL_DEP = 2 


@dataclass
class TestStatList:
  pred      : List[float]
  cl_counter: List[int]
  hl_counter: List[int]
  lo_counter: List[int]
  nl_counter: List[int]
  all_cases : List[int]
  cl_lo     : List[float]
  clhl_lo   : List[float]
  not_looses_lo : List[float]
  delta     : List[float]

  def add_elem(self, test_stat):
    self.pred.append(test_stat.pred)
    self.cl_counter.append(test_stat.cl_counter)
    self.hl_counter.append(test_stat.hl_counter)
    self.lo_counter.append(test_stat.lo_counter)
    self.nl_counter.append(test_stat.nl_counter)
    self.all_cases.append(test_stat.all_cases)
    self.cl_lo.append(test_stat.cl_lo)
    self.clhl_lo.append(test_stat.clhl_lo)
    self.not_looses_lo.append(test_stat.not_looses_lo)
    self.delta.append(test_stat.delta)


@dataclass
class AlgoConf:
  sort_clmn: str
  filt_clmn: str
  min_val  : float = 0
  max_val  : float = 0
  PRED_MIN : float = 0.1
  PRED_MAX : float = 1
  topX     : int = 5
  mode     : ACCUM_DELTA_MODE = ACCUM_DELTA_MODE.ALL_DEP
  top5_full: bool = False # False provide better result. Now this feature is disabled
  result   : float = 0
  cases    : int = 0
  def set_top5_full(self, flag):
    self.top5_full = flag 

  def get_mode_desc(self):
    if self.mode == ACCUM_DELTA_MODE.ALL_DEP:
      return "ALL_DEP"
    elif self.mode == ACCUM_DELTA_MODE.PART:
      return "PART"
    elif self.mode == ACCUM_DELTA_MODE.SIMPLE:
      return "SIMPLE"
    return "N/A"

  def get_details(self):
    #out_str = "AlgoConf(result=" + str(round(self.result,2)).rjust(6, ' ') + "; sort_clmn=" + self.sort_clmn + ", filt_clmn=" + self.filt_clmn + ", min_val=" + str(round(self.min_val,3))+ ", max_val=" + str(round(self.max_val,3)) + ", PRED_MIN=" + str(round(self.PRED_MIN,3)) + ", PRED_MAX=" + str(round(self.PRED_MAX,3)) + ", cases=" + str(self.cases) + ", Top5_full=" + str(self.top5_full) + ", topX=" + str(self.topX) + ", mode=" + self.get_mode_desc() + ")"
    out_str = "AlgoConf(result=" + str(round(self.result,2)).rjust(6, ' ') + "; sort_clmn=" + self.sort_clmn + ", filt_clmn=" + self.filt_clmn + ", min_val=" + str(round(self.min_val,3))+ ", max_val=" + str(round(self.max_val,3)) + ", PRED=" + str(round(self.PRED_MIN,3)) + "-" + str(round(self.PRED_MAX,3)) + ", cases=" + str(self.cases) + ", topX=" + str(self.topX) + ", mode=" + self.get_mode_desc() + ")"
    return out_str

  def get_final_line(self):
    Vol_pre_s = -1
    Vol_pre_e = -1
    Vol_avg_s = -1
    Vol_avg_e = -1

    if self.filt_clmn == 'Vol_pre':
      Vol_pre_s = round(self.min_val, 3)
      Vol_pre_e = round(self.max_val, 3)
    elif self.filt_clmn == 'Vol_avg':
      Vol_avg_s = round(self.min_val, 3)
      Vol_avg_e = round(self.max_val, 3)

    # ['Ticker', 'Model', 'Period',     'Pred_s', 'Pred_e', 'Vol_pre_s', 'Vol_pre_e', 'Vol_avg_s', 'Vol_avg_e', "Mode",     "cl_counter", "hl_counter", "lo_counter", "nl_counter", "all_cases", "cl_lo", "clhl_lo", "not_looses_lo=", "delta", 'Result']
    out_str = str(round(self.PRED_MIN,3)) + "," + str(round(self.PRED_MAX,3)) + "," + str(Vol_pre_s) + "," + str(Vol_pre_e) + "," + str(Vol_avg_s) + "," + str(Vol_avg_e) + "," + str(self.mode.value)
    return out_str


  def get_title(self):
    title = "### Top" + str(self.topX) + " day sorted by " + self.sort_clmn + " and filtered by " + self.filt_clmn + " Top5_full=" + str(self.top5_full) + " ###"
    return title

  def get_title_full(self):
    title = self.get_title()
    title = title + str(self.min_val) + "-" + str(self.max_val)
    return title

  def do_task(self, final_stat_df, pdf):
    self.init_setup()
    step = 0.05
    if self.filt_clmn == 'Pred':
      step = 0.005
   
    title = self.get_title()
    print("do_task: ", self.get_details())

    # IMPORTANT: here we filter by all params
    # Firstly filter by params and then collect TOP examples
    if self.top5_full == False:
      filtered_stat_df = final_stat_df[(final_stat_df['Pred'] >= self.PRED_MIN) & (final_stat_df['Pred'] <= self.PRED_MAX)]
      top5_df = get_topX_by_clmn(filtered_stat_df, self.topX, filt_clmn=self.filt_clmn, filt_val_min=self.min_val, filt_val_max=self.max_val)
    else:
      len_1 = len(final_stat_df)
      filtered_stat_df = final_stat_df[(final_stat_df['Pred'] >= self.PRED_MIN) & (final_stat_df['Pred'] <= self.PRED_MAX) & (final_stat_df[self.filt_clmn] >= self.min_val) & (final_stat_df[self.filt_clmn] <= self.max_val)]
      len_2 = len(filtered_stat_df)
      print("do_task: df size updated from ", len_1, " into ", len_2) 
      top5_df = get_topX_by_clmn_without_filter(filtered_stat_df, self.topX)

    if(len(top5_df) == 0):
      print("Warning1: empty df!!!")
      return None  

    stat_list_into_title_page(top5_df, pdf, title)
    rc = get_stat_by_clmn(top5_df, self.min_val, self.max_val, step, self.filt_clmn, pdf)
    if(len(top5_df) == 0):
      print("Warning2: empty df!!!")
      return None  

    title = self.get_title_full() + "\n" + "Preliminary statistics: " + str(rc[2])
    if self.top5_full == False:
      if rc[0] > self.min_val:
        print("!!! min_val tuned from", self.min_val, " into ", rc[0])
        self.min_val = rc[0]
      ret = draw_delta_accum(self.mode, top5_df, pdf, title, self.filt_clmn, self.min_val, self.max_val)
    else:
      ret = draw_delta_accum(self.mode, top5_df, pdf, title) 
 
    if(ret is None):
      print("Warning3: empty df!!!")
      return None  

    self.result = ret[0]
    self.cases = len(ret[1])
    return ret[0], ret[1], rc[2] # accum_delta, result_df, test_stat

  def init_setup(self):
    #if self.sort_clmn == 'Pred' and self.filt_clmn == 'Pred':
    #  self.PRED_MIN = self.min_val 
    #  self.PRED_MAX = self.max_val
    #elif self.sort_clmn != 'Pred':
    #  self.PRED_MIN = self.min_val 
    #  self.PRED_MAX = self.max_val
    return False

    print("Sorted params:", self.sort_clmn, self.PRED_MIN, self.PRED_MAX)

  def do_task_simple(self, final_stat_df, pdf):
    self.init_setup()
    step = 0.05
    if self.filt_clmn == 'Pred':
      step = 0.005
   
    title = self.get_title_full()
    print(title)
    top5_df = get_topX_by_clmn(final_stat_df, self.topX, filt_clmn=self.sort_clmn, filt_val_min=self.PRED_MIN, filt_val_max=self.PRED_MAX)
    if(len(top5_df) == 0):
      print("Warning5: empty df!!!")
      return None  

    stat_list_into_title_page(top5_df, pdf, title)

    # Get statistics and exit on the firstExit
    firstExit = True
    rc = get_stat_by_clmn(top5_df, self.min_val, self.max_val, step, self.filt_clmn, pdf, firstExit)
    if(len(top5_df) == 0):
      print("Warning3: empty df!!!")
      return None  

    ret = draw_delta_accum(self.mode, top5_df, pdf, title, self.filt_clmn, self.min_val, self.max_val)
    self.result = ret[0]
    self.cases = len(ret[1])
    return ret[0], ret[1], rc[2] # accum_delta, result_df, test_stat

  def get_accum_delta(self, final_stat_df):
    self.init_setup()
    date_list = final_stat_df.Entry_date.unique()
    filtered_stat_df = get_topX_by_clmn(final_stat_df, self.topX, filt_clmn=self.sort_clmn, filt_val_min=self.PRED_MIN, filt_val_max=self.PRED_MAX)
    if(len(filtered_stat_df) == 0):
      print("Warning6: empty df!!!")
      return None  

    ret_code = get_accum_details(filtered_stat_df, date_list, self.mode)
    accum_delta  = ret_code[0]
    case_counter = ret_code[1]
    case_counter = ret_code[2]
    str_details  = ret_code[3]
    accum_list   = ret_code[4]
    delta_list   = ret_code[5]
    return accum_list, delta_list, date_list

def filter_by_clmn(final_stat_df, filt_clmn=None, filt_val_min=None, filt_val_max=None):
  print("get_topX_by_clmn:", filt_clmn, filt_val_min, filt_val_max)  
  date_list = final_stat_df.Entry_date.unique()

  # Filter df by value column 
  if filt_clmn is None:
    filtered_stat_df = final_stat_df
    sorted_clmn = 'Pred'
  elif filt_val_max is None:
    filtered_stat_df = final_stat_df[(final_stat_df[filt_clmn] >= filt_val_min)]
    sorted_clmn = filt_clmn
  else:
    filtered_stat_df = final_stat_df[(final_stat_df[filt_clmn] >= filt_val_min) & (final_stat_df[filt_clmn] <= filt_val_max)]
    sorted_clmn = filt_clmn
  
  return sorted_clmn, filtered_stat_df


def get_topX_by_clmn_without_filter(filtered_stat_df, top_x, filt_clmn=None):
  print("get_topX_by_clmn_without_filter:", filt_clmn)  
  date_list = filtered_stat_df.Entry_date.unique()

  if filt_clmn is None:
    sorted_clmn = 'Pred'
  else:
    sorted_clmn = filt_clmn

  dataset_df = pd.DataFrame(columns=RESULT_PRED_STAT_COLUMNS)
  print(date_list, "\nLen=", len(date_list))
  for day in date_list:
    filtered_df = filtered_stat_df[(filtered_stat_df['Entry_date'] == day)]
    filtered_df = filtered_df.sort_index().sort_values(sorted_clmn, kind='mergesort', ascending=False).head(top_x)
    filtered_df = filtered_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False) # Sorted by Pred by default
    dataset_df = pd.concat([dataset_df, filtered_df], axis=0)    

  print("dataset_df:\n", dataset_df.to_string(), "Len=", len(dataset_df))
  print("Raw delta:", dataset_df['Delta'].sum())
  return dataset_df

def get_topX_by_clmn(final_stat_df, top_x, filt_clmn=None, filt_val_min=None, filt_val_max=None):
  print("get_topX_by_clmn:", filt_clmn, filt_val_min, filt_val_max)  
  date_list = final_stat_df.Entry_date.unique()

  rc = filter_by_clmn(final_stat_df, filt_clmn, filt_val_min, filt_val_max)
  sorted_clmn = rc[0]
  filtered_stat_df = rc[1]

  dataset_df = pd.DataFrame(columns=RESULT_PRED_STAT_COLUMNS)
  print(date_list, "\nLen=", len(date_list))
  for day in date_list:
    filtered_df = filtered_stat_df[(filtered_stat_df['Entry_date'] == day)]
    filtered_df = filtered_df.sort_index().sort_values(sorted_clmn, kind='mergesort', ascending=False).head(top_x)
    filtered_df = filtered_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False) # Sorted by Pred by default
    dataset_df = pd.concat([dataset_df, filtered_df], axis=0)    

  print("dataset_df:\n", dataset_df.to_string(), "Len=", len(dataset_df))
  print("Raw delta:", dataset_df['Delta'].sum())
  return dataset_df


def get_mode_str(mode):
  if mode == ACCUM_DELTA_MODE.SIMPLE:
    return "(Simple sum of percents)"
  elif mode == ACCUM_DELTA_MODE.PART:
    return "(Only part (20%) of deposit per position allocated. Compound interest calculated)"
  elif mode == ACCUM_DELTA_MODE.ALL_DEP:
    return "(All deposit allocated. Compound interest calculated)"

  return "(Unknown mode)"


def get_accum_details(filtered_stat_df, date_list, mode):
  print(date_list)
  accum_delta = 0
  accum_list = []
  delta_list = [] 
  case_counter = 0 
  deposit = 1
  str_details=""
  for day in date_list:
    day_delta = 0
    day_delta_base = 0
    filtered_df = filtered_stat_df[(filtered_stat_df['Entry_date'] == day)].head(TOP_X)
    filtered_df.sort_index().sort_values('Pred', kind='mergesort', ascending=False)
    if mode == ACCUM_DELTA_MODE.SIMPLE:
      day_delta_base = filtered_df['Delta'].sum()
      day_delta = day_delta_base
    elif mode == ACCUM_DELTA_MODE.ALL_DEP:
      if len(filtered_df) > 0:
        day_delta_base = filtered_df['Delta'].sum()/len(filtered_df) 
        day_delta = deposit*day_delta_base    
      deposit += day_delta/100
    elif mode == ACCUM_DELTA_MODE.PART:
      day_delta_base = filtered_df['Delta'].sum()/TOP_X
      day_delta = deposit*day_delta_base
      deposit += day_delta/100

    accum_delta += day_delta

    case_counter+=len(filtered_df)
    str_day = ""
    str_title = "Accumulated calculation details" + get_mode_str(mode)
    df_str = ""
    if len(filtered_df) > 0:
      df_str = filtered_df.to_string()
    if mode == ACCUM_DELTA_MODE.SIMPLE:
      str_day = day + " " + str(len(filtered_df)) + " Day delta: " + str(day_delta_base) + " Compound Day delta: " + str(day_delta) + " Accum delta: " + str(accum_delta) + " Case counter: " + str(case_counter) + "\n" + df_str
    else:
      str_day = day + " " + str(len(filtered_df)) + " Day delta: " + str(day_delta_base) + " Compound Day delta: " + str(day_delta) + " Accum delta: " + str(accum_delta) + " Case counter: " + str(case_counter) + " Deposit: " + str(deposit) + "\n" + df_str
    str_details+=str_day + "\n"
   
    accum_list.append(accum_delta)
    delta_list.append(day_delta)

  return accum_delta, case_counter, case_counter, str_details, accum_list, delta_list


def draw_delta_accum(mode, final_stat_df, pdf, str_title, filt_clmn=None, filt_val=None, filt_val_max=None):
  print("draw_delta_accum", mode, filt_clmn, filt_val, filt_val_max)  
  date_list = final_stat_df.Entry_date.unique()

  filtered_stat_df = get_topX_by_clmn(final_stat_df, TOP_X, filt_clmn=filt_clmn, filt_val_min=filt_val, filt_val_max=filt_val_max)
  if(len(filtered_stat_df) == 0):
    print("WARNING: Empty df can not be shown!!!")
    return None

  # Filter df by value column 
  if filt_clmn is None:
    # Do nothing
    print("No conditions to filter")
  elif filt_val_max is None:
    str_title += "\n" + filt_clmn + ">=" + str(filt_val)
  else:
    str_title += "\n" + filt_clmn + ">=" + str(filt_val) + "&" + filt_clmn + "<=" + str(filt_val_max)

  print("str_title:", str_title)
  print("filtered_stat_df:\n", filtered_stat_df, "Len=", len(filtered_stat_df))
  print("Raw delta:", filtered_stat_df['Delta'].sum())

  stat_list_into_title_page(filtered_stat_df, pdf, str_title)
  # Cases dataset detailed info
  txt_into_page(filtered_stat_df, pdf, str_title)

  str_title = "Accumulated calculation details" + get_mode_str(mode)
  ret_code = get_accum_details(filtered_stat_df, date_list, mode)
  accum_delta  = ret_code[0]
  case_counter = ret_code[1]
  case_counter = ret_code[2]
  str_details  = ret_code[3]
  accum_list   = ret_code[4]
  delta_list   = ret_code[5]

  print(str_title, "\n", str_details)
  only_txt_into_page(pdf, str_title, str_details, 8, 3)
  str_title+= "\n" + get_mode_str(mode) + " Accum delta: "  + str(accum_delta) + " Case counter:" + str(case_counter)

  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.95, str_title, transform=fig.transFigure, size=9)    

  plt.plot(date_list, accum_list, color='g', label = "Accumulated delta")
  plt.plot(date_list, delta_list, color='b', label = "Delta")
  plt.axhline(y = 0.0, color = 'k', linestyle = '-')
  plt.legend()
  pdf.savefig()
  plt.close(fig)
  return accum_delta, filtered_stat_df

def draw_test_stat_list(stat_list, title_str, pdf):  
  title1 = ["cl_counter", "hl_counter", "lo_counter", "nl_counter", "all_cases"]
  y1 = [stat_list.cl_counter, stat_list.hl_counter, stat_list.lo_counter, stat_list.nl_counter, stat_list.all_cases]
  title2 = ["cl_lo", "clhl_lo", "not_looses_lo"]
  y2 = [stat_list.cl_lo, stat_list.clhl_lo, stat_list.not_looses_lo]
  title3 = ["delta"]
  y3 = [stat_list.delta] 
  colours = ['g', 'b', 'r', 'k', 'y']
  print(title_str)
  for i in [title1, title2, title3]:
    title = i
    fig = plt.figure(figsize=(12, 8))
    plt.text(0.01, 0.95, title_str, transform=fig.transFigure, size=9)    
    #plt.title('Ticker Close, High, Low')
    if i == title1:
      y_axis = y1
    elif i == title2:
      y_axis = y2
    elif i == title3:
      y_axis = y3
 
    counter = 0
    for y in y_axis: 
      plt.plot(stat_list.pred, y, color=colours[counter], label = title[counter])
      counter+=1
    plt.legend()
    pdf.savefig()
    plt.close(fig)

def get_stat_by_clmn(final_stat_df, start_val, end_val, step, clmn, pdf=None, firstExit=False):
  print("### get_stat_by_clmn: Ordered by", clmn, "###", start_val, end_val, step)
  test_stat_list = TestStatList([],[],[],[],[],[],[],[],[],[])
  x_pred = start_val
  max_delta = -100
  x_axis = 0
  test_stat_out = TestStat()
  while x_pred <= end_val:
    test_stat = get_common_stat_ext(final_stat_df, x_pred, "Classify", clmn)
    print(x_pred, test_stat) 
    if firstExit == True:
      print("Exit by flag firstExit")
      max_delta = test_stat.delta
      x_axis = x_pred
      test_stat_out = test_stat 
      break
      
    if max_delta <= test_stat.delta:
      max_delta = test_stat.delta
      x_axis = x_pred
      test_stat_out = test_stat 
    if test_stat.all_cases == 0:
      break
    test_stat_list.add_elem(test_stat)
    x_pred = x_pred + step

  first_line = "Searching " + clmn + " axis value that corresponds MAX delta. X is increased [" + str(start_val) + "..." + str(end_val) + "]\n" 
  str_lable = first_line + clmn + " statistics:" + "\n" + "Max simple delta=" + str(max_delta) + ". " + clmn + "=" + str(x_axis)
  print(str_lable)
  if pdf is not None and firstExit == False:
    print("Drawing statistics by draw_test_stat_list()")
    draw_test_stat_list(test_stat_list, str_lable, pdf)

  return x_axis, max_delta, test_stat_out  

def get_stat_by_clmn_rev(final_stat_df, start_val, end_val, step, clmn, pdf=None):
  print("### get_stat_by_clmn_rev: Ordered by", clmn, "###")
  test_stat_list = TestStatList([],[],[],[],[],[],[],[],[],[])
  x_pred = end_val
  max_delta = -1000
  x_axis = 0
  test_stat_out = TestStat()
  final_stat_df = final_stat_df[(final_stat_df[clmn] >= start_val)]
  while x_pred >= start_val:
    test_stat = get_common_stat_ext(final_stat_df, x_pred, "Classify", clmn, True)
    print(x_pred, test_stat) 
    if max_delta < test_stat.delta:
      max_delta = test_stat.delta
      x_axis = x_pred
      test_stat_out = test_stat 
    if test_stat.all_cases == 0:
      break
    test_stat_list.add_elem(test_stat)
    x_pred = x_pred - step

  first_line = "Searching " + clmn + " axis value that corresponds MAX simple delta. X is decreased [" + str(end_val) + "..." + str(start_val) + "]\n" 
  str_lable = first_line + clmn + " reverse statistics:" + "\n" + "Max simple delta=" + str(max_delta) + ". " + clmn + "=" + str(x_axis)
  #print(str_lable)
  if pdf is not None:
    draw_test_stat_list(test_stat_list, str_lable, pdf)
  return x_axis, max_delta, test_stat_out  


def txt_into_page(final_stat_df, pdf, title=""):
  str_title = get_str_title(final_stat_df)

  plt.rcParams['font.family'] = 'Serif'
  fig = plt.figure(figsize=(12, 9))
  fig.clf()
  fig.text(0.05, 0.95, title, transform=fig.transFigure, size=7)
  fig.text(0.05, 0.89, str_title, transform=fig.transFigure, size=5)    
  fig.text(0.05, 0.87, final_stat_df.to_string(), horizontalalignment='left', verticalalignment='top', transform=fig.transFigure, size=3)    
   
  pdf.savefig()
  plt.close()


def get_str_title(final_stat_df):
  pred_min = final_stat_df['Pred'].min()
  pred_max = final_stat_df['Pred'].max()

  vol_pre_min = final_stat_df['Vol_pre'].min()
  vol_pre_max = final_stat_df['Vol_pre'].max()

  vol_avg_min = final_stat_df['Vol_avg'].min()
  vol_avg_max = final_stat_df['Vol_avg'].max()

  days_len = len(final_stat_df.Entry_date.unique())
  days_range = str(final_stat_df.at[final_stat_df.index[0], 'Entry_date']) + "..." + final_stat_df.at[final_stat_df.index[len(final_stat_df) - 1], 'Entry_date']

  str_title = "Pred: " + str(pred_min) + "..." + str(pred_max) + "\n"
  str_title += "Vol_pre: " + str(vol_pre_min) + "..." + str(vol_pre_max) + "\n"
  str_title += "Vol_avg: " + str(vol_avg_min) + "..." + str(vol_avg_max) + "\n"
  str_title += "Cases: " + str(len(final_stat_df)) + "\n"
  str_title += "Days range: " + days_range + " Days:" + str(days_len) + "\n"
  str_title += "Raw delta: " + str(final_stat_df['Delta'].sum()) + "\n"
  str_title += "Min Vol_MLN_RUB: " + str(final_stat_df['Vol_MLN_RUB'].min())
  return str_title

def stat_list_into_title_page(final_stat_df, pdf, title=""):
  model_name = ""
  if len(title) > 0:
    model_name = title + "\n"  
  model_name += final_stat_df.at[final_stat_df.index[0], 'Model']

  str_title = get_str_title(final_stat_df)

  plt.rcParams['font.family'] = 'Serif'
  fig = plt.figure(figsize=(12, 9))
  fig.clf()
  fig.text(0.5, 0.9, model_name, transform=fig.transFigure, size=7, ha="center")
  fig.text(0.1, 0.7, str_title, transform=fig.transFigure, size=7)    
   
  pdf.savefig()
  plt.close()


def get_pdf_stat_file_name(model_name, start=None, end=None, postfix=""):
  str_mask = "model_ds_01d_"
  id = model_name.find(str_mask) + len(str_mask)

  if start is None: 
    OUTPUT_PDF_FILE = model_name[id:-4] + "_" + datetime.now().strftime('%H%M%S') + postfix + ".pdf" 
  else:
    start_end = "_"+ start.replace('-', '') + "_" + end.replace('-', '') + "_"
    OUTPUT_PDF_FILE = model_name[id:-4] + start_end + datetime.now().strftime('%H%M%S') + postfix + ".pdf" 
   
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  pdf_file = OUT_DIR + '/'+ OUTPUT_PDF_FILE

  return pdf_file


def draw_delta_from_x(final_stat_df, pdf, start_val=0.1, end_val=1, step=0.05, clmn='Pred', desc_str=''):
  print("draw_delta_from_pred", start_val, end_val, step, clmn)  
  str_title='Delta from ' + clmn + ": " + str(start_val) + "-" +  str(end_val) + "(" + str(step) + ")" + desc_str

  final_stat_df = final_stat_df[(final_stat_df[clmn] >= start_val) & (final_stat_df[clmn] <= end_val)]

  xlist = []
  ylist = []
  delta_norm_l = []
  ylen = []
  min_v  = start_val
  offset = start_val 
  while offset < end_val:
    min_v = offset
    if(min_v >= end_val):
      break;

    offset = round(offset + step, 2)
    filter_df = final_stat_df[(final_stat_df[clmn] > min_v) & (final_stat_df[clmn] <= offset)]
    num_of_cases = len(filter_df)
    delta = 0
    delta_norm = 0
    if num_of_cases != 0: 
      delta_norm = sum(filter_df['Delta'])/num_of_cases
      delta = sum(filter_df['Delta'])   
    print(min_v, "-", offset, "Delta:", delta, "Size:", num_of_cases)
    if num_of_cases == 0:
      break; 

    xlist.append(offset)
    ylist.append(delta)
    delta_norm_l.append(delta_norm)
    ylen.append(num_of_cases)

  fig = plt.figure(figsize=(12, 8))    
  fig.clf()
  plt.title(str_title)
  plt.plot(xlist, ylen, color='k', label = 'Num of cases per range')
  # display (x, y) values next to each point
  for i, x in enumerate(xlist):
    y = ylen[i]
    range_x = str(round((xlist[i]-step), 2)) + "-" + str(round(xlist[i],2))
    shift = 0
    if (end_val-start_val)/step >= 20:
      shift = (i%5)* max(ylen)/15
    elif(end_val-start_val)/step >= 12:
      shift = (i%3)* max(ylen)/15

    plt.text(x, y+shift, f"{y:.0f}({range_x})", horizontalalignment="center")

  plt.legend()
  pdf.savefig()
  plt.close()


  fig = plt.figure(figsize=(12, 8))    
  fig.clf()
  plt.title(str_title)

  for id, row_id in final_stat_df.iterrows():
    x_val  = final_stat_df.at[id, clmn]
    delta = final_stat_df.at[id, 'Delta']
    colour = 'g'
    if delta < 0:
      colour = 'r'
    plt.vlines(x_val, 0, delta, linestyles='solid', colors=colour)

  pdf.savefig()
  plt.close()


  fig = plt.figure(figsize=(12, 8))    
  fig.clf()
  plt.title(str_title)
  plt.rcParams["figure.figsize"] = [12,8]
  fig, ax1 = plt.subplots()

  ax1.set_title(str_title)
  color = 'tab:green'
  ax1.set_xlabel(clmn)
  ax1.set_ylabel('Normalized sum of delta per range(Sum/Num of cases)', color=color)
  ax1.plot(xlist, delta_norm_l, color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.grid(True)
  for i, x in enumerate(xlist):
    y = delta_norm_l[i]
    range_x = str(round((xlist[i]-step), 2)) + "-" + str(round(xlist[i],2))
    plt.text(x, y, f"{y:.1f}({range_x})", horizontalalignment="center")

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.set_ylabel('Sum of delta per range', color=color)  # we already handled the x-label with ax1
  ax2.plot(xlist, ylist, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  pdf.savefig()
  #plt.close()
  plt.close('all')

def draw_delta_batch(final_stat_df, pdf):
  avg_pred = final_stat_df["Pred"].mean()
  if avg_pred < 0.1:
    draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=0.1, step=0.005, clmn='Pred')
  elif avg_pred < 0.2:
    draw_delta_from_x(final_stat_df, pdf, start_val=0.1, end_val=0.32, step=0.01, clmn='Pred')
  elif avg_pred < 0.3:
    draw_delta_from_x(final_stat_df, pdf, start_val=0.1, end_val=0.5, step=0.01, clmn='Pred')

  draw_delta_from_x(final_stat_df, pdf, start_val=0.1, end_val=1, step=0.1, clmn='Pred') 

  draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=50, step=2.5, clmn='Vol_pre')
  draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=10, step=0.5, clmn='Vol_pre')
  draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=1, step=0.1, clmn='Vol_pre')

  draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=50, step=2.5, clmn='Vol_avg')
  draw_delta_from_x(final_stat_df, pdf, start_val=0, end_val=1, step=0.1, clmn='Vol_avg')
  draw_delta_from_x(final_stat_df, pdf, start_val=1.5, end_val=16.4, step=1, clmn='Vol_avg') # Short case


def draw_delta_batch_tickers(final_stat_df, pdf, desc_str="Detailed tickers statistics"):
  stat_list_into_title_page(final_stat_df, pdf, desc_str)
  rc_df = get_ticker_details(final_stat_df)
  only_txt_into_page(pdf, "Ordered by Delta", rc_df.to_string(), 10, 6)

  ticker_list = final_stat_df.Ticker.unique()
  ticker_list_sort = sorted(ticker_list, key=str.lower)
 
  for ticker_in in ticker_list_sort:
    ticker_df = final_stat_df[(final_stat_df['Ticker'] == ticker_in)]
    draw_delta_from_x(ticker_df, pdf, start_val=0.1, end_val=1, step=0.1, clmn='Pred', desc_str=ticker_in)
    #draw_delta_from_x(ticker_df, pdf, start_val=0, end_val=10, step=1, clmn='Vol_pre', desc_str=ticker_in)
    #draw_delta_from_x(ticker_df, pdf, start_val=0, end_val=10, step=1, clmn='Vol_avg', desc_str=ticker_in)


def get_ticker_details(stat_df):
  tickers_stat_df = pd.DataFrame(columns=TICKERS_STAT_DF_COLUMNS)
  ticker_list = stat_df.Ticker.unique()
  for ticker_in in ticker_list:
    ticker_df = stat_df[(stat_df['Ticker'] == ticker_in)]
    ticker_sz = len(ticker_df)
    delta_sum = sum(ticker_df['Delta'])
    cl_num = len(ticker_df[(ticker_df['Classify'] == 'CL0')])
    nl_num = len(ticker_df[(ticker_df['Delta'] >= 0)])
    lo_num = len(ticker_df[(ticker_df['Delta'] < 0)])
    tickers_stat_df.loc[len(tickers_stat_df)] = [ticker_in, ticker_sz, delta_sum, cl_num, nl_num, lo_num]

  tickers_stat_df = tickers_stat_df.sort_index().sort_values('Delta', kind='mergesort', ascending=False)
  return tickers_stat_df



### INTRADAY ANALYTICS ###
# Return x axis and normalized y axes
def get_normalized_index_close_list(ticker, entry_day, test_day, INTRA_MODE, stock):
  if stock != MOEX_STOCK:
    print("!!!AHTUNG!!! ", stock, " is NOT supported")
    return None

  index_df = load_df_ext(ticker, test_day, test_day, INTRA_MODE, stock)
  index_list = index_df.index.values.tolist()
  index_y_list = index_df.Close.values.tolist()
  
  index_entry_df = load_df_ext(ticker, entry_day, entry_day, "1d", stock)
  index_close = index_entry_df.iloc[0]['Close']
  print(ticker, " entry price: ", index_close)
  if(index_close == 0):
    print("!!!AHTUNG!!! ", ticker, " price is 0.")
    return None
 
  for i in range(len(index_y_list)):
    index_y_list[i] = index_y_list[i]/index_close
  #print(index_list)
  #print(index_y_list)
  #print(ticker, "\n", index_df)
  return index_list, index_y_list

def get_normalized_ticker_close_list(ticker, entry_day, test_day, INTRA_MODE, stock, index_x):
  ticker_entry_df = load_df_ext(ticker, entry_day, entry_day, "1d", stock)
  ticker_close = ticker_entry_df.iloc[0]['Close']
  print(ticker, " entry price: ", ticker_close)
  if(ticker_close == 0):
    print("!!!AHTUNG!!! ", ticker, " price is 0.")
    return None

  df = load_df_ext(ticker, test_day, test_day, INTRA_MODE, stock)
  #print(ticker, "\n", df)
  close_list = []
  if df is None:
    print("AHTUNG!!! No data for", ticker)
    for time_st in index_x:
      close_list.append(1)
    return close_list 

  prev_val = 0  
  for time_st in index_x:
    if time_st in df.index:
      prev_val = df.loc[time_st, "Close"]/ticker_close
      close_list.append(prev_val)
    else:
      print(time_st, " is absent for ", ticker, ". Using previous value:", prev_val)
      close_list.append(prev_val)
   
  return close_list

def draw_intraday_data(ticker_list, entry_day, test_day, pdf, stock=MOEX_STOCK, INTRA_MODE="10"):
  rc = get_normalized_index_close_list(MOEX_INDEX, entry_day, test_day, INTRA_MODE, stock)
  index_x = rc[0]
  index_y = rc[1]
  print("MIN index:", min(index_y), "MAX index:", max(index_y))
 
  close_list = []
  tickers_array = []
  ind = 0
  for ticker in ticker_list:
    close_list = get_normalized_ticker_close_list(ticker, entry_day, test_day, INTRA_MODE, stock, index_x)
    tickers_array.append(close_list)
    ind+=1

  print("Num of tickers:", ind)

  str_title = "Index, tickers prices change." + " Entry day:" + str(entry_day) + " Target day:" + str(test_day) 

  axis_x = index_x.copy()
  for j in range(len(index_x)):
    axis_x[j] = index_x[j][11:-3] 
   
  fig = plt.figure(figsize=(12, 8))    
  fig.clf()
  #plt.title("Index, tickers prices change")
  plt.rcParams["figure.figsize"] = [12,8]
  fig, ax1 = plt.subplots()
  
  ax1.set_title(str_title)
  color = 'black'
  ax1.set_xlabel('Timestamps 10M')
  ax1.set_ylabel('Index close price', color=color)
  ax1.plot(axis_x, index_y, color=color, label="Index")
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.grid(True)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:green'
  color_set = ['tab:red', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:cyan' ]
  ax2.set_ylabel('Ticker close price', color=color)  # we already handled the x-label with ax1
  final_y = []
  for i in range(ind):
    print("index: ", i) 
    close_list = tickers_array[i]
    close = round(close_list[len(close_list) - 1], 3)
    label_str = ticker_list[i] + " High: " +  str(round(max(close_list), 3)) + " Close:" + str(close)
    ax2.plot(axis_x, close_list, color=color_set[i], label=label_str)
   
    # Get sum of tickers
    if len(final_y) == 0:
      final_y.extend(close_list)
    else:
      for j in range(len(final_y)):
        final_y[j] += close_list[j]
  
  # Normalize final array
  for j in range(len(final_y)):
    final_y[j] = final_y[j]/ind
  
  average_max = round(max(final_y), 3)
  average_min = round(min(final_y), 3)
  close_price = round(final_y[len(final_y) - 1], 3)
  open_price  = str(round(final_y[0], 3)) + "/" + str(round(final_y[1], 3))
  max_id = final_y.index(max(final_y))
  min_id = final_y.index(min(final_y))
  average_str = "Average, Open " + str(open_price) + " High " + str(average_max) + " at " + str(axis_x[max_id]) + "\nLow " + str(average_min) + " at " + str(axis_x[min_id]) + "\nClose:" + str(close_price)
  ax2.plot(axis_x, final_y, color=color, label=average_str)
  ax2.plot([axis_x[max_id], axis_x[max_id]], [min(final_y), average_max], linestyle='dashed', color="green")
  ax2.plot([axis_x[min_id], axis_x[min_id]], [min(final_y), average_max], linestyle='dashed', color="red")
  plt.legend()
   
  ax2.tick_params(axis='y', labelcolor=color)
  ticks = []
  for i, x in enumerate(axis_x):
    if i%3 == 0:
      ticks.append(x)
  plt.xticks(ticks)
 
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  pdf.savefig()
  #plt.close()
  plt.close('all')
  return open_price, average_min, index_x[min_id], average_max, index_x[max_id], close_price, index_x, final_y

def get_next_day(entry_day):
  ticker_entry_df = load_df_ext(MOEX_INDEX, entry_day, LAST_DATE, "1d", stock=MOEX_STOCK)
  #print("get_next_day()\n", ticker_entry_df)

  #entry_day = datetime.strptime(entry_day, '%Y-%m-%d')
  entry_day += " 00:00:00" 

  date_list = ticker_entry_df.index.to_list()
  #print(date_list)
  if entry_day in date_list:
    entry_id = date_list.index(entry_day)
    #print("entry_id=", entry_id)
    if entry_id >= len(date_list):
      print("!!! AHTUNG: it is last date")
      return None
    else:
      return date_list[entry_id+1][0:10]
  else:
    print("!!! AHTUNG: it is NOT in the index", entry_day)
  return None


# Collect intraday statistics
def collect_intraday_stat(max_df, pdf, details_str=""): 
  print("DEBUG: collect_intraday_stat: ", len(max_df))
  axis_x = []
  axis_y = []
  axis_y_comp = []
  counter = 0
  intra_stat_df = pd.DataFrame(columns=["Date", "Open", "Low", "Low_TM", "High", "High_TM", "Close"])
  date_list = max_df.Entry_date.unique()
  for entry_day in date_list:
    test_day  = get_next_day(entry_day)
    filtered_df = max_df[(max_df['Entry_date'] == entry_day)]
    print(entry_day, "-", test_day, "\n", filtered_df)
    tickers_list = filtered_df.Ticker.to_list()
    rc = draw_intraday_data(tickers_list, entry_day, test_day, pdf)
    index_x = rc[6]
    value_y = rc[7]

    #print("value_y:", value_y) 
    intra_stat_df.loc[len(intra_stat_df)]=[test_day, rc[0], rc[1], rc[2], rc[3], rc[4], rc[5]]
    if len(axis_x) == 0:
      axis_x = index_x # TODO now it stores day and timestamp of the first test day

    if len(axis_y) == 0:
      axis_y      = value_y.copy() # TODO now it accumulates data
      axis_y_comp = value_y.copy()
      counter+=1
      print("axis_y: ", axis_y[0], axis_y[1], axis_y[len(axis_y)-1], "\naxis_y_comp: ", axis_y_comp[0], axis_y_comp[1], axis_y_comp[len(axis_y_comp)-1])
    else:
      if len(axis_y) == len(value_y):
        counter+=1
        #print("BEFORE axis_y: ", axis_y[0], axis_y[1], axis_y[len(axis_y)-1], "\naxis_y_comp: ", axis_y_comp[0], axis_y_comp[1], axis_y_comp[len(axis_y_comp)-1])
        for i in range(len(axis_y)):         
          axis_y[i] = axis_y[i] + value_y[i] - 1
          axis_y_comp[i]= axis_y_comp[i]*value_y[i]
          #print(i, axis_y[i])
        print("axis_y: ", axis_y[0], axis_y[1], axis_y[len(axis_y)-1], "\naxis_y_comp: ", axis_y_comp[0], axis_y_comp[1], axis_y_comp[len(axis_y_comp)-1])
      else:
        print("!!!AHTUNG: size of main axis_y ", axis_y, "is NOT equal to value_y", value_y)

  # Convert data into percent
  for i in range(len(axis_y)):         
    axis_y[i] = (axis_y[i] - 1)*100
    axis_y_comp[i] = (axis_y_comp[i] - 1)*100
    axis_x[i] = axis_x[i][11:-3] 

  max_val = round(max(axis_y_comp),3)
  min_val = round(min(axis_y_comp),3)
  close_price = round(axis_y_comp[len(axis_y_comp) - 1], 3)
  max_id = axis_y_comp.index(max(axis_y_comp))
  min_id = axis_y_comp.index(min(axis_y_comp))
  postfix_str = "\nMax " + str(max_val) + " at " + str(axis_x[max_id]) + "\nmin " + str(min_val) + " at " + str(axis_x[min_id]) + "\nClose price:" + str(close_price)

  label_str = 'Sum of delta: compound percent calculated' + postfix_str
  only_txt_into_page(pdf, "Intraday statistics", intra_stat_df.to_string(), 11, 9)
  # Draw final intraday statustics
  fig = plt.figure(figsize=(12, 8))    
  fig.clf()

  default_str = "Profit in percents if position is closed at this exact time"  
  if details_str == "":
    details_str = "Profit in percents if position is closed at this exact time"
  else:
    details_str = default_str + "\n" + details_str

  plt.title(details_str, fontsize=8)
  plt.plot(axis_x, axis_y, color='b', label = 'Simple sum')
  plt.plot(axis_x, axis_y_comp, color='g', label = label_str)

  ticks = []
  for i, x in enumerate(axis_x):
    if i%3 == 0:
      y = axis_y_comp[i]
      str_x = str(x)
      plt.text(x, y, f"{y:.1f}({str_x})", horizontalalignment="center")
      plt.vlines(x, 0, max(axis_y_comp), linestyles='dotted', colors='k')
      ticks.append(x)
  plt.xticks(ticks)

  plt.legend()
  pdf.savefig()
  plt.close()

def get_dates_by_end_date(end_date, days, stock, index_df=None):
  #print("get_dates_by_end_date(", end_date, days, stock, ") called" )
  start_d = ""
  if index_df is None:
    index_df = get_index_df(FIRST_DATE, end_date, stock)
  
  #print("End date:", end_d)
  length = len(index_df)
  id = -1
  for counter in range(length):
    reverse_id = length - counter - 1
    index_val = index_df.index[reverse_id]
    #print(reverse_id, index_val)
    if str(index_val)[0:10] == end_date[0:10]:
      #print(end_date[0:10], " found") 
      id = reverse_id
      break     

  #print(id, end_date[0:10])
  #print(index_df.tail(days))
  
  if id == -1:
    print(end_date, "NOT found" ) 
    return None
  
  start_id = id - days + 1
  if start_id < 0:
    print(end_date, "NOT enough data in index_df" ) 
    return None
 
  start_d = str(index_df.index[start_id])[0:10]

  #end_d      = ""
  #if stock == NY_STOCK:
  #  end_d = str(datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))[0:10]
  #else:
  #  end_d = end_date
  #print(start_d, "-", end_d)
  return start_d, end_date

def get_previous_day(prev_day):
    rc = get_dates_by_end_date(prev_day, 2, MOEX_STOCK)
    if rc is None:
      LOGS(TRACE,"AHTUNG: get_dates_by_end_date() failed with None")
      sys.exit()
    LOGS(TRACE,"Previous day:", rc[0])
    return rc[0]
