import requests
import apimoex
import pandas as pd
import sys
import datetime

from intra_common_lib import *
from intra_monitor_lib import *
from report_lib import *

TIME_CLMN = 'Time'
VOL_CLMN  = 'Volume'
CL_CLMN   = 'Close'
Vol_Avg_CLMN = 'Vol_Avg'
PREFIX_ACC = 'Acc_'

RESULT_CLMN = 'Result'
PREV_V_CLMN = 'Prev_v'
AVG_V_CLMN  = 'Avg_v'

STAT_COUNTERS_CLMNS = ['Ticker','Open','Price1', 'Price1_date','Up','Down','All','Delta','UpDo','Vol_mln']
counters_df = pd.DataFrame(columns=STAT_COUNTERS_CLMNS)
INTERVAL=10
MAX_HUGE_VAL = 2.5
TM_10_30 = ' 10:30:00'
TM_18_40 = ' 18:40:00'
LEFT_CHART_LIST  = ['Delta','Delta_prev', "Acc_Volume", "Acc_Vol_Avg"]
RIGHT_CHART_LIST = ['Close', 'Volume', Vol_Avg_CLMN]

def get_non_nan_index(y3_list):
  ind = len(y3_list)-1 
  if pd.isna(y3_list[ind]):
    ind = max(index for index, item in enumerate(y3_list) if pd.isna(item) == False)
    LOGS(TRACE, "NON nan index:", ind)
  return ind, y3_list[ind] 

def is_left_clmn(clmn):
  if clmn in LEFT_CHART_LIST:
    return True
  return False

def is_right_clmn(clmn):
  for i in RIGHT_CHART_LIST:
    if clmn.find(i) == 0 or clmn == Vol_Avg_CLMN:
      return True
  return False

def get_end_timestamp_dsc(m30_flag):
  dsc = str(m30_flag)
  if m30_flag == 0:
    dsc += " Standard:" + TM_10_30
  elif m30_flag == 1:
    dsc += " Last available"
  elif m30_flag == 2:
    dsc += TM_18_40
  elif m30_flag == 3:
    dsc += " Current time"
  else:
    dsc += "Unknown"
  return  dsc

def get_end_timestamp(m30_flag, start_date):
  tm = TM_10_30
  # Get current DF
  if m30_flag == 1:
    df = load_df('IMOEX', start_date, start_date, 1, MOEX_STOCK)
    date_last = df.index[len(df)-1]
    tm = date_last[10:]
    LOGS(TRACE,"Available date: ", date_last)
  elif m30_flag == 2:
    tm = TM_18_40
  elif m30_flag == 3:
    tm = " " + str(datetime.now().strftime("%H:%M:00"))
  LOGS(TRACE,"m30_flag=", m30_flag, "End timestamp = ", tm)
  return tm  

def handle_one_ticker(ticker, start_date, end_date):
  #1 minute
  #print("### load_df() call ### Interval", interval)
  df = load_df(ticker, start_date, end_date, INTERVAL, MOEX_STOCK)
  if df is None or len(df) == 0: 
    return None
  if ticker == "LKOH": 
    LOGS(TRACE,df)

  ticker_close = np.array(df['Close'].tolist())
  prev = 0
  up_counter   = 0
  down_counter = 0
  for i in ticker_close:
    if i > prev:
      up_counter += 1

    if i < prev:
      down_counter += 1

    prev = i
  if ticker != "IMOEXF":
    volume_mln = round(np.array(df['Volume'].tolist()).sum()/1000000, 3)
  else:
    volume_mln = np.array(df['Volume'].tolist()).sum()

  upDo = -1 
  if down_counter != 0:
    upDo = round(up_counter/down_counter, 3)
  p_open = ticker_close[0]
  p_close = ticker_close[len(ticker_close)-1]  
  delta = round(p_close/p_open, 4) - 1
  date_close = df.index[len(ticker_close)-1]
  LOGS(TRACE,ticker, ": up_counter =", up_counter, "; down_counter =", down_counter, "; Delta =", delta, "; Seria =", len(ticker_close))  
  return ticker, p_open, p_close, date_close, up_counter, down_counter, len(ticker_close), delta, upDo, volume_mln

def collect_tickers_stat(TICKERS, start_date, end_date, counters_df):
  LOGS(TRACE,"Ticker file: ", TICKERS)
  tickers_df = pd.read_csv(TICKERS, names=["Ticker"], index_col=False)
  tickers_list = tickers_df['Ticker'].unique()
  LOGS(TRACE,"Tickers list:", tickers_list, "Length:", len(tickers_list)) 
 
  for ticker in tickers_list:
    rc = handle_one_ticker(ticker, start_date, end_date)
    if rc is not None:
      row_list = [rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], rc[7], rc[8], rc[9]]
      counters_df.loc[len(counters_df)] = row_list


def show_top_tail_stat(m30_counters_df, sort_clmn='Delta', debug_flag=False):
  m30_counters_df = m30_counters_df.sort_index().sort_values(sort_clmn, kind='mergesort', ascending=False) 
  LOGS(PROD,"Sorted by", sort_clmn)
  if debug_flag == True:
    LOGS(PROD,m30_counters_df.to_string())

  top5_df   = m30_counters_df.head(5)
  top10_df  = m30_counters_df.head(10)
  tail5_df  = m30_counters_df.tail(5)
  tail10_df = m30_counters_df.tail(10)
  pos_top5_counter  = len(top5_df[(top5_df['Result'] > 0)])
  pos_tail5_counter = len(tail5_df[(tail5_df['Result'] > 0)])

  top5  = round(top5_df['Result'].sum()*100, 1)
  top10 = round(top10_df['Result'].sum()*100, 1)
  tail5 = round(tail5_df['Result'].sum()*100, 1)
  tail10= round(tail10_df['Result'].sum()*100, 1)
  LOGS(PROD, "top5=", top5, ",(+", pos_top5_counter,"),  Len=", len(top5_df),"top10=", top10, "Len=", len(top10_df))
  LOGS(PROD, "tail5=", tail5, ",(+", pos_tail5_counter,"), Len=", len(tail5_df), "tail10=", tail10, "Len=", len(tail10_df))

def add_column(clmn, TICKERS, m30_counters_df, start_date, end_date):
  LOGS(TRACE,"add_column: ", clmn, start_date, end_date)  
  m30_counters_df[clmn] = 0
  #LOGS(TRACE,m30_counters_df.to_string())

  counters_df = pd.DataFrame(columns=STAT_COUNTERS_CLMNS)
  collect_tickers_stat(TICKERS, start_date, end_date, counters_df)

  for index, row in m30_counters_df.iterrows():
    ticker = row["Ticker"]
    if clmn == RESULT_CLMN:
      price_30m = row["Price1"]
      price_day_close = counters_df[counters_df.Ticker == ticker].iloc[0]['Price1']
      result = price_day_close/price_30m - 1

      if ticker == "LKOH": 
        day_close_date = counters_df[counters_df.Ticker == ticker].iloc[0]['Price1_date']
        date_30m = row["Price1_date"]
        LOGS(TRACE,"DEBUG: ", ticker, day_close_date, date_30m, price_day_close, price_30m, "Result:", result)
        LOGS(TRACE,"DEBUG: Column Result = Close EOD price/Close 30 minute price - 1")

    elif clmn == PREV_V_CLMN:
      vol_mln = row["Vol_mln"]
      prev_vol_mln = counters_df[counters_df.Ticker == ticker].iloc[0]['Vol_mln']
      result = vol_mln/prev_vol_mln

      if ticker == "LKOH" or ticker == "IMOEX": 
        day_close_date = counters_df[counters_df.Ticker == ticker].iloc[0]['Price1_date']
        date_30m = row["Price1_date"]
        LOGS(TRACE,"DEBUG: ", ticker, day_close_date, date_30m, vol_mln, prev_vol_mln, "Result:", result)
        LOGS(TRACE,"DEBUG: Column Prev_v = vol_mln 30 minute current day/vol_mln 30 minute previous day")

    #LOGS(TRACE,ticker, result)
    m30_counters_df.at[index,clmn] = result

def add_avg_column( TICKERS, m30_counters_df, current_date, time_in, num_days):
  LOGS(TRACE,"add_avg_column: ", current_date, time_in, num_days)  
  m30_counters_df[AVG_V_CLMN] = 0
  #LOGS(TRACE,m30_counters_df.to_string())

  # Get previous day
  days_list = []
  prev_day = current_date 
  for i in range(num_days):
    prev_day = get_previous_day(prev_day)
    days_list.append(prev_day)

    prev_day_30m = prev_day + time_in
    counters_df = pd.DataFrame(columns=STAT_COUNTERS_CLMNS)
    collect_tickers_stat(TICKERS, prev_day, prev_day_30m, counters_df)
    for index, row in m30_counters_df.iterrows():
      ticker  = row["Ticker"]
      prev_vol_mln = counters_df[counters_df.Ticker == ticker].iloc[0]['Vol_mln']
      LOGS(TRACE,ticker, prev_vol_mln)
      m30_counters_df.at[index,AVG_V_CLMN] = m30_counters_df.at[index,AVG_V_CLMN] + prev_vol_mln 

      if ticker == "LKOH" or ticker == "IMOEX": 
        day_close_date = counters_df[counters_df.Ticker == ticker].iloc[0]['Price1_date']
        date_30m = row["Price1_date"]
        LOGS(TRACE,"DEBUG: ", ticker, day_close_date, date_30m, prev_vol_mln)
        LOGS(TRACE,"DEBUG: Column Avg = vol_mln 30 minute current day/vol_mln 30 minute previous day")

  # Calculate average
  for index, row in m30_counters_df.iterrows():
    vol_mln = row["Vol_mln"]
    m30_counters_df.at[index,AVG_V_CLMN] = vol_mln/(m30_counters_df.at[index,AVG_V_CLMN]/num_days)

def filter_and_calc(m30_counters_df, filter_clmn, start_val, end_val, order_clmn, show_df=True, prepend_desc=""):
  LOGS(PROD,start_val, "=<", filter_clmn, "=<", end_val)
  if start_val is not None and end_val is not None:	
    result_df = m30_counters_df[(m30_counters_df[filter_clmn] >= start_val) & (m30_counters_df[filter_clmn] <= end_val)]
  elif start_val is None and end_val is not None:	
    result_df = m30_counters_df[(m30_counters_df[filter_clmn] <= end_val)]
  elif start_val is not None and end_val is None:	
    result_df = m30_counters_df[(m30_counters_df[filter_clmn] >= start_val)]

  result_df = result_df.sort_index().sort_values(order_clmn, kind='mergesort', ascending=False) 
  show_top_tail_stat(result_df, order_clmn, show_df)

def setup_interval(interval):
  global INTERVAL
  INTERVAL = interval
  LOGS(PROD, "Interval:", INTERVAL)

def add_acc_new_column(df, calc_clmn, new_clmn=None):
  LOGS(TRACE,"add_acc_new_column: ", calc_clmn, new_clmn)
  if new_clmn is None:
   new_clmn = PREFIX_ACC + calc_clmn
  df[new_clmn] = 0
  prev_index = None
  for index, row in df.iterrows():
    if prev_index == None:
      df.at[index,new_clmn] = row[calc_clmn]
    else:
      df.at[index,new_clmn] = df.at[prev_index, new_clmn] + row[calc_clmn]
    prev_index = index

def get_avg_stat(ticker, current_date, time_in, num_days, first_volume_skip=True):
  LOGS(TRACE,"get_avg_stat: ", ticker, current_date, time_in, num_days, first_volume_skip)  
  # Get previous day
  days_list = []
  prev_day = current_date
  avg_days_num = num_days
  if first_volume_skip == True:
    avg_days_num-=1
    LOGS(TRACE,"First Volume is NOT included in avarage", Vol_Avg_CLMN, "Avg days number", avg_days_num)  

  i = 0
  for j in range(num_days):
    if i > 0:
      prev_day = get_previous_day(prev_day)

    LOGS(TRACE,"Previous day:", prev_day)  
    prev_day_30m = prev_day + time_in
    day_df = load_df(ticker, prev_day, prev_day_30m, INTERVAL, MOEX_STOCK)
    if day_df is None or len(day_df) == 0: 
      LOGS(TRACE,"Ahtung: df is None for", ticker) 
      avg_days_num = avg_days_num - 1
      LOGS(TRACE,"avg_days_num decreased into", avg_days_num) 
      continue

    days_list.append(prev_day)

    if i == 0:
      df = day_df
      LOGS(TRACE, "First day df:\n", df.to_string())
      df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
      df[TIME_CLMN] = 0
      df[Vol_Avg_CLMN] = 0 
      for index, row in df.iterrows():
        dt = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')       
        df.at[index,TIME_CLMN] = dt.time()
        if first_volume_skip == False:
          df.at[index,Vol_Avg_CLMN] = row['Volume']
      df = df.set_index(TIME_CLMN)
    elif i > 0:
      cl_clmn = CL_CLMN + str(i)
      vol_clmn = VOL_CLMN + str(i)
      df[cl_clmn]  = 0
      df[vol_clmn] = 0        
      for index, row in day_df.iterrows():
        dt = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
        #print(index, dt.time())
        df.at[dt.time(),cl_clmn] = row['Close']
        df.at[dt.time(),vol_clmn] = row['Volume']
        if i == num_days - 1:
          df.at[dt.time(),Vol_Avg_CLMN] = round((df.at[dt.time(),Vol_Avg_CLMN] + row['Volume'])/avg_days_num, 1 )
        else:
          df.at[dt.time(),Vol_Avg_CLMN] = df.at[dt.time(),Vol_Avg_CLMN] + row['Volume']  

    i+= 1 
  #LOGS(TRACE, ticker, prev_day_30m, "\n",df)
  LOGS(TRACE, "Days list", days_list)
  return df, days_list

def normalizer(ticker_y, max_val=MAX_HUGE_VAL, max_flag=False):
  note = ""
  if max_flag == False:
    for j in range(len(ticker_y) - 1):
      if ticker_y[j] > max_val:
        if len(note) == 0 or j == len(ticker_y) - 1:
          note += str(ticker_y[j]) + " -> " + str(max_val) + ". "
        LOGS(TRACE, note)
        ticker_y[j] = max_val 
  else:
    for j in range(len(ticker_y) - 1):
      if ticker_y[j] > max_val:
        tmp = (ticker_y[j]/max(ticker_y)) * max_val
        if len(note) == 0 or j == len(ticker_y) - 1: 
          note += str(ticker_y[j]) + " -> " + str(tmp) + ". "
        ticker_y[j] = tmp  
        #LOGS(TRACE, note, y_list[i])

  return note

def draw_chart_by_df_new(ticker_df, list_clmn, title_str, pdf, normalizeFlag=2):
  #normalizeFlag=2 all axis y is normalized, 1 - only first element is normalized; 0 - no any normalization  
  LOGS(TRACE, "draw_chart_by_df_new: ", list_clmn, title_str, normalizeFlag)
  ticker_x = np.array(ticker_df.index.tolist())
  ticker_x_str = [date_obj.strftime('%H:%M') for date_obj in ticker_x]

  print(title_str)
  fig = plt.figure(figsize=(12, 8))
 
  # Hide some ticks
  divider = 1
  if len(ticker_x_str) > 15:
    divider = round(len(ticker_x_str)/10, 0)
  LOGS(TRACE, "Divider=", divider)
  new_ticker_x_str = [ticker_x_str[i] if( i%divider == 0 ) else '' for i in range(len(ticker_x_str))]
  #print("new_ticker_x_str:", new_ticker_x_str)
  #plt.xticks(new_ticker_x_str)
  ax = plt.gca()
  ax.axes.xaxis.set_ticklabels(new_ticker_x_str)
 
  plt.text(0.01, 0.95, title_str, transform=fig.transFigure, size=9)    

  i = 0
  note = ""
  colour_list = ['k', 'b', 'y', 'r', 'c', 'm']
  colour_y = 'b'

  ax2_flag = False
  for clmn in list_clmn:
    LOGS(TRACE, "Column:", clmn)
    if i <= 5:
      colour_y = colour_list[i]
    else:
      i = 0

    if is_right_clmn(clmn):
      LOGS(TRACE, "Right column:", clmn) 
      if ax2_flag == False:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2_flag = True

      y3_list = np.array(ticker_df[clmn].tolist())     
      color = 'tab:green'
      color_set = ['tab:green' ]
      label_str = clmn  
      clr = colour_list[i] 
      if clmn == 'Close':
        label_str = 'Close price'
        clr = 'g'        
      elif clmn == 'Volume':
        clr = 'g'        

      ax2.set_ylabel(label_str, color=color)  # we already handled the x-label with ax1
      ax2.plot(ticker_x_str, y3_list, color=clr, label=label_str)
      i+=1
      continue
    else:
      if clmn == 'Delta':
        label_str = 'Delte avg'
        colour_y = 'k'
      elif clmn == 'Delta_prev':
        label_str = 'Delte prev'
        colour_y = 'b'
      else:
        label_str = clmn
  
    ticker_y = np.array(ticker_df[clmn].tolist())
    if normalizeFlag == 2:
      LOGS(TRACE, "FULL normalization")
      if clmn in ["Delta", "Delta_prev"]:
        # Tune huge values for average delta:
        tmp = normalizer(ticker_y, MAX_HUGE_VAL, True)      
        if len(tmp) > 0:
          note =  note + clmn + ": " + tmp
    elif normalizeFlag == 1:
      LOGS(TRACE, "PART normalization")
      if clmn in ["Delta", "Delta_prev"]:
        if ticker_y[0] >= MAX_HUGE_VAL:
          note =  clmn + ": " + str(round(ticker_y[0], 2)) + "->" + str(MAX_HUGE_VAL)
          ticker_y[0] = MAX_HUGE_VAL
    else:
      LOGS(TRACE, "NO normalization")

    plt.plot(ticker_x_str, ticker_y, color=colour_y, label = label_str)
    i+=1
   
  xticks = []
  new_ticker_x_str = []
  for i in range(len(ticker_x_str)):
    if( i%divider == 0 ):
      xticks.append(i)
      new_ticker_x_str.append(ticker_x_str[i])
  LOGS(TRACE, "Ticks:", xticks, new_ticker_x_str)

  #TODO
  ticker = ""
  title_str = ticker + " " + note
  ax.set_title(title_str)
  
  ax.set_xticks(xticks, labels=new_ticker_x_str)
  ax.legend(loc='lower left')
  ax2.legend(loc='upper right')
  ax.tick_params(axis='y', labelcolor='k')
  ax.grid(True)
  ###

  plt.legend()
  pdf.savefig()
  plt.close(fig)


def add_delta_column(df, first_clmn, second_clmn, new_clmn):
  LOGS(TRACE,"add_delta_column: ", first_clmn, second_clmn, new_clmn)
  df[new_clmn] = 0
  for index, row in df.iterrows():
    if df.at[index, second_clmn] == 0:
      df.at[index,new_clmn] = 0
    else:  
      df.at[index,new_clmn] = round(df.at[index, first_clmn]/df.at[index, second_clmn],3 )

  ind, value = get_non_nan_index(df[new_clmn].to_list())
  LOGS(TRACE,"Last delta:", df.at[index,new_clmn], "at", index, "Tuned: ", value, "at", ind, df.index[ind])
  return value

def init_chart_settings():
  # Init data
  #x = np.linspace(0, 400, 400)
  #y = [1 for a in x]
  x=[] 
  y=[]
  plt.ion()
  ax: plt.Axes
  figure, ax = plt.subplots(figsize=(15, 12))

  (line1,) = ax.plot([], [])  
  line1.set_color("black")
  line1.set_label('Delta to average')

  (line2,) = ax.plot(x, y)  
  line2.set_color("blue")
  line2.set_label('Delta to previous')

  (line3,) = ax.plot(x, y)  
  line3.set_color("green")
  line3.set_label('Close Price')


  ax.autoscale(True)
  plt.xlabel("Time")
  plt.ylabel("Delta")

 
  return figure, ax, line1, line2, line3


def draw_new_values(x_list, y_list, figure, ax, line1, ticker_x_str):
  print("draw_new_values: Len=", len(x_list))
  line1.set_xdata(x_list)
  line1.set_ydata(y_list)
  
  # Rescale axes limits
  ax.relim()
  ax.autoscale()
  ax.set_xticks(x_list, labels=ticker_x_str)

  figure.canvas.draw()
  figure.canvas.flush_events()


def draw_new_values_ext(ticker_df, figure, ax, line1, line2, line3, ticker="N/A"):
  ticker_x = np.array(ticker_df.index.tolist())
  ticker_x_str = [date_obj.strftime('%H:%M') for date_obj in ticker_x]
  divider = 1
  if len(ticker_x_str) > 15:
    divider = round(len(ticker_x_str)/10, 0)
  LOGS(TRACE, "Divider=", divider)
  #new_ticker_x_str = [ticker_x_str[i] if( i%divider == 0 ) else '' for i in range(len(ticker_x_str))]

  y_list = np.array(ticker_df['Delta'].tolist())
  x_list = [x for x in range (0 , len(y_list))]

  # Tune huge values for average delta:
  for i in range(len(y_list)):
    note = ""
    if y_list[i] > MAX_HUGE_VAL:
      note += "Avg delta " + str(round(y_list[i],2)) + " -> " + str(MAX_HUGE_VAL) + ". "
      y_list[i] = (y_list[i]/max(y_list))*MAX_HUGE_VAL  
      LOGS(TRACE, note, y_list[i])
  #LOGS(TRACE, "ticker_x:", ticker_x, "\nticker_x_str: ", ticker_x_str, "\nY: ", y_list, "\nX new:", new_ticker_x_str, "Divider:", divider)
  LOGS(TRACE, "draw_new_values_ext: Len=", len(x_list))
  line1.set_xdata(x_list)
  line1.set_ydata(y_list)

  LOGS(TRACE, "draw_new_values_ext: Delta_prev")
  y2_list = np.array(ticker_df['Delta_prev'].tolist())
  #max_y2 = max(y2_list)
  #max_y  = max(y_list) #tune for Max Avg - doesn't work since there re huge values in 
  #for j in range(len(y2_list)):
  #  if y2_list[j] > max_y:
  #    LOGS(TRACE, y2_list[j], max_y2)
  #    tmp = float(y2_list[j])/float(max_y2)* max_y
  #    tmp = round(tmp, 2)
  #    note += "Avg delta " + str(round(y2_list[j],2)) + " -> " + str(tmp) + "."
  #    y2_list[j] = tmp
  #    LOGS(TRACE, note)
  line2.set_xdata(x_list)
  line2.set_ydata(y2_list)

  y3_list = np.array(ticker_df['Close'].tolist())
  #line3.set_xdata(x_list)
  #line3.set_ydata(y3_list)

  ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:green'
  color_set = ['tab:green' ]
  ax2.set_ylabel('Ticker close price', color=color)  # we already handled the x-label with ax1
  ax2.plot(x_list, y3_list, color=color_set[0], label="Close price")
  ax2.legend(loc='upper right')

  xticks = []
  new_ticker_x_str = []
  for i in range(len(ticker_x_str)):
    if( i%divider == 0 ):
      xticks.append(i)
      new_ticker_x_str.append(ticker_x_str[i])
  LOGS(TRACE, "Ticks:", xticks, new_ticker_x_str)

  ax.set_xticks(xticks, labels=new_ticker_x_str)
  ax.legend(loc='upper left')
  ax.tick_params(axis='y', labelcolor='k')
  ax.grid(True)

  ind, value = get_non_nan_index(y3_list)   
  title_str = ticker + " Price:" + str(y3_list[ind]) + " at " + str(ticker_x[ind]) + ". " + note
  ax.set_title(title_str)
  
  # Rescale axes limits
  ax.relim()
  ax.autoscale()

  plt.pause(0.0001)
  figure.canvas.draw()
  figure.canvas.flush_events()
  
def get_metrics(df):
  LOGS(TRACE, "get_metrics(): df.Close", df.Close)
  diff = -1
  if df.Close[0] != 0:
    diff = df.Close[len(df) - 1]/df.Close[0]

  max_delta = -1
  if min(df) != 0:
    max_delta = max(df.Close)/min(df.Close)
  return round(diff, 2), round(max_delta, 2)

def get_metrics_dsc(df):
  rc = get_metrics(df)
  out_rc = "_D" + str(rc[0]).ljust(4 ,'0') + "_M" + str(rc[1]).ljust(4 ,'0') + "_"
  LOGS(TRACE, "get_metrics_dsc() returns: ", out_rc)
  return out_rc
 
  
def get_full_stat(df, last=0):
  cl_up_avg_up   = 0
  cl_up_avg_down = 0
  cl_up_pre_up   = 0
  cl_up_pre_down = 0

  cl_down_avg_up   = 0
  cl_down_avg_down = 0
  cl_down_pre_up   = 0
  cl_down_pre_down = 0
  
  close      = np.array(df['Close'].tolist())
  delta_avg  = np.array(df['Delta'].tolist())
  delta_pre = np.array(df['Delta_prev'].tolist())  
  for i in range(len(close)):   
    if i == 0:
      continue
    if close[i] > close[i-1] and delta_avg[i] > delta_avg[i-1]:
      cl_up_avg_up += 1     
     
    if close[i] > close[i-1] and delta_avg[i] < delta_avg[i-1]:
      cl_up_avg_down += 1     
    
    if close[i] > close[i-1] and delta_pre[i] > delta_pre[i-1]:
      cl_up_pre_up += 1     

    if close[i] > close[i-1] and delta_pre[i] < delta_pre[i-1]:
      cl_up_pre_down += 1     
      
    if close[i] < close[i-1] and delta_avg[i] > delta_avg[i-1]:
      cl_down_avg_up += 1     
     
    if close[i] < close[i-1] and delta_avg[i] < delta_avg[i-1]:
      cl_down_avg_down += 1     
    
    if close[i] < close[i-1] and delta_pre[i] > delta_pre[i-1]:
      cl_down_pre_up += 1     

    if close[i] < close[i-1] and delta_pre[i] < delta_pre[i-1]:
      cl_down_pre_down += 1     
  LOGS(TRACE, "FULL stat: ", cl_up_avg_up, cl_up_avg_down, cl_up_pre_up, cl_up_pre_down, cl_down_avg_up, cl_down_avg_down, cl_down_pre_up, cl_down_pre_down )
  str_list = ['cl_up_avg_up', 'cl_up_avg_down', 'cl_up_pre_up', 'cl_up_pre_down', 'cl_down_avg_up', 'cl_down_avg_down', 'cl_down_pre_up', 'cl_down_pre_down']
  val_list = [cl_up_avg_up, cl_up_avg_down, cl_up_pre_up, cl_up_pre_down, cl_down_avg_up, cl_down_avg_down, cl_down_pre_up, cl_down_pre_down]
  full_stat = "Stat:" + str(str_list) + ". " + str(val_list) + "\n"

  cl_avg_str = cl_up_avg_up + cl_down_avg_down
  cl_avg_inv = cl_up_avg_down + cl_down_avg_up
  cl_pre_str = cl_up_pre_up + cl_down_pre_down
  cl_pre_inv = cl_up_avg_down + cl_down_avg_up

  str_list2 = ['cl_avg_str', 'cl_avg_inv', 'cl_pre_str', 'cl_pre_inv']
  val_list2 = [cl_avg_str, cl_avg_inv, cl_pre_str, cl_pre_inv]
  full_stat = full_stat + str(str_list2) + "\n" + str(val_list2)

  return val_list, full_stat  
       