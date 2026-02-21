import sys
import shutil
import os

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import date
from datetime import datetime, timedelta

import requests
import apimoex

from intra_lib import *
from intra_common_lib import *
from algopack_loader import *


SP500_INDEX = "^GSPC"
MOEX_INDEX = "IMOEX"
CRYPT_INDEX = "BTC-USD"
INDEX_FILE = 'INDEX.csv'
FUTURES_LIST = ['CRH6', 'IMOEXF', 'CRZ4', 'USDRUBF', 'CNYRUBF']

# INTRA_MODE
# 1  - 1 minute
# 10 - 10 minutes
# 60 - 1 hour
# 24 - 1 day
# 7  - 1 week
# 31 - 1 month
# 4  - 3 monthes
def load_df(ticker, start_date, end_date, INTRA_MODE, stock):
  LOGS(TRACE, "load_df:", ticker, start_date, end_date, INTRA_MODE, stock)
  if stock == MOEX_STOCK and INTRA_MODE == '1d':
    INTRA_MODE = '24'
    LOGS(TRACE,"INTRA_MODE 1d substituted by", INTRA_MODE, "for", stock)

  CHECK_ENABLE = False
  if stock == NY_STOCK:
    try:
      if len(INTRA_MODE) > 0:
        ticker_data_df = yf.download(ticker, start=start_date, end=end_date, interval=INTRA_MODE, progress=False, show_errors=False, threads=False, timeout=1)  
      else: 
        ticker_data_df = yf.download(ticker, start=start_date, end=end_date, progress=True, show_errors=True, threads=True)
      if len(ticker_data_df) == 0:
        return None;
    except ValueError: 
      print("Exception catched on ticker", ticker)
      return None;
    except Exception as ex:
      print("Exception catched on ticker", ticker, ex)
      return None;
  else:
    i = 1
    while(1):
      try:

        # Let's use ALGOPACK for INTRA_MODE = 1 and 10  
        if (INTRA_MODE == 1 or INTRA_MODE == 10):
          ticker_data_df = load_df_moex(ticker, start_date, end_date, INTRA_MODE)
          break

        with requests.Session() as session:
          if ticker == MOEX_INDEX:
            data = apimoex.get_market_candles(session, ticker, INTRA_MODE, start_date, end_date, ('begin', 'open', 'high', 'low', 'close', 'value'), 'index', 'stock')
          elif ticker == 'USD000UTSTOM' or ticker == 'CNYRUB':
            data = apimoex.get_market_candles(session, ticker, INTRA_MODE, start_date, end_date, ('begin', 'open', 'high', 'low', 'close', 'value'), market = "selt", engine = "currency")
          elif ticker in FUTURES_LIST:
            data = apimoex.get_board_candles(session, ticker, interval=INTRA_MODE, start=start_date, end=end_date, columns=('begin', 'open', 'high', 'low', 'close', 'volume'), market = "forts", engine = "futures") # Brent
          else:
            data = apimoex.get_board_candles(session, ticker, INTRA_MODE, start_date, end_date, ('begin', 'open', 'high', 'low', 'close', 'value')) # Statistics by intervals 10 minutes
          ticker_data_df = pd.DataFrame(data)
      except ValueError: 
        print("Exception 1 catched on ticker", ticker)
        return None;
      except Exception as ex:
        print("Exception 2 catched on ticker", ticker, ex, " Try:", i)
        i+=1
        if i > 4: 
          return None;
        else:
          time.sleep(2)
          continue;

      break;

  if len(ticker_data_df) == 0:
    print("No data loaded for ticker", ticker)
    return None;
   
  # Tune output date in order to handle it in general way
  if stock == NY_STOCK:
    # Drop column 'Adj Close'
    ticker_data_df = ticker_data_df.drop('Adj Close', axis=1)
  else:
    # Rename columns like at NY stock: Datetime,Open,High,Low,Close,Volume
    ticker_data_df = ticker_data_df.rename(columns={'begin':'Datetime'})
    ticker_data_df = ticker_data_df.rename(columns={'open':'Open'})
    ticker_data_df = ticker_data_df.rename(columns={'high':'High'})
    ticker_data_df = ticker_data_df.rename(columns={'low':'Low'})
    ticker_data_df = ticker_data_df.rename(columns={'close':'Close'})
    if ticker in FUTURES_LIST:
      ticker_data_df = ticker_data_df.rename(columns={'volume':'Volume'})
    else: 
      ticker_data_df = ticker_data_df.rename(columns={'value':'Volume'})
   
    ticker_data_df = ticker_data_df.set_index('Datetime')
  if CHECK_ENABLE == True:
    bad_entry_counter = 0
    for id, row_id in ticker_data_df.iterrows():
      if pd.isna(ticker_data_df.at[id,'Open']):
        bad_entry_counter = bad_entry_counter + 1
      ticker_data_df = ticker_data_df.drop(labels=id, axis=0)
    print("Bad entries", bad_entry_counter)

  #print("After\n", ticker_data_df.to_string())
  return ticker_data_df

def save_df(HISTORY_DATA_DIR, ticker, ticker_data_df, stock):
  # Save both real name and index name
  if ticker == MOEX_INDEX or ticker == SP500_INDEX or ticker == CRYPT_INDEX:
    file_name = HISTORY_DATA_DIR + '/' + INDEX_FILE
    ticker_data_df.to_csv(file_name)

  file_name = HISTORY_DATA_DIR + '/' + ticker + '.csv'
  ticker_data_df.to_csv(file_name) 


def load_history_data(input_file, start_date, end_date, HISTORY_DATA_DIR, zip_flag, test_flag, INTRA_MODE, stock=NY_STOCK):
  LOG_FILE = "loader_logs_" + str(datetime.now().strftime('%Y%m%d_%H%M%S')) + ".txt"
  sys.stdout = open(LOG_FILE, 'w', buffering=1)
  print("START:", datetime.now())
  print(input_file, start_date, end_date, HISTORY_DATA_DIR, zip_flag, test_flag, INTRA_MODE, stock)

  ZIP_FILE = HISTORY_DATA_DIR + ".zip"

  if zip_flag == '1':
    print("Ticker file:", input_file, "; OUT dir:", HISTORY_DATA_DIR, "; OUT zip file:", ZIP_FILE)
  else:
    print("Ticker file:", input_file, "; OUT dir:", HISTORY_DATA_DIR)
  print ("Dates:", start_date, "-", end_date)

  shutil.rmtree(HISTORY_DATA_DIR, ignore_errors=True) 
  os.mkdir(HISTORY_DATA_DIR)
  
  df_tickers = pd.read_csv(
      input_file,
      names=['Symbol'], index_col=False)
  
  print(df_tickers)
  
  print("Number of tickers: ", len(df_tickers))

  if stock == NY_STOCK:
    ticker = SP500_INDEX
  elif stock == CRYPT_STOCK:
    ticker = CRYPT_INDEX
    stock  = NY_STOCK
  elif stock == MOEX_STOCK:
    ticker = MOEX_INDEX
  else:
    print("Unknown stock: ", stock)
    return None
     
  ticker_data_df = load_df(ticker, start_date, end_date, INTRA_MODE, stock)
  print(ticker, ":")
  print(ticker_data_df)

  if ticker_data_df is not None:
    save_df(HISTORY_DATA_DIR, ticker, ticker_data_df, stock)
  
  counter = 0
  for index, row in df_tickers.iterrows():
    ticker = df_tickers.at[index, 'Symbol']
    #if counter % 1 == 0:
    print(counter, ticker)
    ticker_data_df = load_df(ticker, start_date, end_date, INTRA_MODE, stock)

    if ticker_data_df is None:
      continue     
   
    if len(ticker_data_df) < 7:
      print("Incorrect data: Not enough. Ticker:", ticker, "Entries:", len(ticker_data_df))
      continue;

    if ticker_data_df is not None:
      save_df(HISTORY_DATA_DIR, ticker, ticker_data_df, stock)
  
    counter = counter + 1
    # For debug purpose
    if counter == 1:
      print("LEN:", len(ticker_data_df))
      print(ticker_data_df)

  if test_flag == "T":
    out_tickers_file = HISTORY_DATA_DIR + '/' + "tickers.csv"
    shutil.copy2(input_file, out_tickers_file)
    print(out_tickers_file, "created")    

  out_file = HISTORY_DATA_DIR
  if zip_flag == '1':
    shutil.rmtree(ZIP_FILE, ignore_errors=True)
    shutil.make_archive(HISTORY_DATA_DIR, 'zip', ".", HISTORY_DATA_DIR)
    out_file = ZIP_FILE
    shutil.rmtree(HISTORY_DATA_DIR, ignore_errors=True) 
    
  print("FINISHED:", datetime.now())
  return out_file


def load_df_ext(ticker, start_date, end_date, interval, stock):
  DATASET_DIR_MOEX = "DS_MOEX"
  DATASET_DIR_NY   = "DS_NY"
  DATASET_DIR_CR   = "DS_NY"
  ZERO_TIME_POSTFIX = " 00:00:00"

  LOGS(TRACE, "load_df_ext:", ticker, start_date, end_date, interval, stock)
  if interval == "10":
    DATASET_DIR_MOEX = "DS_10M"  
  elif interval != "1d" and interval != "24":
    print("load_df_ext: interval ", interval, "NOT supported")
    return None

  start_d = start_date
  end_d = end_date
  increment = 1
  if stock == MOEX_STOCK:
    if len(start_date) == 10:
      start_d = start_date + ZERO_TIME_POSTFIX
    if len(end_date) == 10:
      end_d = end_date + ZERO_TIME_POSTFIX
    DIR = DATASET_DIR_MOEX
  elif stock == NY_STOCK:
    DIR = DATASET_DIR_NY 
  elif stock == CRYPT_STOCK:
    DIR = DATASET_DIR_CR
  else:
    print("ERROR: udefined stock ", stock)
    return None 
  
  input_file = DIR + "\\" + ticker + ".csv"
  if os.path.isfile(input_file):
    df = pd.read_csv(input_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
    #print(df.to_string())
    if start_d in df.index and end_d in df.index:
      a = df.index.get_loc(start_d)
      b = df.index.get_loc(end_d) + increment
      #print(a, start_d, b, end_d)
      df = df.iloc[a:b]
      #print(df.to_string())
      return df
    else:
      print("No data for", start_d, "and", end_d, "in", input_file)  
  else:
    print(input_file, "doesn't exist. So load from the provider")   

  # For NY stock end date should be more than requested end date
  end_d = end_date 
  if stock == NY_STOCK or stock == CRYPT_STOCK:
    end_d = str(datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))[0:10]

  # Redefine stock for crypto since data are taken from the NY stock
  if stock == CRYPT_STOCK:  
    stock = NY_STOCK

  df = load_df(ticker, start_date, end_d, interval, stock)
  #print(df)
  return df
