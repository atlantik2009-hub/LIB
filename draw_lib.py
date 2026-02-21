import sys, os, time, tempfile, random, shutil
import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta

from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import matplotlib.pyplot as plt

#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *
from loader_lib import *
from intra_test_lib import *

OUT_DIR = "PDF_DIR"

def draw_ds_in_dir(DATASET_DIR, OUTPUT_PDF_FILE):
  counter = 0
  entries = os.listdir(DATASET_DIR)
  pdf = PdfPages(OUTPUT_PDF_FILE)

  for entry in entries:
    if os.path.isdir(entry) or entry.find(".csv") == -1:
      print(entry, "skipped")
      continue

    ticker_file = DATASET_DIR + '/'+ entry
    print(counter, "!!!", ticker_file, "!!!")
    ticker_data = pd.read_csv( ticker_file, names=CSV_COLUMNS, index_col = INDEX_COL, skiprows = 1)
    draw_chart_by_df(ticker_data, ticker_file, pdf)
    counter+=1

  pdf.close()

def generate_pdf_file(ticker_df, title_str, OUTPUT_PDF_FILE):
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  pdf_file = OUT_DIR + '/'+ OUTPUT_PDF_FILE

  pdf = PdfPages(pdf_file)
  draw_chart_by_df(ticker_df, title_str, pdf)
  pdf.close()

def draw_chart_by_df(ticker_df, title_str, pdf):  
  ZERO_TIME = "00:00:00"
  ticker_x = np.array(ticker_df.index.tolist())
  # Check if it is day info and we can cut off 00:00:00 time
  if(len(str(ticker_x[0])) > 10):
    print( str(ticker_x[0]), "to be tuned, Length=", len(str(ticker_x[0])) )
    length_x = len(ticker_x)
    if str(ticker_x[0])[11:] == str(ticker_x[length_x-1])[11:] and str(ticker_x[0])[11:] == ZERO_TIME:
      print(ZERO_TIME, "is cut off")
      for i in range(length_x):
        ticker_x[i] = str(ticker_x[i])[0:10]

  ticker_close = np.array(ticker_df['Close'].tolist())
  ticker_hi = np.array(ticker_df['High'].tolist())
  ticker_lo = np.array(ticker_df['Low'].tolist())
  ticker_v = np.array(ticker_df['Volume'].tolist())

  print(title_str)
  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.95, title_str, transform=fig.transFigure, size=9)    
  plt.title('Ticker Close, High, Low')
  plt.plot(ticker_x, ticker_close, color='b', label = 'Close')
  plt.plot(ticker_x, ticker_hi, color='g', label = 'High')
  plt.plot(ticker_x, ticker_lo, color='r', label = 'Low')
  plt.legend()
  pdf.savefig()
  plt.close(fig)

  fig = plt.figure(figsize=(12, 8))
  plt.text(0.01, 0.95, title_str, transform=fig.transFigure, size=9)    
  plt.title('Ticker Volume')
  #plt.bar(ticker_x, ticker_v, color='b', label = 'Volume')
  plt.plot(ticker_x, ticker_v, color='b', label = 'Volume')
  plt.legend()
  pdf.savefig()
  plt.close(fig)


def draw_ticker_base(ticker, start_date, end_date, INTRA_MODE, title, stock, pdf, show_delta=False):
  ticker_data_df = load_df_ext(ticker, start_date, end_date, INTRA_MODE, stock)
  if ticker_data_df is None:
    return

  print(ticker_data_df.tail(10).to_string())
  start_tm = ticker_data_df.index[0] 
  end_tm = ticker_data_df.index[len(ticker_data_df)-1]

  delta_str = ""
  if show_delta == True:
    ticker_close = np.array(ticker_data_df['Close'].tolist())
    if ticker_close[0] > 0:
      delta = round(ticker_close[len(ticker_close) - 1]/ticker_close[0], 3) 
      delta_str = " Delta: "  + str(delta)

  title_str = title + "\n" + ticker + " " + str(INTRA_MODE) + "m, " + str(start_tm) + "-" + str(end_tm) + delta_str
  draw_chart_by_df(ticker_data_df, title_str, pdf)

def get_stat_file_by_model_name(model_name):
  DIR_IN = "MODELS"
  rc = ""
  files = os.listdir(DIR_IN)
  files.sort()
  # Remove .zip and dir prefix
  id = model_name.find("\\")
  if id == -1:
   id = 0 
  model_name = model_name[id+1:-4]
  for f in files:
    stat_file = DIR_IN + "/" + f
    #print(stat_file)
    if( stat_file.find(model_name) != -1 and 
        stat_file.find("ST_")      != -1 and 
        stat_file.find(".csv")     != -1 ):
      rc = stat_file
      break
  print("get_stat_file_by_model_name() returns [", rc, "]")
  return rc 


def draw_pred_range(ticker, model_name, start_pred, pred, pdf):
  rc = -1
  stat_file = get_stat_file_by_model_name(model_name)
  if len(stat_file) == 0:
    return -1

  stat_data = pd.read_csv( stat_file, names=TEST_STAT_COLUMNS, header=None, skiprows=[0])
  
  # Tune ticker name properly
  if len(stat_data) == 0:
    print("stat_data empty")
    return rc
  prefix = stat_data['File'].at[0]
  id = prefix.find("/")
  ticker_in = prefix[0:id+1] + ticker + ".csv" 
  print("Checking: ", ticker_in)
  ticker_df = stat_data[stat_data['File'] == ticker_in]
  print(ticker_df, "\nLen:", len(ticker_df))
  if len(ticker_df) == 0:
    print("ticker_df empty")
    return rc
   
  ret_code = process_stat_df(ticker_df, pdf, False, start_pred, "", pred)
  #ret_code[3] stores variable CL/LO corresponding to Pred=pred
  return ret_code[3]


def draw_ticker_charts(ticker, stock, title, start_date, OUTPUT_PDF_FILE, model_name, start_pred, target_pred, end_date="", DIR_IN=OUT_DIR):
  MODEL = get_MODEL_by_model_name(model_name)
  print("draw_ticker_charts():", MODEL, start_date, "-", end_date)

  if stock == NY_STOCK:
    index = SP500_INDEX
    if MODEL.time_i == "1d":
      INTRA_MODE1 = '1d'
      print("draw_ticker_charts(): INTRA_MODE=", INTRA_MODE1) 
    else:
      INTRA_MODE1 = '1m'
      INTRA_MODE10 = '5m'
      end_date = str(date.today() + timedelta(days=1))
  else:
    index = MOEX_INDEX
    if MODEL.time_i == "1d" or MODEL.time_i == "24":
      INTRA_MODE1 = '24'
      print("draw_ticker_charts(): INTRA_MODE=", INTRA_MODE1)
    else:
      INTRA_MODE1 = 1
      INTRA_MODE10 = 10
      end_date = start_date
  print("draw_ticker_charts() after tune:", start_date, "-", end_date)

  # For interval=1d put all data into separate dir NY_PDF_DIR or MOEX_PDF_DIR
  if MODEL.time_i == "1d" or MODEL.time_i == "24":
    DIR_IN = stock + "_" + str(end_date) + "_" + DIR_IN

  if not os.path.exists(DIR_IN):
    os.makedirs(DIR_IN)
  pdf_file = DIR_IN + '/'+ OUTPUT_PDF_FILE
  pdf = PdfPages(pdf_file)

  if MODEL.time_i == "1d":
    # Show 1d ticker info
    draw_ticker_base(ticker, start_date, end_date, INTRA_MODE1, title, stock, pdf) 

    # Show 1d Index info
    draw_ticker_base(index, start_date, end_date, INTRA_MODE1, "", stock, pdf) 

  else:
    # Show 1m ticker info
    draw_ticker_base(ticker, start_date, end_date, INTRA_MODE1, title, stock, pdf) 
    
    # Show 10m ticker info
    draw_ticker_base(ticker, start_date, end_date, INTRA_MODE10, "", stock, pdf)

    # Show 10m Index info
    draw_ticker_base(index, start_date, end_date, INTRA_MODE10, "", stock, pdf) 
  
  #Show prediction statistics
  cl_lo = draw_pred_range(ticker, model_name, start_pred, target_pred, pdf)
  postfix = "_CO" + str(cl_lo)
  pdf.close()

  pdf_file_new = pdf_file[0:-4] + postfix + ".pdf"
  if os.path.exists(pdf_file_new):
    print("File already exists", pdf_file_new)
    if os.path.exists(pdf_file):
      print("Do noting")    
      #shutil.rmtree(pdf_file)
  else:   
    os.rename(pdf_file, pdf_file_new)
    print(pdf_file, "is renamed into", pdf_file_new)
                    