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

INPUT_FILE = "C:\\MAIN\\WORK_DIR\\MOEX_PACK\\AGGREGATE_DIR_FINAL\\final_stat_file_20250627.csv"
#INPUT_FILE = "C:\\MAIN\\WORK_DIR\\MOEX_PACK\\AGGREGATE_DIR_FINAL\\final_stat_file.csv"
FINAL_STAT_COLUMNS = ['Ticker', 'Model', 'Period',    'Pred_s', 'Pred_e', 'Vol_pre_s', 'Vol_pre_e', 'Vol_avg_s', 'Vol_avg_e', "Mode",    "cl_counter", "hl_counter", "lo_counter", "nl_counter", "all_cases", "cl_lo", "clhl_lo", "not_looses_lo", "delta", 'Result']

def select_stat_entries(df, model, pred, vol_pre, vol_avg, desc="", ticker=None):
  #print("select_stat_entries: ", ticker, model, "Pred:", pred, "Vol_pre:", vol_pre,"Vol_avg:", vol_avg)

  out_df = df
  if ticker is not None:
    out_df = out_df[(out_df['Model'] == model) & (out_df['Ticker'] == ticker) ]
  else:     
    out_df = out_df[(out_df['Model'] == model)]

  # For debugging
  #if(len(out_df) > 0):
  #  print(out_df) 
  for id, row_id in out_df.iterrows():
    pred_s = out_df.at[id, 'Pred_s']
    pred_e = out_df.at[id, 'Pred_e']
    vol_pre_s = out_df.at[id, 'Vol_pre_s']
    vol_pre_e = out_df.at[id, 'Vol_pre_e']
    vol_avg_s = out_df.at[id, 'Vol_avg_s']
    vol_avg_e = out_df.at[id, 'Vol_avg_e']
    period = out_df.at[id, 'Period']

    not_looses_lo = out_df.at[id, 'not_looses_lo']
    result = out_df.at[id, 'Result']
    all_cases = out_df.at[id, 'all_cases'] 
    rc = True
    if pred < pred_s or pred > pred_e:
      #print("failed by pred", pred_s, pred_e)
      rc = False

    if vol_pre_s > -1 and (vol_pre < vol_pre_s or vol_pre > vol_pre_e):
      rc = False
      #print("failed by vol_pre", vol_pre_s, vol_pre_e)

    if vol_avg_s > -1 and (vol_avg < vol_avg_s or vol_avg > vol_avg_e):
      rc = False
      #print("failed by vol_avg", vol_avg_s, vol_avg_e)

    # 62/38 = 1.63; 60/40 = 1.5; 65/35 = 1.857; 70/30 = 2.33
    if not_looses_lo < 1.857 and not_looses_lo != 0 and ticker != "IMOEX":
      rc = False     
    
    if rc == False:
      #  out_df.drop(id, inplace = True) 
      continue

    #print(out_df)
    #print("select_stat_entries: ", ticker, model, "Pred:", pred, "Vol_pre:", vol_pre,"Vol_avg:", vol_avg)
    print("\nSuccess: ", ticker, period, model, "Pred:", pred_s, "-", pred_e, "Vol_pre:", vol_pre_s, "-", vol_pre_e, "Vol_avg:", vol_avg_s, "-", vol_avg_e, "not_looses_lo=", not_looses_lo, "Result=", result, "all_cases=", all_cases)
    if len(desc) > 0:
      print(desc)

  return out_df 

def hadle_stat_data_for_ticker_df(df):
  stat_df = pd.read_csv(INPUT_FILE, names=FINAL_STAT_COLUMNS, skiprows = 0, index_col=False)

  print("##### hadle_stat_data_for_ticker_df: Test Len=", len(df), "Stat df Len=", len(stat_df), "#####") 
  for id, row_id in df.iterrows():
    pred = df.at[id, 'Pred']
    vol_pre = df.at[id, 'Vol_pre']
    vol_avg = df.at[id, 'Vol_avg']
    ticker  = df.at[id, 'Ticker']
    model   = df.at[id, 'Model']

    #desc = "          " + ",".join(df.loc[id].apply(str).values)
    desc = str(row_id.to_frame().T)
    #desc = df.iloc[id].to_string(header=False, index=False)

    select_stat_entries(stat_df, model, pred, vol_pre, vol_avg, desc, ticker)
