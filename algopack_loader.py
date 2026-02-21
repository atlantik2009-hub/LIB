import requests
import pandas as pd
from datetime import datetime

from intra_common_lib import *

futures_list = ["IMOEXF", "USDRUBF", "CNYRUBF", "VBH5", "TNH6"]
indexes_list = ["IMOEX"]

def is_time_more_or_equal(time_str1, time_str2):
    """
    Compares two times represented as strings.
    
    Parameters:
    - time_str1 (str): First time in "HH:MM:SS" format.
    - time_str2 (str): Second time in "HH:MM:SS" format.
    
    Returns:
    - bool: True if time_str1 is earlier than time_str2, False otherwise.
    """
    
    # Convert strings to datetime objects
    time1 = datetime.strptime(time_str1, "%H:%M:%S")
    time2 = datetime.strptime(time_str2, "%H:%M:%S")

    # Compare times
    return time1 >= time2

# INTRA_MODE
# 1  - 1 minute
# 10 - 10 minutes
# 60 - 1 hour
# 24 - 1 day
def load_df_moex(ticker, start_date, end_date, INTRA_MODE):
    LOGS( TRACE, "load_df_moex:", ticker, start_date, end_date, INTRA_MODE )

    """
    Load data from MOEX API for a given ticker and date range.
    
    Parameters:
    - ticker (str): The stock ticker symbol.
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.
    - INTRA_MODE (int): The interval for data retrieval (1, 10, 60, 24).
    
    Returns:
    - pd.DataFrame: A DataFrame containing the retrieved data.
    """
    
    # Check if start_date and end_date are the same day
    if start_date[:10] != end_date[:10] and (INTRA_MODE == 1 or INTRA_MODE == 10):
        raise ValueError("Start and end dates must be the same for INTRA_MODE 1 and 10.")
    

    # Initialize an empty DataFrame
    final_df = pd.DataFrame()

    payload = {}
    headers = {
      'Accept': 'application/json',
      'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3NzM1MDQwNDcsImlhdCI6MTc3MDkxMjA0NywianRpIjoiY2FlZGQ5NmMtN2MxZC00ZWM4LWJlYjItZTU4OTVhMjA1ZGUzIiwiaXNzIjoiaHR0cHM6Ly9zc28yLm1vZXguY29tL2F1dGgvcmVhbG1zL2NyYW1sIiwiYXVkIjpbImFjY291bnQiLCJpc3MiXSwic3ViIjoiZjowYmE2YThmMC1jMzhhLTQ5ZDYtYmEwZS04NTZmMWZlNGJmN2U6M2M5NWExYzktMTQ0YS00Zjc2LThkOWEtMDBlZWIzNjFkZTQwIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiaXNzIiwic2lkIjoiNzJhNjlhNDEtMjNhYy00N2FlLWJjMjgtOWU0OGUzNjVkNmNmIiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBpc3NfYWxnb3BhY2sgcHJvZmlsZSBvZmZsaW5lX2FjY2VzcyBlbWFpbCBiYWNrd2FyZHNfY29tcGF0aWJsZSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiM2M5NWExYzktMTQ0YS00Zjc2LThkOWEtMDBlZWIzNjFkZTQwIiwic2Vzc2lvbl9zdGF0ZSI6IjcyYTY5YTQxLTIzYWMtNDdhZS1iYzI4LTllNDhlMzY1ZDZjZiJ9.a8elDn-sk_g9RGj2nAqvztsiG4eVLoJrXO3rUfhiK6eOW7eRDCXFYWTj7fAueRaf6EEUvO3X_QCaxbqKB3yDxCzwv4obkLEXbl24sxRWQdg3FsX0-t4-GGqJMM3OGCoBpYxy3fP7JwxuYb4i8rYwZWBhqQSeqjEFjAAti8nynynOE5aF2jpmOve_R9-31LqBtXBYpRiG-lkhWj-tBxBVWmrNRvFuwr-KELjio_z4waOIdU0SlfjzB4osKkvut-ibTxFUEmPwbJkP-vIhFPpkUUUI1iO57ie_Xu1NyetEFRjmOk4w56YnFtLXjMi2QG-KE7hs-UpP29t6k43BzuJmhg'
    }
    
    # Handle different INTRA_MODE values
    if INTRA_MODE == 1:
        start_time = ["06:00:00", "15:01:00", "23:31:00"]
        end_time =   ["15:00:00", "23:30:00", "23:59:59"]

        if len(end_date) == 19:
            end_time_in = end_date[11:]
        else:      
            end_time_in = "23:59:59"

        # Split the day into a few parts: morning and afternoon
        for i in range(len(start_time)):               
            if is_time_more_or_equal(end_time[i], end_time_in) == True:
               end_time_out = end_time_in
            else:
               end_time_out = end_time[i]
            if ticker in futures_list:
                url = f"https://apim.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/{ticker}/candles.json?from={start_date}%20{start_time[i]}&till={end_date[:10]}%20{end_time_out}&interval={INTRA_MODE}"
            elif ticker in indexes_list:
                url = f"https://apim.moex.com/iss/engines/stock/markets/index/boards/sndx/securities/{ticker}/candles.json?from={start_date}%20{start_time[i]}&till={end_date[:10]}%20{end_time_out}&interval={INTRA_MODE}"
            else:
                url = f"https://apim.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/candles.json?from={start_date}%20{start_time[i]}&till={end_date[:10]}%20{end_time_out}&interval={INTRA_MODE}"
        
            LOGS( TRACE, "URL:", url)       
            # Fetch data for the morning session
            response = requests.get(url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                df_morning = pd.DataFrame(data['candles']['data'], columns=data['candles']['columns'])
                final_df = pd.concat([final_df, df_morning], ignore_index=True)
            else:
                LOGS(PROD, f"Error: {response.status_code}")

            if is_time_more_or_equal(end_time[i], end_time_in) == True:
               break 

            
    else:
        # For other intervals (10, 60, 24), fetch data for the entire day
        url = f"https://apim.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/candles.json?from={start_date}&till={end_date}&interval={INTRA_MODE}"
        response = requests.get(url, url, headers=headers, data=payload)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['candles']['data'], columns=data['candles']['columns'])
            final_df = pd.concat([final_df, df], ignore_index=True)
        else:
            LOGS(PROD,f"Error: {response.status_code}")

    #print(final_df)     
    return final_df
