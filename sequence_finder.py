import pandas as pd
import math

from intra_common_lib import *


def find_sequences(df, N=10, H=5, X=-1, Y=9.9):
    """
    Function to find sequences in N time intervals where Delta and Delta_prev decrease from 1st to 10th element
    with upper deviations of H% and then sharply increase by Y% on the N-th element.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with Delta, Delta_prev columns and Time index in "10:10:00" format.
    - N (int): Number of time intervals to analyze.
    - H (float): Allowed upper deviation in percentage.
    - X (float): Minimum decrease from 1st to 10th element in percentage.
    - Y (float): Minimum increase on the N-th element in percentage.
    
    Returns:
    - tuple: Two lists with time indices for Delta and Delta_prev.
    """
    delta_avg_list = []
    delta_prev_list = []
    
    for clmn in ['Delta', 'Delta_prev']:
        # Iterate over all possible sequences of length N
        for i in range(len(df) - N + 1):
            # Get the current sequence
            seq = df.iloc[i:i + N]
        
            
            # Check Delta
            delta_values = seq[clmn].values
            bad_seria = True
            for val in delta_values:
                if math.isnan(val):
                    bad_seria = False
                    break
       
            if bad_seria == False:
                #print("Skipped NaN value: ", delta_values)
                continue     

            close_values = seq['Close'].values
       
            if check_sequence(delta_values, H, X, Y):
                price_delta = get_price_stat(close_values, seq.index[-1], df) 

                LOGS(PROD, delta_values, "added", clmn, "Price:", price_delta)
                if clmn == 'Delta':
                    delta_avg_list.append(seq.index[-1])
                elif clmn == 'Delta_prev':
                    delta_prev_list.append(seq.index[-1])
                
    LOGS(PROD, "Delta_avg_list:", ", ".join([dt.strftime("%H:%M") for dt in delta_avg_list]))
    LOGS(PROD, "Delta_prev_list:", ", ".join([dt.strftime("%H:%M") for dt in delta_prev_list]))

    return delta_avg_list, delta_prev_list

def get_price_stat(values, index, df):
    SERIA_AFTER = 5 # entry price included
    target_id = len(values) - 1
    close_change = round(100*(values[target_id] - values[target_id-1])/values[target_id-1], 2)

    new_index = df.index.get_loc(index) + SERIA_AFTER

    close_5m = -1
    if new_index < len(df):
        close_5m = df.iloc[new_index]['Close']         
        close_5m_change = round(100*(close_5m - values[target_id])/values[target_id], 2)

        list_5m = [] 
        for i in range(SERIA_AFTER + 1):
            idx = df.index.get_loc(index) + i   
            list_5m.append(df.iloc[idx]['Close'])
        print( index.strftime("%H:%M"), "df.loc[index, 'Close']=", df.loc[index, 'Close'], "values[target_id]=", values[target_id], "close_5m=", close_5m, list_5m ) 

    return close_change, close_5m_change

def check_sequence(values, H, X, Y):
    """
    Check if the sequence meets the conditions.
    
    Parameters:
    - values (list): List of values to check.
    - H (float): Allowed upper deviation in percentage.
    - X (float): Minimum decrease from 1st to 10th element in percentage.
    - Y (float): Minimum increase on the N-th element in percentage.
    
    Returns:
    - bool: True if the sequence meets the conditions, False otherwise.
    """
    target_id = len(values) - 1
    diff = -1
    up_move  = -1
    up_move2 = -1
    up_move3 = -1
    if values[0] != 0:
        diff = values[target_id - 1]/values[0]
    if values[target_id - 1] != 0:
        up_move = values[target_id] / values[target_id - 1]
    if values[target_id - 2] != 0:
        up_move2 = values[target_id] / values[target_id - 2]
    if values[target_id - 3] != 0:
        up_move3 = values[target_id] / values[target_id - 3]

    #print(values, diff, up_move)

    for j in range(len(values) - 2):
        # Check decrease from 1st to 10th element             
        if values[j + 1] > values[j] * (1 + H/100):
            #print("1st condition", j, values[j], values[j - 1], (values[j - 1] * (1 + H/100)))
            return False


    # Check overall decrease from 1st to 10th element
    # Skip the check for X=-1
    if X > 0 and values[target_id - 1] > values[0] * (1 - X/100):
        #print("2nd condition")
        return False

    # Check increase on the N-th element   
    flag = -1
    delta = -1
    if up_move > ( 1 + Y/100):
        flag = 1
        delta = round((up_move*100 - 100), 1)
    elif up_move2 > ( 1 + Y/100):
        flag = 2
        delta = round((up_move2*100 - 100), 1)
    elif up_move3 > ( 1 + Y/100):
        flag = 3
        delta = round((up_move3*100 - 100), 1)
    else: 
        #print("3rd condition", up_move)
        return False

    #Check if Delta >=1 less than in previous day
    if values[target_id] < 1:
        return False
     
    LOGS(PROD, values[0], "->", values[target_id - 1], "->", values[target_id], "Delta Up", flag, "=", round((up_move*100 - 100), 1) )
    return True

