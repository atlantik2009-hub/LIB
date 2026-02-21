import sys
from datetime import datetime

TRACE = 1
PROD  = 2

SYS_DEBUG_LEVEL = TRACE;

def get_debug_lvl_dsc(debug_lvl):
  if debug_lvl == TRACE:
    dsc = "TRACE"
  elif debug_lvl == PROD:
    dsc = "PROD"
  return dsc

def setup_debug_level(debug_lvl):
  global SYS_DEBUG_LEVEL
  SYS_DEBUG_LEVEL = debug_lvl
  LOGS(PROD, "DEDUG LEVEL:", get_debug_lvl_dsc(debug_lvl))

def LOGS(*arg):
  if int(arg[0]) < SYS_DEBUG_LEVEL:
    return
  level_name = "PROD" if arg[0] == PROD else "TRACE"

  # arg[1] stores the DEBUG level 
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  message = ' '.join(str(f) for f in arg[1:])
  formatted = f"{timestamp} [{level_name}] {message}"  
  print (formatted)
#    print (' '.join(str(f) for f in arg[1:]))
