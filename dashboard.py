import os
from datetime import datetime

os.makedirs("dashboard", exist_ok=True)
LOG_FILE = f"dashboard/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

def dash(message):  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} [{level_name}] {message}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
