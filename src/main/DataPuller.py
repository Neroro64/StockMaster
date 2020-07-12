import time, requests, json
from yahoo_fin.stock_info import *
   
def pull_dax(start, end, interval):
    data = get_data("^GDAXI", start_date=start, end_date=end, interval=interval)
    return data

def pull_dax_current():
    data = get_live_price("^GDAXI")
    return data

