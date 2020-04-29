import time, requests, json
from yahoo_fin.stock_info import *

def pull_inv(appid, Res="15", Symbol=172, Start="2020-01-01", End="2020-04-28"):
    """
    Resolution : 1(min), 5(min), 15(min), 30(min), 60(min), D(ay), W(eek), M(onth)
    Symbol : 172 (DAX), 8873(US30), 25685 (OMXS30), NASDAQ%20%3AAMD (AMD)
    Start : start date
    End : end data
    """
    Start = time.mktime(time.strptime(Start, "%Y-%m-%d"))
    End = time.mktime(time.strptime(End, "%Y-%m-%d"))

    """
    Dax data, 2020/01/01/08:00 - 2020/04/28/22:00, interval = 15 min
    """
    DAX = "https://tvc4.forexpros.com/{4}/1588169735/13/13/16/history?symbol={0}&resolution={1}&from={2}&to={3}".format(Symbol, Res, Start, End, appid)
    
    req = requests.get(url=DAX)
    if (req.status_code == 200):
        json.dumps(req.json(), "./Data/DAX/[{0}]{1}-{2}.json".format(Res, Start, End))
        # with open("./Data/DAX/[15]2020_01_01-2020_04_28.json", "w") as f:
        #     f.write(req)
    else:
        raise Exception("Access denied")


def pull_yahoo(ticker, start, end, interval,INDEX="DAX"):
    data = get_date(ticker, start_date=start, end_date=end,interval=interval)
    data.to_csv(r"./Data/{}/{}{}-{}.csv".format(INDEX, interval, start, end))
    # print(get_analysts_info("^GDAXI"))
    # dax_history_daily = get_data("^GDAXI", start_date='01/01/1999', end_date='28/04/2020', interval="1d")
    # dax_history_daily.to_csv(r"./Data/DAX/[Daily]1999_01_01-2020_04_28.csv")
    # dax_history_daily = get_data("^GDAXI", start_date='01/01/1999', end_date='28/04/2020', interval="1wk")
    # dax_history_daily.to_csv(r"./Data/DAX/[Weekly]1999_01_01-2020_04_28.csv")
    # dax_history_daily = get_data("^GDAXI", start_date='01/01/1999', end_date='28/04/2020', interval="1mo")
    # dax_history_daily.to_csv(r"./Data/DAX/[Monthly]1999_01_01-2020_04_28.csv")
