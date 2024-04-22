from datetime import datetime, timedelta
from time import sleep
import pandas as pd 
import numpy as np 
import requests
import json
import pandas as pd 
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import sys
import os
from dotenv import load_dotenv

sys.path.append("..")
from utilities.strToDatetime import date_to_time
from utilities.tradingTime import generate_trading_time

#load the environment variable into the file
load_dotenv()

TIMEOUT = 1000

DATA_URL = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{start_time}/{end_time}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

SMA_URL = "https://api.polygon.io/v1/indicators/sma/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=20&series_type=close&order=desc&apiKey={api_key}&limit=5000"

EMA_URL = "https://api.polygon.io/v1/indicators/ema/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=20&series_type=close&order=desc&apiKey={api_key}&limit=5000"

MACD_URL = "https://api.polygon.io/v1/indicators/macd/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&short_window=12&long_window=26&signal_window=9&series_type=close&order=desc&apiKey={api_key}&limit=5000"

RSI_URL = "https://api.polygon.io/v1/indicators/rsi/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=14&series_type=close&order=desc&apiKey={api_key}&limit=5000"

NEWS_URL = """https://api.polygon.io/v2/reference/news?ticker=SPY&published_utc.gte={lower_time}&published_utc.lt={upper_time}&order=asc&limit=1000&sort=published_utc&apiKey={api}"""

DOLLAR_URL = "https://api.polygon.io/v2/aggs/ticker/UUP/range/1/minute/{start_time}/{end_time}?adjusted=true&sort=asc&limit=50000&apiKey={api}"

DICT_MAPPING = {"v":"volume", 'c' : "curr_price", "h" : "high_price", "l" : "low_price", "t" : "utc_timestamp", "n" : "n_transactions"}

class StockScrapeWorker():
    def __init__(self, start_time : str, end_time : str, api_key = os.getenv("api"), 
                 storage_dir : str = "../raw_data"):
        self.start_time = start_time = date_to_time(start_time)
        self.end_time = end_time = date_to_time(end_time)
        self.api = api_key
        self.storage_dir = Path(storage_dir)
    
    def scrape_single_day(self, cur_date : datetime):
        """Scrape the stock in a single day. Return the dataframe or None"""
        start_time, end_time = generate_trading_time(cur_date)
        url = DATA_URL.format(start_time=start_time, 
                              end_time=end_time, 
                              api_key=self.api)
        # get the response from the request
        response = requests.get(url, timeout=TIMEOUT)
        retry=0
        while response.status_code != 200 and retry < 20:
            response = requests.get(url, timeout=TIMEOUT)
            retry += 1
        if response.status_code != 200:
            return None
        
        response_dict = json.loads(response.text)
        if response_dict["resultsCount"] == 0:
            return None
        
        results = response_dict["results"]
        sorted_results = sorted(results, key=lambda x: x['t'])
        df = pd.DataFrame(sorted_results)
        df.rename(columns=DICT_MAPPING, inplace=True)
        df = df.loc[:, list(DICT_MAPPING.values())]

        # scrap the SMA
        while True:
            sma_url = SMA_URL.format(start_time=start_time, end_time=end_time, api_key=self.api)
            sma_response = requests.get(sma_url, timeout=TIMEOUT)
            sma_results = json.loads(sma_response.text)
            if "values" not in sma_results["results"]:
                continue
            break
        sma_df = pd.DataFrame(sma_results["results"]["values"])
        sma_df.columns = ["utc_timestamp", "sma_value"]
        sma_df = sma_df.groupby("utc_timestamp").mean().reset_index()
        sma_df = sma_df.sort_values(by="utc_timestamp").reset_index(drop=True)
        df = df.merge(sma_df, on="utc_timestamp", how="left")
        
        # Scrape the EMA
        while True:
            ema_url = EMA_URL.format(start_time = start_time,
                                    end_time = end_time, api_key = self.api)
            ema_response = requests.get(ema_url, timeout=TIMEOUT)
            ema_results = json.loads(ema_response.text)
            if "values" not in ema_results["results"]:
                continue
            break 
        ema_df = pd.DataFrame(ema_results["results"]["values"])
        ema_df.columns = ["utc_timestamp", "ema_value"]
        ema_df = ema_df.groupby("utc_timestamp").mean().reset_index() 
        ema_df = ema_df.sort_values(by="utc_timestamp").reset_index(drop=True)
        df = df.merge(ema_df, on="utc_timestamp", how="left")

        # Scrape the macd
        while True:
            macd_url = MACD_URL.format(start_time=start_time, end_time=end_time, api_key=self.api)
            macd_response = requests.get(macd_url, timeout=TIMEOUT)
            macd_results = json.loads(macd_response.text)
            if "values" not in macd_results["results"]:
                continue
            break
        macd_df = pd.DataFrame(macd_results["results"]["values"])
        macd_df.columns = ["utc_timestamp", "macd_val", "macd_sig", "macd_hist"]
        macd_df = macd_df.groupby("utc_timestamp").mean().reset_index()
        macd_df = macd_df.sort_values(by="utc_timestamp").reset_index(drop=True)
        df = df.merge(macd_df, on="utc_timestamp", how="left") 

        # Scrape the RSI
        while True:
            rsi_url = RSI_URL.format(start_time=start_time, end_time=end_time, api_key=self.api)
            rsi_response = requests.get(rsi_url, timeout=TIMEOUT)
            rsi_results = json.loads(rsi_response.text)
            if "values" not in rsi_results["results"]:
                continue
            break
        rsi_df = pd.DataFrame(rsi_results["results"]["values"])
        rsi_df.columns = ["utc_timestamp", "rsi_value"]
        rsi_df = rsi_df.groupby("utc_timestamp").mean().reset_index()
        rsi_df = rsi_df.sort_values(by="utc_timestamp").reset_index(drop=True)
        df = df.merge(rsi_df, on="utc_timestamp", how="left") 

        utc_times = df["utc_timestamp"]
        # Add the news in the df
        rows = [
            {"utc_timestamp" : utc_times[0], "news_text" : []}
        ] # zero row 
        for i in range(1, len(utc_times)):
            lower_time = utc_times[i - 1]
            upper_time = utc_times[i]
            url = (NEWS_URL.format(lower_time=lower_time, upper_time=upper_time, api=self.api))
            response = requests.get(url, timeout=120)
            response_json = (json.loads(response.text))
            response_result = response_json["results"]
            row_dict =  {"utc_timestamp" : upper_time}
            # news list 
            news_text_lst = []
            for news in response_result: 
                text = news["title"]  + news["description"] if "description" in news else news["title"]
                news_text_lst.append(text)
            row_dict["news_text"] = news_text_lst
            rows.append(row_dict)
        news_df = pd.DataFrame(rows)
        df = df.merge(news_df, on="utc_timestamp", how="left") 
        
        # add dollar index
        rows = []
        prev_dollar_index = -1
        for i, dollar_end in enumerate(utc_times):
            # (start, end) alway put the dollar index to the end timestamp
            dollar_start = dollar_end - 1000 * 72 * 60 * 60 if i == 0 else utc_times[i - 1]
            dollar_response = requests.get(DOLLAR_URL.format(start_time=dollar_start, end_time=dollar_end, api=self.api), timeout=120)
            dollar_response = json.loads(dollar_response.text)
            
            # check error
            if i == 0 and dollar_response["resultsCount"] == 0:
                raise Exception("Dollar index is not found before")
            if dollar_response["resultsCount"] > 0:
                row = {
                    "utc_timestamp" : dollar_end,
                    "dollar_index" : (dollar_response["results"][-1]['c'])
                }
                prev_dollar_index = dollar_response["results"][-1]['c']
            else:
                row = {
                    "utc_timestamp" : dollar_end,
                    "dollar_index" : prev_dollar_index
                }
            rows.append(row)
        dollar_df = pd.DataFrame(rows)
        df = df.merge(dollar_df, on="utc_timestamp", how="left") 

        return df
    
    def scrape_days(self):
        """Scrape the stock in the interval [self.start-time, self.end-time]"""
        start_date, end_date = self.start_time.date(), self.end_time.date()
        file_path = self.storage_dir / f"{start_date}-{end_date}.csv"
        print(f"Start Stock {start_date}-{end_date}")
        cur_datetime = self.start_time
        all_df = pd.DataFrame([], columns=DICT_MAPPING.values())
        while cur_datetime <= self.end_time:
            df = self.scrape_single_day(cur_datetime)
            if df is not None:
                all_df = pd.concat([all_df, df], ignore_index = True) if df is not None else df
            #update 
            cur_datetime += timedelta(days=1)
        # write the data to local disk 
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)
        
        # convert the start and end time into the date format
        print(f"Finish Stock {start_date}-{end_date}(len={len(all_df)})")
        all_df.to_csv(file_path, index=False)
        return file_path


class StockDataSraper():
    """This is a stockscraper. Multi-process data scraper"""
    def __init__(self, start_time : str, end_time : str, storage_dir = "../raw_data", api_key = os.getenv("api"), n_proccesses = 80):
        self.start_time = date_to_time(start_time)
        self.end_time = date_to_time(end_time)
        if api_key is None:
            raise FileExistsError("STOCK API DOES NOT EXIST")
        self.api = api_key
        self.executor = ProcessPoolExecutor(n_proccesses)
        # intialize the storage directory if the directory does not exist
        self.storage_dir = Path(storage_dir) / f"{self.start_time.date()}-{self.end_time.date()}"
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)

    def generate_time_interval(self, step = timedelta(days=1)):
        """Generate multiple intervals to scrape data with multiple processes
        Return: the intervals contain interval representing the date in string type
        """
        time_intervals = []
        cur_date, end_date = self.start_time, self.end_time
        while True:
            upper_date = cur_date + step
            interval = [str(cur_date.date()), str(upper_date.date()) if upper_date <= end_date else str(end_date.date())]
            time_intervals.append(interval)
            cur_date = (upper_date + timedelta(days=1))
            if cur_date > end_date:
                break
        return time_intervals
    
    def scrape_data(self):
        """Scrape the data"""
        workers = []
        time_intervals = self.generate_time_interval()
        for start_time, end_time in time_intervals:
            workers.append(StockScrapeWorker(start_time, end_time, storage_dir=self.storage_dir))
        futures = [self.executor.submit(worker.scrape_days) for worker in workers]
        file_paths = [future.result() for future in futures]
        # merge data together into a single file
        all_df : pd.DataFrame = None
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            # delete the dataframe in the disk
            os.remove(file_path)
            if len(df) > 0:
                all_df = pd.concat([all_df, df], ignore_index=True) if all_df is not None else df
        all_df = all_df.sort_values(by="utc_timestamp")
        # store the dataframe into 
        storage_dir = Path(self.storage_dir)
        all_df.to_csv(storage_dir / "all_df.csv", index=False) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    args = parser.parse_args()
    StockDataSraper(start_time=args.start_date, end_time=args.end_date).scrape_data()

if __name__ == "__main__":
    main()