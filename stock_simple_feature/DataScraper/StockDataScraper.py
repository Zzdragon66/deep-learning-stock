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

DATA_URL = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{start_time}/{end_time}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

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
        # TODO(Allen): Add the news data
        start_time, end_time = generate_trading_time(cur_date)
        url = DATA_URL.format(start_time=start_time, 
                              end_time=end_time, 
                              api_key=self.api)
        # get the response from the request
        response = requests.get(url, timeout=60)
        retry=0
        while response.status_code != 200 and retry < 20:
            response = requests.get(url, timeout=60)
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
        return df

    def scrape_days(self):
        """Scrape the stock in the interval [self.start-time, self.end-time]"""
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
        start_date, end_date = self.start_time.date(), self.end_time.date()
        file_path = self.storage_dir / f"{start_date}-{end_date}.csv"
        print(f"Stock {start_date}-{end_date}(len={len(all_df)})")
        all_df.to_csv(file_path, index=False)
        return file_path


class StockDataSraper():
    """This is a stockscraper. Multi-process data scraper"""
    def __init__(self, start_time : str, end_time : str, storage_dir = "../raw_data", api_key = os.getenv("api"), n_proccesses = 24):
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

    def generate_time_interval(self, step = timedelta(days=5)):
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