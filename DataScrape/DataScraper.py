from datetime import datetime, timedelta
import pandas as pd 
import pandas as pd 
from pathlib import Path
import argparse
import pytz
import os
from concurrent.futures import ProcessPoolExecutor
from DataScraperWorker import DataScrapeWorker

#Helper function 1
def date_to_time(date_str : str):
    """Utility function to convert date string to dates"""
    format = "%Y-%m-%d"
    time_val = datetime.strptime(date_str, format)
    return time_val

class StockDataSraper():
    """This is a stockscraper. Multi-process data scraper"""
    def __init__(self, start_time : str, end_time : str, api : str, 
                db_name : str, db_user : str, db_password : str,
                 n_proccesses = 12):
        self.start_time = date_to_time(start_time)
        self.end_time = date_to_time(end_time)
        self.db_config = {
            "db_name" : db_name,
            "db_user" : db_user,
            "db_password" : db_password
        }
        self.api = api
        self.executor = ProcessPoolExecutor(n_proccesses)
        
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
            workers.append(
                DataScrapeWorker(start_time, 
                                  end_time, 
                                  api = self.api,
                                  db_name = self.db_config["db_name"],
                                  db_user = self.db_config["db_user"],
                                  db_pwd = self.db_config["db_password"]))
        futures = [self.executor.submit(worker.scrape_days) for worker in workers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--api", type=str, required=True)
    parser.add_argument("--db_name", type=str, default="stock")
    parser.add_argument("--db_user", type=str, default="stock")
    parser.add_argument("--db_password", type=str, default="stock")

    args = parser.parse_args()
    scraper = StockDataSraper(start_time=args.start_date, 
                    end_time=args.end_date,
                    api = args.api,
                    db_name=args.db_name,
                    db_user=args.db_user,
                    db_password=args.db_password
                    )
    scraper.scrape_data()

if __name__ == "__main__":
    main()