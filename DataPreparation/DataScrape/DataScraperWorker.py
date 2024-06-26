import pandas as pd 
import numpy as np 
import os
import pytz
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json
import argparse
from sqlalchemy import create_engine

TIMEOUT = 60

DATA_URL = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{start_time}/{end_time}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

SMA_URL = "https://api.polygon.io/v1/indicators/sma/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=20&series_type=close&order=desc&apiKey={api_key}&limit=5000"

EMA_URL = "https://api.polygon.io/v1/indicators/ema/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=20&series_type=close&order=desc&apiKey={api_key}&limit=5000"

MACD_URL = "https://api.polygon.io/v1/indicators/macd/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&short_window=12&long_window=26&signal_window=9&series_type=close&order=desc&apiKey={api_key}&limit=5000"

RSI_URL = "https://api.polygon.io/v1/indicators/rsi/SPY?timestamp.gte={start_time}&timestamp.lt={end_time}&timespan=minute&adjusted=true&window=14&series_type=close&order=desc&apiKey={api_key}&limit=5000"

NEWS_URL = """https://api.polygon.io/v2/reference/news?ticker=SPY&published_utc.gte={lower_time}&published_utc.lt={upper_time}&order=asc&limit=1000&sort=published_utc&apiKey={api}"""

DOLLAR_URL = "https://api.polygon.io/v2/aggs/ticker/UUP/range/1/minute/{start_time}/{end_time}?adjusted=true&sort=asc&limit=50000&apiKey={api}"

DICT_MAPPING = {"v":"volume", 'c' : "curr_price", "h" : "high_price", "l" : "low_price", "t" : "utc_timestamp", "n" : "n_transactions"}

# Helper Function 1

def date_to_time(date_str : str):
    """Utility function to convert date string to dates"""
    format = "%Y-%m-%d"
    time_val = datetime.strptime(date_str, format)
    return time_val
# Helper Function 2
def generate_trading_time(cur_date : datetime):
    """Generate the UTC timestamp for trading day

    Args:
        cur_date (datetime): datetime
    """
    new_york_tz = pytz.timezone('America/New_York')
    trading_start = cur_date.replace(hour=9, minute=30, second=0, tzinfo=new_york_tz)
    trading_end = cur_date.replace(hour=16, minute=0, second=0, tzinfo=new_york_tz)
    start_timestamp, end_timestamp = int(trading_start.astimezone(pytz.utc).timestamp()*1000), int(trading_end.astimezone(pytz.utc).timestamp()*1000)
    return start_timestamp, end_timestamp

class DataScrapeWorker():
    def __init__(self, 
                 start_time : str, 
                 end_time : str, 
                 api : str, 
                db_name : str, 
                db_user : str, 
                db_pwd : str):
        self.start_time = date_to_time(start_time)
        self.end_time = date_to_time(end_time)
        self.api = api

        self.db_config = {
            "dbname" : db_name,
            "user" : db_user,
            "password" : db_pwd,
            "host" : "localhost", 
            "port" : "5432"
        }
    
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
        
        print(f"{cur_date} Finish scraping basic data")

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
        print(f"{cur_date} Finish scraping SMA data")
        
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
        print(f"{cur_date} Finish scraping EMA data")

        # Scrape the macd
        try:
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
            print(f"{cur_date} Finish scraping MACD data")
        except:
            print("Error on MACD")
            raise NotImplementedError

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
        print(f"{cur_date} Finish scraping RSI data")
        
        
        #utc_times = df["utc_timestamp"]

        # Add the news in the df
        # TODO(Allen): Handle this later
        # rows = [
        #     {"utc_timestamp" : utc_times[0], "news_text" : ""}
        # ] # zero row 
        # for i in range(1, len(utc_times)):
        #     lower_time = utc_times[i - 1]
        #     upper_time = utc_times[i]
        #     url = (NEWS_URL.format(lower_time=lower_time, upper_time=upper_time, api=self.api))
        #     response = requests.get(url, timeout=TIMEOUT)
        #     response_json = (json.loads(response.text))
        #     response_result = response_json["results"]
        #     row_dict =  {"utc_timestamp" : upper_time}
        #     text = ""
        #     if len(response_result) > 0:
        #         news = response_result[0] 
        #         text = news["title"]  + news["description"] if "description" in news else news["title"] 
        #     row_dict["news_text"] = text
        #     rows.append(row_dict)
        # news_df = pd.DataFrame(rows)
        # df = df.merge(news_df, on="utc_timestamp", how="left") 
        # print(f"{cur_date} Finish scraping NEWS data")
        
        # add dollar index
        # rows = []
        # prev_dollar_index = -1
        # for i, dollar_end in enumerate(utc_times):
        #     # (start, end) alway put the dollar index to the end timestamp
        #     dollar_start = dollar_end - 1000 * 72 * 60 * 60 if i == 0 else utc_times[i - 1]
        #     dollar_response = requests.get(DOLLAR_URL.format(start_time=dollar_start, end_time=dollar_end, api=self.api), timeout=120)
        #     dollar_response = json.loads(dollar_response.text)
            
        #     # check error
        #     if i == 0 and dollar_response["resultsCount"] == 0:
        #         raise Exception("Dollar index is not found before")
        #     if dollar_response["resultsCount"] > 0:
        #         row = {
        #             "utc_timestamp" : dollar_end,
        #             "dollar_index" : (dollar_response["results"][-1]['c'])
        #         }
        #         prev_dollar_index = dollar_response["results"][-1]['c']
        #     else:
        #         row = {
        #             "utc_timestamp" : dollar_end,
        #             "dollar_index" : prev_dollar_index
        #         }
        #     rows.append(row)
        # dollar_df = pd.DataFrame(rows)
        # df = df.merge(dollar_df, on="utc_timestamp", how="left")
        # print(f"{cur_date} Finish scraping Dollar data")
        return df.dropna()
    
    def scrape_days(self):
        """Scrape the stock in the interval [self.start-time, self.end-time]"""
        start_date, end_date = self.start_time.date(), self.end_time.date()
        
        print(f"Start Stock {start_date}-{end_date}")
        cur_datetime = self.start_time
        all_df = pd.DataFrame([], columns=DICT_MAPPING.values())
        while cur_datetime <= self.end_time:
            df = self.scrape_single_day(cur_datetime)
            if df is not None and len(df) > 0:
                all_df = pd.concat([all_df, df], ignore_index = True)
            #update 
            cur_datetime += timedelta(days=1)
        # make sure the datafram is not None
        if all_df is not None and len(all_df) > 0:
            self.db_insert(all_df) # insert data into database

    def db_insert(self, df : pd.DataFrame):
        """Insert the data into the database
        """
        engine = create_engine(
            "postgresql://{username}:{password}@localhost:5432/{db_name}".format(
                username = self.db_config["user"],
                password = self.db_config["password"],
                db_name = self.db_config["dbname"]
            ),
            isolation_level="AUTOCOMMIT"
        )
        try:
            with engine.begin() as connection:
                df.to_sql(
                    name = "rawtable",
                    con = connection,
                    index = False,
                    if_exists = "append",
                    method = "multi",
                    chunksize=500
                )
        except Exception as e:
            print("Error during insertion: ", e)

def main():
    #__init__(self, start_time : str, end_time : str, api : str, 
    #            db_name : str, db_user : str, db_pwd : str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--api",type=str, required=True)
    parser.add_argument("--db_name", type=str, default="stock")
    parser.add_argument("--db_user", type=str, default="stock")
    parser.add_argument("--db_pwd", type=str, default="stock")
    args = parser.parse_args()
    worker = DataScrapeWorker(
        start_time=args.start_time,
        end_time = args.end_time,
        api = args.api,
        db_name = args.db_name,
        db_user = args.db_user,
        db_pwd = args.db_pwd
    )

if __name__ == "__main__":
    main()