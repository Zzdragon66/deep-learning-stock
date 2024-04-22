import numpy as np
import pandas as pd 
import argparse 
from pathlib import Path
import sys
import joblib
import pytz
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor

sys.path.append("../")
from FeatureEngineering.FeatureEngineer import FeatureEngineer

FEATURE_COLS = ["volume", "curr_price", "high_price", "low_price", "n_transactions"]
TARGET_COL = "curr_price"


class TrainDataGeneratorWorker():
    """worker is responsible for generating the train data with lookback
    the number of output should be N - lookbacks
    """
    
    def __init__(self, input_df : pd.DataFrame, write_dir : str, batch_idx, lookback = 1000):
        """Initialize the TrainDataGenerator

        Args:
            input_df (pd.DataFrame): input dataframe
            write_dir (str): the write directory 
            batch_idx (_type_): the batch number for the input dataframe
            lookback (int, optional): _description_. Defaults to 1000.
        """
        # Make the X and y directory 
        write_dir_path = Path(write_dir)
        self.X_dir, self.y_dir = write_dir_path / "X", write_dir_path / "y"
        
        _ = self.X_dir.mkdir() if not self.X_dir.exists() else None
        _ = self.y_dir.mkdir() if not self.y_dir.exists() else None

        self.batch_idx = batch_idx
        self.df, self.lookback = input_df, lookback

    def generate_train_data(self):
        """Generate the train data from the input_df"""
        X, y, len_data = [], [], len(self.df)
        feature_cols = [col for col in self.df.columns if col != "utc_timestamp"]
        for i in range(self.lookback, len_data):
            input_idxs, output_idx = range(i - self.lookback, i), i
            X_i = self.df.iloc[input_idxs, :].loc[:, feature_cols].values  
            y_i = self.df.iloc[output_idx].loc[TARGET_COL]
            _, _ = X.append(X_i), y.append(y_i)
        X_arr, y_arr = np.array(X), np.array(y)
        X_arr_filename, y_arr_filename = self.X_dir / f"{self.batch_idx}.npy", self.y_dir / f"{self.batch_idx}.npy"
        _, _ = np.save(X_arr_filename, X_arr), np.save(y_arr_filename, y_arr)
        print(f"Finish Batch {self.batch_idx} data generation.")

class TrainDataGenerator():
    """Generate the train data based on the features """
    def __init__(self, start_date : str, end_date : str, if_feature_engineer = False, lookback = 1000, raw_dir : str = "../raw_data", write_dir = "../train_data"):
        # Get the raw and write directory
        date_dir = f"{start_date}-{end_date}"
        self.raw_dir, self.write_dir = Path(raw_dir), Path(write_dir)
        self.raw_dir, self.write_dir = self.raw_dir/date_dir, self.write_dir/date_dir
        print(self.raw_dir)
        self.write_dir = self.write_dir / "feature_engineer" if if_feature_engineer else self.write_dir / "not_feature_engineer"
        self.lookback = lookback
        self.scaled_df = self.load_raw_data(if_feature_engineer)
        

    def load_raw_data(self, if_feature_engineer = False):
        """Load the raw data in raw data directory and return the raw data in a single dataframe"""
        if not self.raw_dir.exists():
            raise FileExistsError("Raw data directory does not exist")
        df_path = str(self.raw_dir / "all_df.csv")
        
        fe = FeatureEngineer(df_path)
        if if_feature_engineer:
            #fe.generate_trading_time()
            fe.moving()
            fe.minmaxscaler(filename="engineer_scaler.save")
            fe.remove_na()
            fe.write_data(self.raw_dir / "engineered_df.csv")
        else:
            #fe.generate_trading_time()
            fe.minmaxscaler(filename="scaler.save")
            fe.remove_na()
            fe.write_data(self.raw_dir / "scaled_df.csv")

        return fe.get_df()
    
    def generate_dfs(self):
        """Generate the dataframes"""
        dfs = []
        df_len = len(self.scaled_df)
        step = 10000
        cur_idx = self.lookback
        while cur_idx < df_len:
            start, end = cur_idx - self.lookback, min(cur_idx + step, df_len)
            dfs.append(self.scaled_df.iloc[start:end, :])
            cur_idx = end
        return dfs
    
    def generate_train_data(self):
        """Generate the train data"""
        dfs = self.generate_dfs()

        if not self.write_dir.exists():
            self.write_dir.mkdir(parents=True)

        # multiple-process data generation
        workers = [
           TrainDataGeneratorWorker(df, self.write_dir, idx, lookback=self.lookback) for idx, df in enumerate(dfs)
        ]
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(worker.generate_train_data) for worker in workers]
            _ = [future.result() for future in futures]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--if_engineer", type=bool, default=False)
    args = parser.parse_args() 
    generator = TrainDataGenerator(start_date=args.start_date, end_date=args.end_date, if_feature_engineer=args.if_engineer) 
    print("Initialize the train data generator")
    generator.generate_train_data()

if __name__ == "__main__":
    main()
        