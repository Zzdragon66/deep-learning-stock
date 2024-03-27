from pathlib import Path
import numpy as np
import pandas as pd 
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import joblib

MEAN_COLS = ["volume", "n_transactions"]
STD_COLS = ["volume", "curr_price", "high_price", "low_price", "n_transactions"]
LABELS_COLS = ["news_label"]

class FeatureEngineer():

    def __init__(self, df_path : str):
        df_path = Path(df_path)
        if not df_path.exists():
            raise FileNotFoundError("Dataframe is not found")
        self.df = pd.read_csv(df_path)
        print(self.df.columns)
        self.df_path = df_path
    
    def generate_onehot(self):
        onehot = OneHotEncoder(sparse_output=False)
        onehot_df = pd.DataFrame(onehot.fit_transform(self.df[["news_label"]].values), columns = ["Neg", "Neu", "Pos"])

        for col in onehot_df:
            self.df[col] = onehot_df[col]
        self.df = self.df.drop("news_label", axis = 1)
        return self.df
    
    def generate_trading_time(self):
        df_time = pd.to_datetime(self.df['utc_timestamp'], unit='ms')
        hr_column = df_time.dt.hour
        min_column = df_time.dt.minute
        self.df["hour"] = hr_column
        self.df["minute"] = min_column
        self.df["weekday"] = df_time.dt.weekday 
        print(self.df.columns)
        return self.df

    def moving(self, points=10, in_place = True):
        # return the moving average of the dataframe
        col_names = [col + f"_avg{points}" for col in MEAN_COLS]
        new_cols = self.df.loc[:, MEAN_COLS].rolling(window=points).mean()
        new_cols.columns=col_names
        new_df = pd.concat([self.df, new_cols], axis=1)
        # moving std
        col_names = [col + f"_std{points}" for col in STD_COLS]
        new_cols = self.df.loc[:, STD_COLS].rolling(window=points).std()
        new_cols.columns=col_names
        new_df = pd.concat([new_df, new_cols], axis=1) 
        if in_place:
            self.df = new_df
            return self.df
        return new_df
    
    def minmaxscaler(self, inplace=True, models_dir : str = "../models", filename = "minmaxscaler.save"):
        #scaler = MinMaxScaler().fit_transform()
        feature_cols = [colname for colname in self.df.columns if colname != "utc_timestamp" and colname not in LABELS_COLS]
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(self.df.loc[:, feature_cols])
        self.df.loc[:, feature_cols] = scaled_values
                                            
        # save the minmaxscaler to the models directory
        scaler_filename = Path(models_dir) / filename
        joblib.dump(scaler, str(scaler_filename))
        return self.df
    
    def remove_na(self, inplace=True):
        mask = self.df.isna().any(axis=1)
        new_df = self.df.loc[~mask, :]
        if inplace:
            self.df = new_df
            return self.df
        return new_df
        
    def write_data(self, write_path):
        print()
        self.df.to_csv(write_path, index=False)

    def get_df(self):
        return self.df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    args = parser.parse_args()
    feature_engineer = FeatureEngineer(df_path=args.df_path)
    feature_engineer.moving()
    feature_engineer.minmaxscaler()
    feature_engineer.remove_na()


if __name__ == "__main__":
    main()
        

