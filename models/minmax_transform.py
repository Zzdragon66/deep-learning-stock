# load the minmax data and perform the transform/inverse-transform

import json


class MinMaxScaler():

    def __init__(self):
        self.scaler_stat = None
        with open("./minmax.json", "r") as f:
            self.scaler_stat = json.load(f)
    

    def transform(self, df):
        for colname in self.scaler_stat:
            colmin, colmax = self.scaler_stat[colname]
            transformed_val = (df[colname] - colmin) / (colmax - colmin)
            df[colname] = transformed_val
        return df


    def inverse_transform(self, df):
        for colname in self.scaler_stat:
            colmin, colmax = self.scaler_stat[colname]
            transformed_val = (df[colname] * (colmax - colmin)) + colmin
            df[colname] = transformed_val
        return df