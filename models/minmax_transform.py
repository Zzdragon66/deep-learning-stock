# load the minmax data and perform the transform/inverse-transform

import json
from pathlib import Path

class MinMaxScaler():

    def __init__(self, model_dir : Path):
        if not model_dir.exists():
            raise FileNotFoundError("Model directory file is not found")
        self.scaler_stat = None
        with open(model_dir / "./minmax.json", "r") as f:
            self.scaler_stat = json.load(f)
    

    def transform(self, df):
        for colname in self.scaler_stat:
            colmin, colmax = self.scaler_stat[colname]
            transformed_val = (df[colname] - colmin) / (colmax - colmin)
            df[colname] = transformed_val
        return df


    def inverse_transform(self, array, target_col = "curr_price"):
        """Inverse transform the columns"""
        # assume input has shape of (N, 1)
        if len(array.shape) == 2:
            N, _ = array.shape
            array = array.reshape(N)
        colmin, colmax = self.scaler_stat[target_col]
        transformed_val = array * (colmax - colmin) + colmin
        return transformed_val