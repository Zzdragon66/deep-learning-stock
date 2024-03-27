import joblib
import pandas as pd 
import numpy as np
import sys
import os
from pathlib import Path

google_colab = bool(int(input("If colab(0 for no, 1 for yes)")))
if google_colab:
  cur_dir = Path("/content/drive/MyDrive/stock_project/stock_indicator_news_dollar/models/")
else:
  cur_dir = Path("../")

CURR_PRICE_IDX = 1

def load_scaler(if_engineer):
    """Load the scaler"""
    if if_engineer:
        return joblib.load(cur_dir / "engineer_scaler.save")
    return joblib.load(cur_dir / "scaler.save")

def inverse_transform(input_array, if_engineer):
    """inverse transform the input array"""
    scaler = load_scaler(if_engineer)
    N = len(input_array)
    input_array = input_array.reshape(N, 1)
    dummy_arr = np.zeros((N, scaler.n_features_in_))
    dummy_arr[:, [CURR_PRICE_IDX]] = input_array 
    inverse_arr = scaler.inverse_transform(dummy_arr)
    return inverse_arr[:, CURR_PRICE_IDX]