import numpy as np
from pathlib import Path
import os

from concurrent.futures import ProcessPoolExecutor

def extract_number(file_name_str : str)->int:
    return int(file_name_str.split(".")[0].split("-")[-1])

def load_with_key(npy_dir, X_file):
    """Load the data while preserving the ordering"""
    # load the np files with key as the number 
    return (extract_number(X_file), np.load(npy_dir/X_file))

# def load_data(data_dir : Path, sub_dir : Path):
#     """Return X and y

#     Args:
#         data_dir (Path): the data directory 
#         sub_dir (Path): sub directory in the data directory
#     """

#     if not data_dir.exists() or not (data_dir / sub_dir).exists():
#         raise FileExistsError("File is not found")

#     npy_dir = data_dir / sub_dir / "npy"
#     if not npy_dir.exists():
#         raise FileExistsError("npy files are not found")
    # npy_files = os.listdir(npy_dir)

    # X_files = list(sorted(filter(lambda x:x[0] == "X", npy_files), key = extract_number))
    # y_files = list(sorted(filter(lambda x:x[0] == "y", npy_files), key = extract_number))
    # print(X_files, y_files)

    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(load_with_key, npy_dir, X_file) for X_file in X_files]
    #     X_list = [future.result() for future in futures]
    # X_list = sorted(X_list, key=lambda x:x[0])

    # print([x[0] for x in X_list])
    # X_list = list(map(lambda x:x[-1], X_list))
    # X = np.concatenate(X_list)
        
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(load_with_key, npy_dir, y_file) for y_file in y_files]
    #     y_list = [future.result() for future in futures]
    # y_list = sorted(y_list, key=lambda x:x[0])
    # print([x[0] for x in y_list])
    # y_list = list(map(lambda x:x[-1], y_list))
    # y = np.concatenate(y_list)

    # return X, y

def load_single_data(data_dir : Path, sub_dir : Path):
    """Load the numpy in the data_dir / sub_dir

    Args:
        data_dir (Path): the whole data directory
        sub_dir (Path): the subdirectory (e.g half_year)
    """

    # there should be only a single X and y
    X = np.load(data_dir / sub_dir / "single-X.npy")
    y = np.load(data_dir / sub_dir / "single-y.npy") 
    return X,y 