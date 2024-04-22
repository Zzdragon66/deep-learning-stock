import pandas as pd
import numpy as np
import os
from pathlib import Path

TRAIN_DATE_DIR = "../train_data"

def load_data(train_data_dir : str, start_date : str, end_date : str, if_feature_engineer : bool, number_of_data = 100000):
    """Load the data from train_path
    Args:
        train_path (str, optional): _description_. Defaults to TRAIN_DATE_DIR.
        number_of_data (int, optional): _description_. Defaults to 100000.
    Raises:
        FileNotFoundError: the train path is not found
    Returns: X,y where X is the training data, y is the predicted data
    """
    train_dir = Path(train_data_dir)
    date_dir = train_dir / f"{start_date}-{end_date}"
    if if_feature_engineer:
        date_dir = date_dir / "feature_engineer"
    else:
        date_dir = date_dir / "not_feature_engineer"

    X, y = None, None
    if not date_dir.exists():
        print(os.listdir(train_dir))
        print(str(date_dir))
        raise FileNotFoundError("Train data does not exists")
    X_train_path, y_train_path = date_dir / "X", date_dir / "y"
    train_data_files = zip(sorted(os.listdir(X_train_path), key=lambda x:int(x.split(".")[0]), reverse=False),
                           sorted(os.listdir(y_train_path), key=lambda x:int(x.split(".")[0]), reverse=False))
    for X_train_file, y_train_file in train_data_files:
        print(X_train_file, y_train_file)
        X_batch, y_batch = np.load(X_train_path / X_train_file), np.load(y_train_path / y_train_file)
        X = np.concatenate([X, X_batch]) if X is not None else X_batch
        y = np.concatenate([y, y_batch]) if y is not None else y_batch
        assert len(X) == len(y)
        if len(X) >= number_of_data:
            break
    print("X data shape is: ", X.shape, "y data shape is: ",y.shape)
    return X, y


def main():
    pass

if __name__ == "__main__":
    main()