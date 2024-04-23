# generate the npy with python multi-process
from pathlib import Path
import pandas as pd
import numpy as np 
import os
from concurrent.futures import ProcessPoolExecutor

class NumpyGenerator():
    """The worker class for the numpy generator"""

    def __init__(self, data_dir : str, file_name : str):
        self.data_dir = Path(data_dir)
        self.file_path = self.data_dir / file_name
        self.file_idx = int(file_name.split(".")[0])
        if not self.file_path.exists():
            raise FileNotFoundError("CANNOT FIND THE FILE")
        # store the numpy in the directory of npy
        self.storage_dir = self.data_dir / "npy"
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)
        print(
            f"Finish initialize the npy generator with file path {self.file_path} and storage path {self.storage_dir} and index is {self.file_idx}"
        )
        self.df = None

    def generator_numpy_array(self):
        """Generate the numpy array with single """
        if self.df is None:
            self.df = pd.read_csv(self.file_path)
        y = self.df["y"].values
        # reshape the columns in to X
        feature_cols = list(filter(lambda x: x!="y" and x != "index",self.df.columns))
        n_features = int(len(feature_cols) / 1000)
        X = self.df.loc[:, feature_cols].values
        X = X.reshape(-1, 1000, n_features)
        # store X and y
        np.save(self.storage_dir / f"X-{self.file_idx}.npy", X)
        np.save(self.storage_dir / f"y-{self.file_idx}.npy", y) 

    # def generate_sequence(self): # TODO(Allen) : need to check the spark implementation
    #     # TODO(Allen)
    #     self.df = pd.read_csv(self.file_path) if self.df is None else self.df
    #     y = self.df["y"].values
    #     # reshape the columns in to X
    #     feature_cols = list(filter(lambda x: x!="y" and x != "index",df.columns))
    #     n_features = len(feature_cols)
    #     X = self.df.loc[:, feature_cols].values
    #     X = X.reshape(-1, 1000, n_features)
    #     # store X and y
    #     np.save(self.storage_dir / f"X-{self.file_idx}.npy", X)
    #     np.save(self.storage_dir / f"y-{self.file_idx}.npy", y)  

class MultiNumpyGenerator():
    """Generate the numpy with multiple processes"""
    def __init__(self, data_dir : str):
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileExistsError("Data directory is not found")
        self.data_dir = data_dir
        self.file_names = list(filter(lambda x:x[-4:] == ".csv", os.listdir(data_path)))
    
    def generate_numpy(self):
        workers = [NumpyGenerator(self.data_dir, file_name) for file_name in self.file_names] 
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker.generator_numpy_array) for worker in workers]
            results = [future.result() for future in futures]

    def generate_numpy_sequence(self):
        #TODO(Allen): generate the y sequence
        pass
