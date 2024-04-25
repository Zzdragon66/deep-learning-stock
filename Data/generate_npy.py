# generate the npy with python multi-process
from pathlib import Path
import pandas as pd
import numpy as np 
import os
from concurrent.futures import ProcessPoolExecutor
import argparse

class NumpyGenerator():
    """The worker class for the numpy generator"""

    def __init__(self, data_dir : str, file_name : str):
        self.data_dir = Path(data_dir)
        self.file_path = self.data_dir / file_name
        self.file_idx = int(file_name.split(".")[0])
        if not self.file_path.exists():
            raise FileNotFoundError("CANNOT FIND THE FILE")
        # store the numpy in the directory of npy
        self.df = None

    def get_df(self):
        """Generate the numpy array with single """
        if self.df is None:
            self.df = pd.read_csv(self.file_path)
        return self.df

        if self.df is None:
            self.df = pd.read_csv(self.file_path)
            #self.df = self.df.sort_values(by="index", ascending=True)
        y = self.df["y"].values
        # reshape the columns in to X
        feature_cols = list(filter(lambda x: x !="y" and x != "index",self.df.columns))
        n_features = int(len(feature_cols) / 1000)
        X = self.df.loc[:, feature_cols].values
        X = X.reshape(-1, 1000, n_features)
        # index 
        index = self.df["index"]
        X = np.array(list(zip(index, X)))
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
    
    def generate_single_numpy(self):
        workers = [NumpyGenerator(self.data_dir, file_name) for file_name in self.file_names]
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker.get_df) for worker in workers]
            results = [future.result() for future in futures]
        
        new_df = pd.concat(results, axis = 0).sort_values(by = "index")
        y = new_df["y"].values
        # reshape the columns in to X
        feature_cols = list(filter(lambda x: "y" not in x and x != "index",new_df.columns))

        n_features = int(len(feature_cols) / 1000)
        X = new_df.loc[:, feature_cols].values
        X = X.reshape(-1, 1000, n_features)
        print(X.shape)
        # store X and y
        np.save (Path(self.data_dir) / "single-X.npy", X)
        np.save(Path(self.data_dir) / "single-y.npy", y) 

    def generate_multi_numpy(self, n_outputs = 5):
        y_cols = [f"y{i}" for i in range(n_outputs)]
        workers = [NumpyGenerator(self.data_dir, file_name) for file_name in self.file_names]
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker.get_df) for worker in workers]
            results = [future.result() for future in futures]
        new_df = pd.concat(results, axis = 0).sort_values(by = "index")
        y = new_df.loc[:, y_cols].values
        # reshape the columns in to X
        feature_cols = list(filter(lambda x: "y" not in x and x != "index",new_df.columns))
        print(len(feature_cols))
        n_features = int(len(feature_cols) / 1000)
        X = new_df.loc[:, feature_cols].values
        X = X.reshape(-1, 1000, n_features)
        print(X.shape)
        # store X and y
        np.save (Path(self.data_dir) / "multi-X.npy", X)
        np.save(Path(self.data_dir) / "multi-y.npy", y) 

def main():
    parser = argparse.ArgumentParser("npy generation")
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--single_output", required=True, type=str)
    args = parser.parse_args()
    generator = MultiNumpyGenerator(args.dir)
    if (eval(args.single_output)):
        generator.generate_single_numpy()
    else:
        generator.generate_multi_numpy()

if __name__ == "__main__":
    main()