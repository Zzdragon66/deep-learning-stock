# generate the npy with python multi-process
from pathlib import Path
import pandas as pd 

class NumpyGenerator():
    """The worker class for the numpy generator"""

    def __init__(self, data_dir : str, file_name : str):
        self.data_dir = Path(data_dir)
        self.file_path = self.data_dir / file_name
        if not self.file_path.exists():
            raise FileNotFoundError("CANNOT FIND THE FILE")
        # store the numpy in the directory of npy
        self.storage_dir = self.data_dir / "npy"
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)
        print(
            f"Finish initialize the npy generator with file path {self.file_path} and storage path {self.storage_dir}"
        )
    
    def generator_single(self):
        df = pd.read_csv(self.file_path)
        y = df["y"].values

    def generate_sequence(self):
        # TODO(Allen)
        df = pd.read_csv(self.file_path)
        pass

