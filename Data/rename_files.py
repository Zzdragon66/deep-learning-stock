import os 
import sys
import pandas as pd
from pathlib import Path
import argparse


def rename_files(dir : str):
    dir_path = Path(dir)
    csv_files = sorted(list(filter(lambda x: x[-4:] == ".csv", os.listdir(dir_path))))
    removal_files = list(filter(lambda x: x[-4:] != ".csv", os.listdir(dir_path)))
    # remove the files not csv 
    for file in removal_files:
        os.remove(dir_path / file)
    # change the csv file names
    mapping = []
    for csv_file in csv_files:
        temp_df = pd.read_csv(dir_path / csv_file,
                    nrows=1)
        mapping.append((csv_file, temp_df["index"].values[0]))
    sorted_filenames = sorted(mapping, key = lambda x:x[1])
    for i, filename in enumerate(sorted_filenames):
        org_name = dir_path / filename[0]
        changed_name = dir_path / f"{i}.csv" 
        os.rename(org_name, changed_name)
        print(f"Change from {org_name} to {changed_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    rename_files(args.dir)

if __name__ == "__main__":
    main()