import pandas as pd 
import numpy as np 
import argparse
from sqlalchemy import create_engine
import json
from pathlib import Path

DATABASE_URL = "postgresql://{username}:{password}@localhost/{db_name}"

class MinMaxScaler():

    def __init__(self, db_name : str, db_user : str, db_password : str, minmax_storage : str):
        # perform the minmax scale 
        self.db_config = {
            "db_name" : db_name,
            "db_user" : db_user,
            "db_password" : db_password
        }
        self.minmaxstorage = Path(minmax_storage)
        if not self.minmaxstorage.exists():
            raise FileNotFoundError("MinMax stat storage missing")

    def init_db_connection(self, db_name : str, db_user : str, db_password: str):
        """Initialize the database connection"""
        engine = create_engine(DATABASE_URL.format(
            username = db_user,
            password = db_password,
            db_name = db_name
            ), 
            isolation_level="AUTOCOMMIT"
        )
        return engine

    def minmaxscale(self):
        """
            1. Query the data to get numerical features 
            2. Get the data
            3. Perform the min/max scale on each column
            3. Write the data back to db and statistics write to model file
        """
        # initialize the db engine 
        db_engine = self.init_db_connection(
            db_name = self.db_config["db_name"],
            db_user = self.db_config["db_user"],
            db_password=self.db_config["db_password"]
        )

        # Step 1
        col_query = "select * from engineeredtable limit 1"
        feature_cols = list(filter(lambda x: x!="utc_timestamp", pd.read_sql(col_query, db_engine).columns))
        print("Finish Step 1")
        # Step 2
        data_query = "select * from engineeredtable order by utc_timestamp"
        df = pd.read_sql(data_query, db_engine)
        print("Finish step 2")
        # Step 3
        minmax_stat = dict()
        for feature_col in feature_cols:
            colmin = min(df.loc[:, feature_col])
            colmax = max(df.loc[:, feature_col])
            minmax_stat[feature_col] = [colmin, colmax]
            col_val = df.loc[:, feature_col]
            col_val = (col_val - colmin) / (colmax - colmin)
            df.loc[:, feature_col] = col_val
        print("Finish step 3")
        # Step 4: write data back to db
        try:
            with db_engine.begin() as connection:
                df.to_sql(
                    name = "transformedtable",
                    con = connection,
                    index = True,
                    if_exists = "append",
                    method = "multi",
                    chunksize=1000
                )
        except Exception as e:
            print("Error during insertion: ", e)
        print("Finish writing to db")
        # write the json file 
        with open(self.minmaxstorage / "minmax.json", "w") as f:
            json.dump(minmax_stat, f, indent=4)
        print("Finish writing to json file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default = "stock")
    parser.add_argument("--db_user", type=str, default = "stock")
    parser.add_argument("--db_password", type=str, default="stock")
    parser.add_argument("--minmax_storage_dir", type=str, default="../models")
    args = parser.parse_args()
    minmax_scaler = MinMaxScaler(args.db_name, args.db_user, args.db_password, args.minmax_storage_dir)
    minmax_scaler.minmaxscale()

if __name__ == "__main__":
    main()