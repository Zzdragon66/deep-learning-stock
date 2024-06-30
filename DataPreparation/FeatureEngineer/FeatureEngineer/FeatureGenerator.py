# Generate features from existing features 
import argparse
from sqlalchemy import create_engine
import pandas as pd 

DATABASE_URL = "postgresql://{username}:{password}@localhost/{db_name}"

class FeatureGenerator():

    def __init__(self, db_name : str, db_user : str, db_password : str, 
                start_cnt : int, end_cnt : int, lookback = 15):
        assert start_cnt <= end_cnt 
        
        self.db_config = {
            "db_name" : db_name,
            "db_user" : db_user,
            "db_password" : db_password
        }
        self.start_cnt = start_cnt
        self.end_cnt = end_cnt
        self.lookback = lookback

    def init_db_connection(self, db_name : str, db_user : str, db_password: str):
        engine = create_engine(DATABASE_URL.format(
            username = db_user,
            password = db_password,
            db_name = db_name
            ), 
            isolation_level="AUTOCOMMIT"
        )
        return engine
    
    def generate_feature(self):
        """Generate the feature of mean and standard deviation of the each value"""
        points = self.lookback
        query_str = \
        f"""
        select *
        from (select *, row_number() over(order by utc_timestamp) as cnt_index
        from rawtable) t1
        where cnt_index between {self.start_cnt} and {self.end_cnt}
        """
        db_engine = self.init_db_connection(
            self.db_config["db_name"],
            self.db_config["db_user"],
            self.db_config["db_password"]
        )
        MEAN_COLS = ["volume", "n_transactions"]
        STD_COLS = ["volume", "curr_price", "high_price", "low_price", "n_transactions"]
        df = pd.read_sql(query_str, db_engine)
        del df["cnt_index"]
        col_names = [col + f"_avg{points}" for col in MEAN_COLS]
        new_cols = df.loc[:, MEAN_COLS].rolling(window=points).mean()
        new_cols.columns=col_names
        new_df = pd.concat([df, new_cols], axis=1)
        # moving std
        col_names = [col + f"_std{points}" for col in STD_COLS]
        new_cols = df.loc[:, STD_COLS].rolling(window=points).std()
        new_cols.columns=col_names
        new_df = pd.concat([new_df, new_cols], axis=1)
        new_df = new_df.dropna() 
        self.write_back_db(new_df)

    def write_back_db(self, df):
        """Write data back"""
        db_engine = self.init_db_connection(
            self.db_config["db_name"],
            self.db_config["db_user"],
            self.db_config["db_password"]
        )
        try:
            with db_engine.begin() as connection:
                df.to_sql(
                    name = "engineeredtable",
                    con = connection,
                    index = False,
                    if_exists = "append",
                    method = "multi",
                    chunksize=1000
                )
        except Exception as e:
            print("Error during insertion: ", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="stock")
    parser.add_argument("--db_user", type=str, default="stock")
    parser.add_argument("--db_password", type=str, default="stock")
    parser.add_argument("--start_timestamp", type=str, required=True) 
    parser.add_argument("--end_timestamp", type=str, required=True)
    args = parser.parse_args()

if __name__ == "__main__":
    main()