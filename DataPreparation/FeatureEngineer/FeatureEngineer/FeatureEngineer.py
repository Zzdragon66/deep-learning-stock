from FeatureGenerator import FeatureGenerator
from sqlalchemy import create_engine
import argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd 

DATABASE_URL = "postgresql://{username}:{password}@localhost/{db_name}"

class FeatureEngineer():
    """
        Use multiple process to generate the features from the existing features 
        Write the data back to postgresql database with new features and table name is `engineeredtable`
    """

    def __init__(self, db_name : str, db_user : str, db_password : str, n_processes = 10, lookback = 15):
        """Initalize the feature engineer"""
        self.db_engine = self.init_db_connection(db_name, db_user, db_password)
        self.n_processes = n_processes
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.lookback = lookback


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
    
    def distribute_task(self):
        """Distribute the feature engineer tasks across multiple processes"""
        # get the number of raws in the db
        query_str = """
        select count(*) as n_rows 
        from rawtable
        """
        n_rows = pd.read_sql(query_str, self.db_engine).values[0][0]
        step = n_rows // self.n_processes
        cnt_record = []
        start, end = 1, step
        while end < n_rows:
            cur_range = (start, end)
            cnt_record.append(cur_range)
            start = min(end + 1, n_rows)
            end = min(start + step, n_rows)
            start -= self.lookback
        cnt_record.append((start, end))


        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            jobs = [executor.submit(FeatureGenerator(self.db_name, self.db_user, self.db_password, start_cnt, end_cnt, self.lookback).generate_feature) for start_cnt, end_cnt in cnt_record]
            results = [job.result() for job in jobs]
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="stock")
    parser.add_argument("--db_user", type=str, default="stock")
    parser.add_argument("--db_password", type=str, default="stock")
    args = parser.parse_args()
    feature_engineer = FeatureEngineer(args.db_name, args.db_user, args.db_password)
    feature_engineer.distribute_task()


if __name__ == "__main__":
    main()