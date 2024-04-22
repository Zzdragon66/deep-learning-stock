import psycopg2
from psycopg2 import OperationalError
import argparse


def init_db(db_name : str, db_user : str, db_pwd : str):
    db_config = {
        "dbname" : db_name,
        "user" : db_user,
        "password" : db_pwd,
        "host" : "localhost", 
        "port" : "5432"
    }
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**db_config)
        print("Connection established")

        create_db_str = """
        CREATE TABLE IF NOT EXISTS rawtable (
            volume real,
            curr_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            utc_timestamp BIGINT,
            n_transactions BIGINT, 
            sma_value DOUBLE PRECISION,
            ema_value DOUBLE PRECISION,
            macd_val DOUBLE PRECISION,
            macd_sig DOUBLE PRECISION,
            macd_hist DOUBLE PRECISION,
            rsi_value DOUBLE PRECISION,
            PRIMARY KEY (utc_timestamp)
        )
        """ 
        #TODO(Allen) : new_test missing now
        #TODO(Allen) : add the dollar index
        #TODO(Allen): Delete the insertion string

        insert_str = """
        INSERT INTO RawTable VALUES (
            119456.0, 
            452.44, 
            452.49, 
            452.325, 
            to_timestamp(1693578360000 / 1000),
            1511, 
            452.6735100000002, 
            452.5686021183715, 
            0.0612289380588322, 
            0.120270734986005,
            -0.0590417969271728, 
            47.77993949433881, 
            '{}'::text[], 
            28.97
        );
        """

        insert_str2 = """
        INSERT INTO RawTable VALUES (
            131142.0, 451.87, 452.08, 451.84, to_timestamp(1693579200000 / 1000.0),
            1552, 452.30369500000006, 452.2366754939516, -0.1216644620575948, -0.087646529505465,
            -0.0340179325521298, 41.28528630504331, ARRAY['Jobs +187K, Unemployment +3.8%: Cooling but HealthyThis looks like a very favorable report, labor market- and inflation-wise. Goldilocks, in fact.'], 28.995
        );
        """

        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(create_db_str)
        conn.commit()
        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"Unable to connect to the database: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--db_user", type=str, required=True)
    parser.add_argument("--db_pwd", type=str, required=True)
    args = parser.parse_args()
    init_db(args.db_name, args.db_user, args.db_pwd)