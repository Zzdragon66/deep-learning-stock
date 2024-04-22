from datetime import datetime
import pytz
def generate_trading_time(cur_date : datetime):
    """Generate the UTC timestamp for trading day

    Args:
        cur_date (datetime): datetime
    """
    new_york_tz = pytz.timezone('America/New_York')
    trading_start = cur_date.replace(hour=9, minute=30, second=0, tzinfo=new_york_tz)
    trading_end = cur_date.replace(hour=16, minute=0, second=0, tzinfo=new_york_tz)
    start_timestamp, end_timestamp = int(trading_start.astimezone(pytz.utc).timestamp()*1000), int(trading_end.astimezone(pytz.utc).timestamp()*1000)
    return start_timestamp, end_timestamp