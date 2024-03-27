import unittest
from StockDataScraper import StockDataSraper
from datetime import datetime


class TestStockDataSraper(unittest.TestCase):

    def setUp(self):
        """Generate the stock data scraper"""
        self.stock_data_scaper = StockDataSraper("2024-01-01", "2024-02-29")

    def test_datetime(self):
        self.assertEqual(self.stock_data_scaper.start_time, datetime(2024, 1, 1))
        self.assertEqual(self.stock_data_scaper.end_time, datetime(2024, 2, 29))

    def test_timeinterval(self):
        time_intervals = self.stock_data_scaper.generate_time_interval()
        last_time_interval = time_intervals[-1]
        self.assertEqual(last_time_interval[-1], str(datetime(2024, 2, 29).date()))