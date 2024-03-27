# deep-learning-stock

## Paper

[Paper](./paper.pdf)

## Project Description

* This project contains the data scraping, data manipulation, and implementation of the deep learning models for stock prediction
* Each folder corresponds to a dataset the different features 
  * `stock_simple_feature` : price, volume, high, low, transaction counts 
  * `stock_indicator`: features in `stock_simple_feature` + techinical indicators
  * `stock_indicator_news_dollar` : features in `stock_indicator` + sentiment news + us dollar index

## Tools

1. Python
   1. Multiprocessing for data scraping and data manipulation
2. Pytorch
   1. Framework for deep learning model

## Running the project

1. Make sure there is `.env` file under `DataScraper`
   1. It should contains a api key which has following format `api=value`
2. At the folder `DataSraper` folder run `python3 StockDataScraper.py --start_date "2023-09-01" --end_date "2024-03-15"`
   1. It will get the stock data from [https://polygon.io/](https://polygon.io/) (YOU NEED TO PAY FOR THE API)
3. At the folder `TrainDataGenerator` folder run `python3 TrainDataGenerator.py --start_date "2023-09-01" --end_date "2024-03-15"`
   1. Train data will be generated in the format of `.npy`
4. Go to `models` to train the models