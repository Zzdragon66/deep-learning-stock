# Deep Learning for Stock Prediction

This repository is dedicated to a deep learning project focused on predicting stock prices. It encompasses the entire process from data scraping and manipulation to the application of advanced deep learning models. The project's goal is to explore and evaluate the effectiveness of various neural network architectures in forecasting stock market trends.

## Research Paper 

[Paper](./paper.pdf)

## Overview

The project is structured around three main components: data preparation, model development, and experimentation. The data is gathered and processed to feed into different deep learning models, each designed to predict stock prices based on historical data. The models are then evaluated to determine their predictive accuracy and efficiency.

### Data Description

The datasets used in this project are derived from various stock market features and are categorized into three types, each adding more complexity and dimensions to the data:

- **stock_simple_feature**: This dataset includes basic stock market attributes like price, volume, high, low, and transaction counts.
- **stock_indicator**: Extends `stock_simple_feature` by incorporating technical indicators, providing a more detailed view of the market's behavior.
- **stock_indicator_news_dollar**: The most comprehensive dataset, including all features from `stock_indicator` plus sentiment analysis from news articles and the US dollar index, offering a holistic view of the factors influencing stock prices.
  - Sentiment labels are generated from pre-trained BERT model and fine-tuned on manually labelled news dataset.

### Tools and Libraries

The project utilizes several tools and libraries, primarily focused on Python and PyTorch:

1. **Python**: A versatile programming language used for data scraping, data manipulation, and model implementation.
   - Utilizes multiprocessing for efficient data handling.
2. **PyTorch**: A powerful deep learning framework that facilitates the construction and training of neural network models.

### Project Structure

- **Paper**: Contains the research paper detailing the methodologies, experiments, and findings of the project.
- **DataScraper**: A module for collecting stock data from online sources.
- **TrainDataGenerator**: A tool for preparing and transforming the data into a format suitable for training the deep learning models.
- **Models**: This directory houses the implementation of the neural network models used for stock prediction. 

## Getting Started

### Prerequisites

- Ensure you have Python and PyTorch installed.
- An API key from [Polygon.io](https://polygon.io/) is required for data scraping (note that this may involve costs).

### Running the Project

1. **Data Scraping**:
   - Navigate to the `DataScraper` directory.
   - Create a `.env` file containing your API key in the format `api=value`.
   - Run the script to scrape data: `python3 StockDataScraper.py --start_date "2023-09-01" --end_date "2024-03-15"`.

2. **Data Preparation**:
   - Move to the `TrainDataGenerator` directory.
   - Execute the script to generate training data: `python3 TrainDataGenerator.py --start_date "2023-09-01" --end_date "2024-03-15"`.

3. **Model Training**:
   - Go to the `models` directory.
   - Train the models using the prepared datasets and provided trainer class.
     - The training script is pending implementation, as Jupyter notebooks were initially used for model training. This will be addressed in future updates.
     - Due to GitHub's restrictions on file size, the BERT model weights are not included in this repository.

## Research and Findings

The project investigates the application of Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and a hybrid CNN-LSTM architecture in predicting stock prices. The study provides insights into how different data features and neural network architectures impact the predictive performance on stock market data.

## License

This project is open-sourced under the MIT license.
