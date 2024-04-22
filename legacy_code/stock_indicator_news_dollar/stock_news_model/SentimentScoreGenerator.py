import numpy as np 
import pandas as pd 
import sys
from pathlib import Path
import torch
from collections import Counter
from transformers import BertForSequenceClassification, BertTokenizer
import argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.append("../")
from utilities.strToDatetime import date_to_time

DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    torch.mps.empty_cache()
else:
    DEVICE = torch.device("cpu")
    torch.cuda.empty_cache()
print(DEVICE)

class SentimentScoreGenerator():
    def __init__(self, start_date : str, end_date : str):
        self.start_time = date_to_time(start_date)
        self.end_date = date_to_time(end_date)
        df_path = Path(f"../raw_data/{start_date}-{end_date}/all_df.csv")
        if not df_path.exists():
            raise FileNotFoundError(df_path)
        self.raw_df = pd.read_csv(df_path)
        print("Raw df has shape of: ", self.raw_df.shape)
        self.new_df = None
        self.model, self.tokenizer = None, None
        self.save_path = Path(f"../raw_data/{start_date}-{end_date}/news_df.csv")

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained("./bert_finetune")
        self.tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer")
        self.model.to(DEVICE)

    def generate_sentiment(self):
        self.load_model()
        news_labels = []
        with torch.no_grad():
            self.model.eval()
            for idx, news_date in enumerate(self.raw_df["news_text"]):
                print(f"Start processing data {idx}")
                cur_labels = []
                for news in news_date: # if there is data
                    inputs = self.tokenizer(news, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
                    output = self.model(**inputs)
                    prediction = torch.argmax(output.logits, dim=1).cpu().item()
                    cur_labels.append(prediction)
                most_common_label = Counter(cur_labels).most_common()[0][0] if len(cur_labels) > 0 else 1
                
                news_labels.append(most_common_label)
        self.news_df = self.raw_df.drop("news_text", axis = 1)
        self.news_df["news_label"] = news_labels
        self.news_df.to_csv(self.save_path, index=False)
        return self.news_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    args = parser.parse_args()
    senti_generator = SentimentScoreGenerator(args.start_date, args.end_date)
    senti_generator.generate_sentiment()

if __name__ == "__main__":
    main()