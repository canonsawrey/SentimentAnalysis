# Libraries
import transformers
import torch
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# NOTE: may need to install transformers for running this for the first time

# Local
from training_data_source import TrainingDataSource
from model import Model

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
MAX_LENGTH = 55  # NOTE: this value will need to change with a different training data set. TODO make this dynamic
BATCH_SIZE = 16
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class BERTModel(Model):
    def __init__(self):
        self.model: transformers.BertModel = None

    # Builds a model_info object from a DataSource
    def build_from_data_source(self, data_source: TrainingDataSource):
        df = data_source.df_data()
        train_data_loader = create_data_loader(df, tokenizer, MAX_LENGTH, BATCH_SIZE)
        self.model = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.sentence.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
