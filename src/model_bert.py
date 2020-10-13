# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# including some code from there

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Local imports
from model import Model
from training_data_source import TrainingDataSource
from util import standardize

WEIGHTS = 'distilbert-base-uncased'


# Takes a little while, since it actually runs the model on all sentences.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:, 0, :].numpy()


"""

BERTModel - Actually an instance of DistilBERT - lighter and faster than BERT, while coming close in ability
    set_size: how many entries should be used in training
    lr: The LogisticRegression model that will be used on the output of BERT
    tokenizer: BERT Tokenizer
    model: BERT Model
    max_len: Maximum length token found in training set. All vectors will be truncated or padded to this length
    scaler: Used to make training the LR model easier and more likely to converge

"""


class BERTModel(Model):
    def __init__(self, set_size=2000):
        self.set_size = set_size
        self.lr = LogisticRegression(max_iter=2000)
        self.tokenizer = ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)
        self.model = ppb.DistilBertModel.from_pretrained(WEIGHTS)
        self.max_len = 0
        self.scaler = StandardScaler()

    # We want the sentences to all be the same length; pad with 0's to make it so
    def pad_tokens(self, tokenized):
        for i in tokenized.values:
            if len(i) > self.max_len:
                self.max_len = len(i)
        padded = np.array([i + [0] * (self.max_len - len(i)) for i in tokenized.values])
        return padded

    # Builds a model_info object from a TrainingDataSource
    def build_from_data_source(self, data_source: TrainingDataSource):
        # Download the 2000 sentiment-labeled sentences to a Pandas dataframe.
        df = data_source.df_data()[:self.set_size]
        # Turn the dataset's sentences into BERT tokens.
        tokens = df['sentences'].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

        # Pad the tokens with zeros, so that the sentences are all the same length.
        padded_tokens = self.pad_tokens(tokens)

        # Extract the vectors corresponding to the first token, CLS, for each sentence. Represents the overall
        # meaning of the sentence for classification tasks as best as possible
        vectors = get_bert_sentence_vectors(self.model, padded_tokens)
        # Scale to allow the LR model to easier train
        self.scaler.fit(vectors)

        self.lr.fit(self.scaler.transform(vectors), df['labels'])
        print("Finished training BERT\n")

    # Classifies the sentence; returns tuple: (classification, prob(classification))
    def classify(self, sentence: str):
        return self.batch_classify([sentence])[0]

    # Classifies the group of sentences; returns list of tuple: [(classification, prob(classification))]
    def batch_classify(self, sentences: [str]):
        sentences = pd.Series(data=sentences)
        # Turn the dataset's sentences into BERT tokens. Truncate if too long
        tokens = sentences.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
        tokens = pd.Series([i[:self.max_len] for i in tokens.values])
        # Pad with 0s
        padded_tokens = np.array([i + [0] * (self.max_len - len(i)) for i in tokens.values])
        vectors = get_bert_sentence_vectors(self.model, padded_tokens)
        # Run the vectors through LR model
        prediction = self.lr.predict(self.scaler.transform(vectors))
        prediction_prob = self.lr.predict_log_proba(self.scaler.transform(vectors))
        evaluations = [(prediction[i], max(standardize(prediction_prob[i]))) for i in range(0, len(prediction))]
        return evaluations
