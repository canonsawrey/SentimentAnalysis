# HW6 - "step 0" following Jay Alammar's tutorial,
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# including some code from there

import pandas as pd
import torch
import nltk
import transformers as ppb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# Added imports
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Location of SST2 sentiment dataset
SST2_LOC = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
WEIGHTS = 'distilbert-base-uncased'
# Performance on whole 6920 sentence set is very similar, but takes rather longer
SET_SIZE = 2000

# Download the dataset from its Github location, return as a Pandas dataframe
def get_dataframe():
    df = pd.read_csv(SST2_LOC, delimiter='\t', header=None)
    return df[:SET_SIZE]

# Extract just the labels from the dataframe
def get_labels(df):
    return df[1]

# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# We want the sentences to all be the same length; pad with 0's to make it so
def pad_tokens(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    return padded

# Grab a trained DistiliBERT model
def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

# This step takes a little while, since it actually runs the model on all sentences.
# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:,0,:].numpy()

# Start my code
# Download the 2000 sentiment-labeled sentences to a Pandas dataframe.
df = get_dataframe()
# Create a BERT tokenizer and turn the dataset's sentences into BERT tokens.
tokenizer = get_tokenizer()
tokens = get_tokens(df, tokenizer)
# Pad the tokens with zeros, so that the sentences are all the same length.
padded_tokens = pad_tokens(tokens)
# Call get_bert_sentence_vectors() to extract the vectors corresponding to the first token,
# CLS, for each sentence. This vector is trained to represent the overall meaning of the sentence
# for classification tasks as best as possible
model = get_model()
vectors = get_bert_sentence_vectors(model, padded_tokens)

# 1.
def find_closest_sentences(vecs, sentences):
    min_ind_1 = 0
    min_ind_2 = 0
    min_diff = float('INF')
    for i in range(0, len(vecs)):
        for j in range(0, len(vecs)):
            if i >= j:
                continue
            diff = np.linalg.norm(vecs[j] - vecs[i])
            if diff < min_diff and diff != 0:
                min_diff = diff
                min_ind_1 = i
                min_ind_2 = j
    print("Indices: " + str(min_ind_1) + ", " + str(min_ind_2))
    print(str(min_ind_1) + ": " + sentences[min_ind_1])
    print(str(min_ind_2) + ": " + sentences[min_ind_2])
# find_closest_sentences(vectors, df[0])

# 3.
def visualize_data(vecs, labels):
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(vecs)
    plt.scatter(pca_vecs[:, 0], pca_vecs[:, 1], c=labels)
    plt.show()
# visualize_data(vectors, get_labels(df))

# 4.
# To separate into train and test:
train_features, test_features, train_labels, test_labels = train_test_split(vectors, get_labels(df))
def train_gaussian_naive_bayes(train_features, train_labels):
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    return gnb

def train_adaboost(train_features, train_labels):
    ada = AdaBoostClassifier()
    ada.fit(train_features, train_labels)
    return ada

def train_nearest_neighbors(train_features, train_labels):
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_labels)
    return knn

def train_classic_mlp_classifier(train_features, train_labels):
    mlc = MLPClassifier(activation='logistic')
    mlc.fit(train_features, train_labels)
    return mlc

def train_deep_mlp_classifier(train_features, train_labels):
    dmlc = MLPClassifier(hidden_layer_sizes=(100,100,))
    dmlc.fit(train_features, train_labels)
    return dmlc

def train_logistic_regression(train_features, train_labels):
    lr = LogisticRegression()
    lr.fit(train_features, train_labels)
    return lr

# General purpose scikit-learn classifier evaluator.  The classifier is trained with .fit()
def evaluate(classifier, test_features, test_labels):
    return classifier.score(test_features, test_labels)

# gnb_classifier = train_gaussian_naive_bayes(train_features, train_labels)
# print('GaussionNB: ' + str(evaluate(gnb_classifier, test_features, test_labels)))
# ada_classifier = train_adaboost(train_features, train_labels)
# print('ADA: ' + str(evaluate(ada_classifier, test_features, test_labels)))
# knn_classifier = train_nearest_neighbors(train_features, train_labels)
# print('KNN: ' + str(evaluate(knn_classifier, test_features, test_labels)))
# mlp_classifier = train_classic_mlp_classifier(train_features, train_labels)
# print('Classic MLP: ' + str(evaluate(mlp_classifier, test_features, test_labels)))
# deep_mlp_classifier = train_deep_mlp_classifier(train_features, train_labels)
# print('Deep MLP: ' + str(evaluate(deep_mlp_classifier, test_features, test_labels)))
# lr_classifier = train_logistic_regression(train_features, train_labels)
# print('LR: ' + str(evaluate(lr_classifier, test_features, test_labels)))

# 8.
# print(df[0][11])

# 9.
# tokens = nltk.word_tokenize(df[0][11])
# tagged = nltk.pos_tag(tokens)
# entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

# 12.
def sentiment_classifier(classifier, bert_model, text):
    tokens = nltk.word_tokenize(text)
    vecs = get_bert_sentence_vectors(bert_model, tokens)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    return (classifier.predict(vecs), entities)
