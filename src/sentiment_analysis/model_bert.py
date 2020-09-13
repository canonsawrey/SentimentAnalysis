# Libraries
import pandas as pd
import torch
import nltk
import transformers as ppb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# NOTE: may need to install transformers for running this for the first time

#Local
from data_source import DataSource

WEIGHTS = 'distilbert-base-uncased'
# Performance on whole 6920 sentence set is very similar, but takes rather longer
SET_SIZE = 2000


# Extract just the labels from the dataframe
def get_labels(df):
    return df[1]


# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)


# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe["sentence"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


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


# Takes a while, runs the model on all sentences.
# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:, 0, :].numpy()


# Various models!


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


class BERTModel:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.train_features = []
        self.test_features = []
        self.train_labels = []
        self.test_labels = []
        self.classifier: MLPClassifier = None  # TODO support other types of models

    # Builds a model_info object from a DataSource
    def build_from_data_source(self, data_source: DataSource):
        self.df = data_source.df_data()[:2000]
        print(self.df)
        # Create a BERT tokenizer and turn the dataset's sentences into BERT tokens.
        tokenizer = get_tokenizer()
        tokens = get_tokens(self.df, tokenizer)
        # Pad the tokens with zeros, so that the sentences are all the same length.
        padded_tokens = pad_tokens(tokens)
        # Call get_bert_sentence_vectors() to extract the vectors corresponding to the first token,
        # CLS, for each sentence. This vector is trained to represent the overall meaning of the sentence
        # for classification tasks as best as possible
        model = get_model()
        vectors = get_bert_sentence_vectors(model, padded_tokens)
        # Create training and test vectors
        self.train_features, self.test_features, self.train_labels, self.test_labels = \
            train_test_split(vectors, get_labels(self.df))
        self.train_classifier()

        print(f'BERT built from {len(data_source.list_data())} pieces of data')

    def train_classifier(self):
        self.classifier = train_classic_mlp_classifier(self.train_features, self.train_labels)

    def sentiment_classifier(self, bert_model, text):
        # TODO
        tokens = nltk.word_tokenize(text)
        vecs = get_bert_sentence_vectors(bert_model, tokens)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged)
        return self.classifier.predict(vecs), entities



# DEV # All below this line is experimental ######################

# Finds to "closest sentences"
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


def visualize_data(vecs, labels):
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(vecs)
    plt.scatter(pca_vecs[:, 0], pca_vecs[:, 1], c=labels)
    plt.show()
# visualize_data(vectors, get_labels(df))


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


