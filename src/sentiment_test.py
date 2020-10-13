"""

A test file for sentiment module testing

"""

from model_bert import BERTModel
from model_bayes_markov import NaiveBayesModel, MarkovModel
from training_data_source_csv import CSVFileTrainingDataSource


train_data = CSVFileTrainingDataSource("stocks_twitter.csv")
train_data.load()

models = [NaiveBayesModel(), MarkovModel(), BERTModel(set_size=100)]
# Note: BERTModel set size should be increased - it is small to allow for quick training, but will have bad performance

for model in models:
    model.build_from_data_source(train_data)

# Let the user test out the sentiment analysis
while True:
    sentence = input("Input sentence to analyze\n")
    labels = [model.classify(sentence) for model in models]
    print("       | Class | Std prob")
    print(f"Markov |   {labels[0][0]}   | {labels[0][1]}")
    print(f"NaiveB |   {labels[1][0]}   | {labels[1][1]}")
    print(f"BERT   |   {labels[2][0]}   | {labels[2][1]}")
