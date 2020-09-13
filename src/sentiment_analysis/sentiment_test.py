"""

A test file for sentiment module testing

"""

from model_bert import BERTModel
from model_bayes_markov import NaiveBayesMarkovModel
from data_source_csv import CSVFileDataSource


data_source = CSVFileDataSource("stocks_twitter.csv")
data_source.load()
nb_mm_model = NaiveBayesMarkovModel()
nb_mm_model.build_from_data_source(data_source)
bert_model = BERTModel()
bert_model.build_from_data_source(data_source)

# Let the user test out the sentiment analysis
while True:
    sentence = input("Input sentence to analyze\n")
    mm = nb_mm_model.markov_model_classify(sentence)
    nb = nb_mm_model.naive_bayes_classify(sentence)
    print("       | Class | Std prob")
    print(f"Markov |   {mm[0]}   | {mm[1]}")
    print(f"NaiveB |   {nb[0]}   | {nb[1]}")
