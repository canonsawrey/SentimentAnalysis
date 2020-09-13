"""

A test file for sentiment module testing

"""

from model_markov import NaiveBayesMarkovModel
from data_source_csv import CSVFileDataSource


data_source = CSVFileDataSource("stocks_twitter.csv")
data_source.load()
model = NaiveBayesMarkovModel()
model.build_from_data_source(data_source)

# Let the user test out the sentiment analysis
while True:
    sentence = input("Input sentence to analyze\n")
    mm = model.markov_model_classify(sentence)
    nb = model.naive_bayes_classify(sentence)
    print("       | Class | Log prob")
    print(f"Markov |   {mm[0]}   | {mm[1]}")
    print(f"NaiveB |   {nb[0]}   | {nb[1]}")
