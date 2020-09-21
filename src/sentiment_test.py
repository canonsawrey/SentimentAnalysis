"""

A test file for sentiment module testing

"""

# from model_bert import BERTModel
from model_bayes_markov import NaiveBayesMarkovModel
from training_data_source_csv import CSVFileTrainingDataSource


train_data = CSVFileTrainingDataSource("stocks_twitter.csv")
train_data.load()
nb_mm_model = NaiveBayesMarkovModel()
nb_mm_model.build_from_data_source(train_data)

# Comment out BERT model while updating
# bert_model = BERTModel()
# bert_model.build_from_data_source(data_source)
# model = bert_model.SentimentClassifier(2)

# Let the user test out the sentiment analysis
while True:
    sentence = input("Input sentence to analyze\n")
    mm = nb_mm_model.markov_model_classify(sentence)
    nb = nb_mm_model.naive_bayes_classify(sentence)
    # brt
    print("       | Class | Std prob")
    print(f"Markov |   {mm[0]}   | {mm[1]}")
    print(f"NaiveB |   {nb[0]}   | {nb[1]}")
    # print(f"BERT   |   {brt[0]}   | {brt[1]}")
