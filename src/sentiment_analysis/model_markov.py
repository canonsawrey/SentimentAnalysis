"""

Naive Bayes and Markov Models to do basic sentiment analysis.

"""

# Libraries
import math
# NLTK is a Python toolkit for natural language processing (NLP)
# import nltk
# nltk.download('punkt')  # Need to get the dataset NLKT relies on before running
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

# Locals
from data_source import DataSource

# Constants
# Probability of a unigram or bigram that hasn't been seen - think  "practically a rounding error"
OUT_OF_VOCAB_PROB = 0.0000000001


# Get words from a line in a consistent way. Uses NLTK, standardizes to lowercase
def tokenize(sentence: str) -> [str]:
    return [t.lower() for t in word_tokenize(sentence)]


"""

NaiveBayesMarkovModel - tracks data about training data. Allows for estimation of sentiment using NB and MM
    word_counts:  5 dicts from string to int, one per sentiment, in a list
    bigram_counts: same
    sentiment_counts:  list containing counts of the sentences with different sentiments
    total_words:  list of counts of words for each sentiment
    bigram_denoms:  Separate counts of how often a word starts a bigram, again one per sentiment.
    total_bigrams: counts of total bigrams for each sentiment level

"""


class NaiveBayesMarkovModel:
    def __init__(self):
        self.word_counts = [{}, {}, {}, {}, {}]
        self.bigram_counts = [{}, {}, {}, {}, {}]
        self.sentiment_counts = [0, 0, 0, 0, 0]
        self.total_words = [0, 0, 0, 0, 0]
        self.bigram_denoms = [{}, {}, {}, {}, {}]
        self.total_bigrams = [0, 0, 0, 0, 0]
        self.total_examples = 0

    # update_word_counts
    # assume space-delimited words/tokens.
    #
    # To "tokenize" the sentence we'll make use of NLTK, a widely-used Python natural language
    # processing (NLP) library.  This will handle otherwise onerous tasks like separating periods
    # from their attached words.  (Unless the periods are decimal points ... it's more complex
    # than you might think.)  The result of tokenization is a list of individual strings that are
    # words or their equivalent.
    #
    # Note that sentiment is an integer, not a string, matching the data format
    def update_word_counts(self, sentence, sentiment):
        # Get the relevant dicts for the sentiment
        s_word_counts = self.word_counts[sentiment]
        s_bigram_counts = self.bigram_counts[sentiment]
        s_bigram_denoms = self.bigram_denoms[sentiment]
        tokens = tokenize(sentence)
        for token in tokens:
            self.total_words[sentiment] += 1
            s_word_counts[token] = s_word_counts.get(token, 0) + 1
        my_bigrams = bigrams(tokens)
        for bigram in my_bigrams:
            s_bigram_counts[bigram] = s_bigram_counts.get(bigram, 0) + 1
            s_bigram_denoms[bigram[0]] = s_bigram_denoms.get(bigram[0], 0) + 1
            self.total_bigrams[sentiment] += 1

    # Builds a model_info object from a DataSource
    def build_from_data_source(self, data_source: DataSource):
        for datum in data_source.list_data():
            try:
                sentiment = datum.sentiment
                sentence = datum.sentence
                self.sentiment_counts[sentiment] += 1
                self.total_examples += 1
                self.update_word_counts(sentence, sentiment)
            except ValueError:
                # Skip bad inputs
                continue
        print(f'Model built from {len(data_source.list_data())} pieces of data')

    # Returns a number indicating sentiment and a log probability of that sentiment (two comma-separated return values).
    def naive_bayes_classify(self, sentence):
        probs = []
        words = tokenize(sentence)
        for i in range(0, 2):
            # Set initial value to prior
            prob = math.log(self.sentiment_counts[i])
            prob -= math.log(self.total_examples)
            for word in words:
                if word in self.word_counts[i]:
                    prob += math.log(self.word_counts[i].get(word))
                    prob -= math.log(self.total_words[i])
                else:
                    prob += math.log(OUT_OF_VOCAB_PROB)
            probs.append(prob)

        return probs.index(max(probs)), max(probs)

    # Like naive Bayes, but now use a bigram model
    def markov_model_classify(self, sentence):
        probs = []
        words = tokenize(sentence)
        for i in range(0, 2):
            # Set inital value to prior
            prob = math.log(self.sentiment_counts[i])
            prob -= math.log(self.total_examples)
            # Handle first word special case
            prev = words[0]
            if prev in self.word_counts[i]:
                prob += math.log(self.word_counts[i].get(prev))
                prob -= math.log(self.total_words[i])
            else:
                prob += math.log(OUT_OF_VOCAB_PROB)
            # Iterate over rest as bigrams
            for word in words[1:]:
                bigram = (prev, word)
                if bigram in self.bigram_counts[i]:
                    prob += math.log(self.bigram_counts[i].get(bigram))
                    prob -= math.log(self.bigram_denoms[i].get(prev))
                else:
                    prob += math.log(OUT_OF_VOCAB_PROB)
                prev = word
            probs.append(prob)
        return probs.index(max(probs)), max(probs)
