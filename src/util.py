"""

Defines common utility functions for use in SentimentAnalysis project

"""

import math


# Turns cumulative log probabilities into relative probs
def standardize(probs):
    exps = [math.exp(x) for x in probs]
    total = sum(exps)
    return [x / total for x in exps]
