# Utility functions

import numpy as np


def zscore(series):
    return (series - series.mean()) / np.std(series)
