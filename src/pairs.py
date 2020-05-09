# A file for containing mean reversions object

from strategy import Strategy
import data
#import statsmodels
#from statsmodels.tsa.stattools import coint
import util
import matplotlib.pyplot as plt
import pandas as pd


class PairsMeanReversion(Strategy):
    def __init__(self, sec1, sec2, size):
        self.sec1 = sec1
        self.sec2 = sec2
        self.size = size
        self.ratios = {}
        self.sec1_data = {}
        self.sec2_data = {}
        self.name = "PMV: " + sec1 + "-" + sec2

    def generate_data(self):
        self.ratios = {}
        self.sec1_data = {}
        self.sec2_data = {}

        sec1_ret = data.data(self.size, self.sec1)
        sec2_ret = data.data(self.size, self.sec2)
        for i in range(0, len(sec1_ret[0])):
            self.sec1_data[sec1_ret[0][i]] = sec1_ret[1][i]
        for i in range(0, len(sec2_ret[0])):
            self.sec2_data[sec2_ret[0][i]] = sec2_ret[1][i]

        for entry in self.sec1_data.keys():
            if entry in self.sec2_data:
                self.ratios[entry] = self.sec1_data[entry] / self.sec2_data[entry]

    def pvalue(self) -> float:
        if len(self.sec1_data) == 0:
            self.generate_data()

        return 1.0

    def description(self) -> str:
        return "Pairs mean reversion pair between " + self.sec1 + " and " + self.sec2 + " over " + str(self.size) + \
               " days"

    def plot(self):
        self.generate_data()
        s1, s2, ratio = pd.DataFrame(data={}), pd.DataFrame(data={}), pd.DataFrame(data={})
        dates = []

        for entry in self.ratios.keys():
            dates.append(entry)
            s1.append(self.sec1_data[entry])
            s2.append(self.sec2_data[entry])
            ratio.append(self.sec1_data[entry] / self.sec2_data[entry])

        ratios_mavg_short = ratio.rolling(window=self.size / 10, center=False).mean()
        ratios_mavg_full = ratio.rolling(window=self.size, center=False).mean()

        plt.plot(dates, ratio)
        plt.plot(dates, ratios_mavg_full)
        plt.plot(dates, ratios_mavg_short)
        plt.show()
