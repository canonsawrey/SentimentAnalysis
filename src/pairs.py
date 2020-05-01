# A file for containing mean reversions object

from strategy import Strategy
import data
#import statsmodels
#from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt


class PairsMeanReversion(Strategy):
    def __init__(self, sec1, sec2, size):
        self.sec1 = sec1
        self.sec2 = sec2
        self.size = size
        self.deltas = {}
        self.sec1_data = {}
        self.sec2_data = {}
        self.name = "PMV: " + sec1 + "-" + sec2

    def generate_data(self):
        self.deltas = {}
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
                self.deltas[entry] = self.sec1_data[entry] - self.sec2_data[entry]

    def pvalue(self) -> float:
        #TODO Calculate the pvalue

        #TODO Check for cointegration of the 2

        return 1.0

    def plot(self):
        self.generate_data()
        dates, s1, s2, d, ratio = [], [], [], [], []
        for entry in self.deltas.keys():
            dates.append(entry)
            d.append(self.deltas[entry])
            s1.append(self.sec1_data[entry])
            s2.append(self.sec2_data[entry])
            ratio.append(self.sec1_data[entry] / self.sec2_data[entry])
        plt.plot(
            # dates, d,
            # dates, s1,
            # dates, s2,
            dates, ratio,
        )
        plt.show()
