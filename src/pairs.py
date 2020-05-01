# A file for containing mean reversions object

from strategy import Strategy
import data


class PairsMeanReversion(Strategy):
    def __init__(self, sec1, sec2, size):
        self.sec1 = sec1
        self.sec2 = sec2
        self.size = size
        self.name = "PMV: " + sec1 + "-" + sec2

    def pvalue(self) -> float:
        sec1_data = data.data(self.size, self.sec1)
        sec2_data = data.data(self.size, self.sec2)
