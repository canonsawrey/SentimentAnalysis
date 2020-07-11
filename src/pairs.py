# A file for containing mean reversions object

from strategy import Strategy
import data
import matplotlib.pyplot as plt
import util
import datetime


class PairsMeanReversion(Strategy):
    def __init__(self, sec1, sec2, size):
        self.sec1 = sec1
        self.sec2 = sec2
        self.size = size
        cutoff_date = datetime.datetime.today() - datetime.timedelta(days=size)
        self.sec1_data = data.daily_df(sec1)
        self.sec1_data = self.sec1_data.loc[self.sec1_data.index > cutoff_date]
        self.sec1_data['Price'] = util.zscore(self.sec1_data['Price'])
        self.sec2_data = data.daily_df(sec2)
        self.sec2_data = self.sec2_data.loc[self.sec2_data.index > cutoff_date]
        self.sec2_data['Price'] = util.zscore(self.sec2_data['Price'])
        self.name = "PMV: " + sec1 + "-" + sec2

    def description(self) -> str:
        return "Pairs mean reversion pair between " + self.sec1 + " and " + self.sec2 + " over " + str(self.size) + \
               " days"

    def plot(self):
        print(self.sec1_data)

        merge_df = self.sec1_data.merge(self.sec2_data, on='Date', suffixes=['_'+self.sec1, '_'+self.sec2])
        print(merge_df)
        merge_df.plot(kind='line', y=['Price_'+self.sec1, 'Price_'+self.sec2])
        plt.show()
