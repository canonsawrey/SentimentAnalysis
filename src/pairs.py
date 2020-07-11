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
        self.sec1_data['Price_z'] = util.zscore(self.sec1_data['Price'])
        self.sec2_data = data.daily_df(sec2)
        self.sec2_data = self.sec2_data.loc[self.sec2_data.index > cutoff_date]
        self.sec2_data['Price_z'] = util.zscore(self.sec2_data['Price'])
        self.name = "PMV: " + sec1 + "-" + sec2

    def description(self) -> str:
        return "Pairs mean reversion pair between " + self.sec1 + " and " + self.sec2 + " over " + str(self.size) + \
               " days"

    def plot(self):
        print(self.sec1_data)

        merge_df = self.sec1_data.merge(self.sec2_data, on='Date', suffixes=['_'+self.sec1, '_'+self.sec2])
        print(merge_df)
        # fig, axes = plt.subplots(nrows=2, ncols=1)
        merge_df.plot(kind='line', y=['Price_z_'+self.sec1, 'Price_z_'+self.sec2])

        merge_df[self.sec1 + '/' + self.sec2] = merge_df['Price_'+self.sec1] / merge_df['Price_'+self.sec2]
        merge_df[self.sec1 + '/' + self.sec2 + "_z"] = util.zscore(merge_df[self.sec1 + '/' + self.sec2])
        merge_df[self.sec1 + '/' + self.sec2 + "_z_rolling5"] = \
            merge_df[self.sec1 + '/' + self.sec2 + "_z"].rolling(5).mean()
        merge_df[self.sec1 + '/' + self.sec2 + "_z_rolling60"] = \
            merge_df[self.sec1 + '/' + self.sec2 + "_z"].rolling(60).mean()
        merge_df.plot(kind='line', y=[self.sec1 + '/' + self.sec2 + '_z',
                                                  self.sec1 + '/' + self.sec2 + "_z_rolling5",
                                                  self.sec1 + '/' + self.sec2 + "_z_rolling60"])
        plt.axhline(y=merge_df[self.sec1 + '/' + self.sec2 + '_z'].mean(), linestyle='dashed')
        # plt.axhline(1.0, color='red')
        # plt.axhline(-1.0, color='green')
        plt.show()
