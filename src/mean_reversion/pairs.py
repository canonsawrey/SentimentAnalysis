# A file for containing mean reversions object

from mean_reversion.strategy import Strategy
import matplotlib.pyplot as plt
from mean_reversion import util, data
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
        self.merge_df = self.sec1_data.merge(self.sec2_data, on='Date', suffixes=['_'+self.sec1, '_'+self.sec2])

    def description(self) -> str:
        return "Pairs mean reversion pair between " + self.sec1 + " and " + self.sec2 + " over " + str(self.size) + \
               " days"

    def analyze(self):
        self.merge_df[self.name_ratio()] = self.merge_df['Price_'+self.sec1] / self.merge_df['Price_'+self.sec2]
        self.merge_df[self.name_ratio_rolling(5)] = \
            self.merge_df[self.name_ratio()].rolling(5).mean()
        self.merge_df[self.name_ratio_rolling(60)] = \
            self.merge_df[self.name_ratio()].rolling(60).mean()

        self.merge_df['std_60'] = self.merge_df[self.name_ratio()].rolling(window=60, center=False).std()
        self.merge_df['z_ratio_60_5'] = \
            (self.merge_df[self.name_ratio_rolling(5)] - self.merge_df[self.name_ratio_rolling(60)]) / self.merge_df['std_60']
        """self.merge_df['buy'] = self.merge_df['z_ratio_60_5']
        self.merge_df['sell'] = self.merge_df['z_ratio_60_5']

        self.merge_df['buy'][self.merge_df['z_ratio_60_5'] > -1] = None
        self.merge_df['sell'][self.merge_df['z_ratio_60_5'] < 1] = None """

    def plot(self):
        self.merge_df.plot(kind='line', y=[self.name_ratio(),
                                           self.name_ratio_rolling(5),
                                           self.name_ratio_rolling(60)])
        plt.show()

        '''self.merge_df.plot(kind='line', y=['z_ratio_60_5'])
        self.merge_df['buy'].plot(color='g', linestyle=None, marker='^')
        self.merge_df['sell'].plot(color='r', linestyle=None, marker='^')
        plt.axhline(0.0, color='black', linestyle='--')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.axhline(-1.0, color='green', linestyle='--')
        plt.legend(['Ratio', 'Buy', 'Sell'])
        plt.show() '''

    # Tests the model on a subset of the data
    # 'Capital' is a measure of the $ we will invest. So for a pairs trade, if that amount is $1000, we will buy $500
    # and short $500 on a 1.00 investment
    # Sd_1 is the percentage we will invest at SD_1
    def test(self, capital, sd_1, sd_2, sd_3):
        state = 0 # Defines the state of the model. Integer representing the model threshold we are currently at
        stocks_sec1 = 0
        stocks_sec2 = 0
        signal = self.merge_df['z_ratio_60_5']
        print(len(signal))


    def name_ratio(self) -> str:
        return self.sec1 + '/' + self.sec2

    def name_ratio_z(self) -> str:
        return self.sec1 + '/' + self.sec2 + "_z"

    def name_ratio_rolling(self, time_frame):
        return self.sec1 + '/' + self.sec2 + "_z_rolling" + str(time_frame)




