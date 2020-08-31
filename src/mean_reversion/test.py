
from mean_reversion.watching import show

for trade in show:
    trade.analyze()
    # trade.plot()
    trade.test(1000, .3, .6, .75)

