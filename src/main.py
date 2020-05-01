# Script which accomplishes the following
#     - Checks 'watching' strategies for potential large gains
#     - Checks active portfolio for sell/divest signals

from watching import trades
import matplotlib.pyplot as plt

strat_dict = {}
for strat in trades:
    strat_dict[strat.name] = strat.pvalue()

plt.bar(range(len(strat_dict)), list(strat_dict.values()), align='center')
plt.xticks(range(len(strat_dict)), list(strat_dict.keys()))
plt.show()
