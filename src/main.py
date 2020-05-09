# Script which accomplishes the following
#     - Checks 'watching' strategies for potential large gains
#     - Checks active portfolio for sell/divest signals
from watching import trades
import matplotlib.pyplot as plt
import alert
from datetime import date

debug = True
strat_dict = {}

best_strat = trades[0]

for strat in trades:
    strat_dict[strat.name] = strat.pvalue()
    if strat.pvalue() > best_strat.pvalue():
        best_strat = strat

message = "Best strategy for " + date.today().strftime("%d/%m/%Y") + ":\n    " + best_strat.description() + \
          "\n    pvalue: " + str(best_strat.pvalue())

if not debug:
    alert.send_message(message)
else:
    print(message)
    plt.bar(range(len(strat_dict)), list(strat_dict.values()), align='center')
    plt.xticks(range(len(strat_dict)), list(strat_dict.keys()))
    plt.show()
