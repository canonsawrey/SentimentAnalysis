# Canon Sawrey
# A repository of potential trades the script is watching

from mean_reversion.pairs import PairsMeanReversion

# Trades that I've backtested and validated
trades = [
    PairsMeanReversion("ADBE", "MSFT", 20),
    #PairsMeanReversion("AAPL", "MSFT"),
    #PairsMeanReversion("SNE", "MSFT"),
]


# Trades that the script will plot for us
show = [
    # PairsMeanReversion("ADBE", "MSFT", 5000),
    PairsMeanReversion("ADBE", "MSFT", 1000),
    # PairsMeanReversion("ADBE", "MSFT", 20),
]

