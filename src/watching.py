# Canon Sawrey
# A repository of potential trades the script is watching

from pairs import PairsMeanReversion

trades = [
    PairsMeanReversion("ADBE", "MSFT", 2000),
    #PairsMeanReversion("AAPL", "MSFT"),
    #PairsMeanReversion("SNE", "MSFT"),
]


# Trades that the script will plot for us
show = [
    PairsMeanReversion("ADBE", "MSFT", 2000),
]

