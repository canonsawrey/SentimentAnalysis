# Sentiment Analysis
### A sentiment analysis API, designed to be
* Extensible 
* Allow for the consumption of data from many different kinds of sources
* Provide estimates from multiple kinds of models
### General info
* Supports Naive Bayes, Markov Chain, BERT models
* Sentiment is classified as either positive (1) or negative (0)
* Model probability of sentiment estimate is provided as well
### Runbook
* Run `python3 sentiment_test.py` in `src/sentiment_analysis` for interactive console analysis
* Current implementation relies on a CSV of sample Twitter data
* To modify and hook up a new data source, implement the `DataSource` interface
* See `sentiment_test.py` for an example of how to build models from the `DataSource`
 