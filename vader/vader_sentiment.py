import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

symbols = ['aapl', 'btc', 'eth', 'spx']

for symbol in symbols:
    print(symbol)
    data = pd.read_pickle('../data/stocktwits/datasets_clean/' + symbol + '_twits_012018_072019_clean.pickle')
    sents = []
    for m in data.message_clean.values:
        score = analyser.polarity_scores(m)
        sent = [m, score['compound']]
        sents.append(sent)
    sents_df = pd.DataFrame(sents, columns=['message', 'score'])
    sents_df.to_pickle('../data/stocktwits/vader_sentiment/vader_' + symbol + '_sentiment.pickle')