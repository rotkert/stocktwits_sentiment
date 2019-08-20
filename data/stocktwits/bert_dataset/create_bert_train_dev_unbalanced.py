import pandas as pd

class_size = 500000
train_size = 700000

data = pd.read_pickle('labeled_dataset.pickle')
label_encoded = data.replace({'sentiment_declared' : {'Bullish' : 1, 'Bearish' : 0}} )

pos = label_encoded[label_encoded.sentiment_declared == 1]
neg = label_encoded[label_encoded.sentiment_declared == 0]

pos_sampled = pos.sample(frac=1)[:class_size]
neg_sampled = neg.sample(frac=1)[:class_size]

balanced = pd.concat([pos_sampled, neg_sampled]).sample(frac=1).reset_index().drop('index', axis=1)

train = balanced[:train_size]
dev = balanced[train_size:]

train_bert = pd.DataFrame({
    'id' : range(len(train)),
    'label' : train.sentiment_declared,
    'alpha' : ['a'] * len(train),
    'text' : train.message
})

dev_bert = pd.DataFrame({
    'id' : range(len(dev)),
    'label' : dev.sentiment_declared,
    'alpha' : ['a'] * len(dev),
    'text' : dev.message
})

train_bert.to_csv('train.tsv', sep='\t', index=False, header=False)
dev_bert.to_csv('dev.tsv', sep='\t', index=False, header=False)




