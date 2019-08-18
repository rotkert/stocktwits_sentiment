import time
import json
import numpy as np
import pandas as pd


def get_attribute_value(data, name):
	if (name) in data.keys():
		return data[name]
	else:
		return None
agg = 0
start = time.time()
for filename in ['spx_twits_012018_052019_all.json', 'btc_x_twits_012018_052019_all.json', 'eth_x_twits_012018_052019_all.json', 'nasdaq_twits_012018_052019_all.json', 'aapl_twits_012018_052019_all.json']:
	print(filename)
	symbol = filename.split('_')[0].upper()
	if symbol == 'ETH':
		symbol = 'ETH-X'
	if symbol == 'BTC':
		symbol = 'BTC-X'
	
	with open(filename, encoding='utf-8') as file:
		tweets = []
		line = file.readline()
		i = 0
		while line:
			if i % 10000 == 0:
				print(i)
			i += 1

			json_line = json.loads(line)
			tweet_data = json_line['data']
			if not json_line or json_line['object'] != 'Message' or json_line['action'] != 'create' or 'body' not in tweet_data or 'created_at' not in tweet_data:
				line = file.readline()
				continue

			message = tweet_data['body']
			created_at = tweet_data['created_at']
			user_official = None
			user_followers = None
			user_ideas = None
			user_likes = None
			user_data = tweet_data['user'] if 'user' in tweet_data.keys() else None
			if user_data is not None:
				user_official = user_data['official'] if 'official' in user_data.keys() else None
				user_followers = user_data['followers'] if 'followers' in user_data.keys() else None
				user_ideas = user_data['ideas'] if 'ideas' in user_data.keys() else None
				user_likes = user_data['like_count'] if 'like_count' in user_data.keys() else None
			
			price = None
			price_data = tweet_data['prices'] if 'prices' in tweet_data.keys() else None
			if price_data is not None and len(price_data) > 0:
				for price_elem in price_data:
					price_symbol = get_attribute_value(price_elem, 'symbol')
					if price_symbol is not None and price_symbol == symbol:
						price = get_attribute_value(price_elem, 'price')
						break
			
			sentiment_declared = None
			entitites_data = get_attribute_value(tweet_data, 'entities')
			if entitites_data is not None:
				sentiment_data = get_attribute_value(entitites_data, 'sentiment')
				if sentiment_data is not None:
					sentiment_declared = get_attribute_value(sentiment_data, 'basic')

			sentiment_score = None
			sentiment_score_data = get_attribute_value(tweet_data, 'sentiment')
			if sentiment_score_data is not None:
				sentiment_score = get_attribute_value(sentiment_score_data, 'sentiment_score')
			added_time = get_attribute_value(json_line, 'time')
			tweet_row = [created_at, message, user_official, user_followers, user_ideas, user_likes, sentiment_declared, sentiment_score, price, added_time]
			tweets.append(tweet_row)

			line = file.readline()
		tweets = pd.DataFrame(tweets, columns=['created_at', 'message', 'user_official', 'user_followers', 'user_ideas', 'user_likes', 'sentiment_declared', 'sentiment_score', 'price', 'time'])
		tweets.index = pd.to_datetime(tweets['created_at'])
		tweets = tweets.drop('created_at', axis=1)
		tweets.to_pickle(filename + '.pickle')
end = time.time()
print(end - start)


