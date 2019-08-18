import os
import json
import pandas as pd
import html
import re
import preprocessor as p

def get_attribute_value(data, name):
	if (name) in data.keys():
		return data[name]
	else:
		return None

tweets = []
for file in ['stocktwits_messages_2019_02', 'stocktwits_messages_2019_03']:
	name, extension = os.path.splitext(file)
	if not extension and name.startswith('stocktwits_messages'):
		print(file + ' has_started')
		with open(file, encoding='utf-8') as f:
			line = f.readline()
			i = 0
			while line:
				if i % 10000 == 0:
					print(i)
				i += 1
				
				json_line = json.loads(line)
				tweet_data = json_line['data']
				if not json_line or json_line['object'] != 'Message' or json_line['action'] != 'create' or 'body' not in tweet_data:
					line = f.readline()
					continue
				message = tweet_data['body']
				
				sentiment_declared = None
				entitites_data = get_attribute_value(tweet_data, 'entities')
				if entitites_data is not None:
					sentiment_data = get_attribute_value(entitites_data, 'sentiment')
					if sentiment_data is not None:
						sentiment_declared = get_attribute_value(sentiment_data, 'basic')
				
				if sentiment_declared is None:
					line = f.readline()
					continue
					
				message_clean = message.replace('(MarketWatch.com%20-%20Software%20Industry%20News)', '')
				message_clean = message_clean.replace('(MarketWatch.com%20-%20Financial%20Services%20Industry%20News)', '')
				message_clean = message_clean.replace('(MarketWatch.com%20-%20MarketPulse)', '')
				message_clean = html.unescape(message_clean)
				message_clean = re.sub('\$\w+\.?\w+\s?', '', message_clean)
				message_clean = p.clean(message_clean)
				
				if not message_clean:
					line = f.readline()
					continue
					
				tweet_row = [message_clean, sentiment_declared]
				tweets.append(tweet_row)
				line = f.readline()
tweets = pd.DataFrame(tweets, columns=['message', 'sentiment_declared'])
tweets.to_pickle('labeled_dataset.pickle')
	