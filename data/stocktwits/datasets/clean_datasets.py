import os
import pandas as pd
import preprocessor as p
import re
import html

for file in os.listdir():
    name, ext = os.path.splitext(file)
    if ext == '.pickle':
        print(file)
        twits = pd.read_pickle(file)
        message_arr = twits.message.values
        message_clean_arr = []
        for i, message in enumerate(message_arr):
            if i % 10000 == 0:
                print(i)
            message_clean = message.replace('(MarketWatch.com%20-%20Software%20Industry%20News)', '')
            message_clean = message_clean.replace('(MarketWatch.com%20-%20Financial%20Services%20Industry%20News)', '')
            message_clean = message_clean.replace('(MarketWatch.com%20-%20MarketPulse)', '')
            message_clean = html.unescape(message_clean)
            message_clean = re.sub('\$\w+\.?\w+\s?', '', message_clean)
            message_clean = p.clean(message_clean)
            message_clean_arr.append(message_clean)
        twits['message_clean'] = message_clean_arr
        twits = twits.reset_index()
        twits.to_pickle(os.path.join('../datasets_clean', name + '_clean.pickle'))