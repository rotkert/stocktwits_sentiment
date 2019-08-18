import os
import json


def extract_twits(filename, symbols):
    extracted_twits = dict()
    for symbol in symbols:
        extracted_twits[symbol] = []

    with open(filename, encoding='utf-8') as file:
        line = file.readline()
        while line:
            json_line = json.loads(line)
            if not json_line:
                continue
            if 'symbols' in json_line['data'].keys():
                for elem in json_line['data']['symbols']:
                    for symbol, twits in extracted_twits.items():
                        if elem['symbol'].upper().startswith(symbol):
                            twits.append(line)
            line = file.readline()

    for symbol, twits in extracted_twits.items():
        save_extracted_twits(symbol, twits)

def save_extracted_twits(symbol, twits):
    filename = symbol.lower().replace('.', '_') + '_twits'
    with open(filename, 'a', encoding='utf-8') as file:
        for line in twits:
            file.writelines(line)


symbols = ['AAPL', 'BTC.X', 'ETH.X', 'NASDAQ', 'SPX']
for file in os.listdir():
    name, extension = os.path.splitext(file)
    if not extension and name.startswith('stocktwits_messages'):
        print(file + 'has_started')
       	extract_twits(file, symbols)
        print(file + 'has_ended')
