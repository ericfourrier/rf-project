#!/usr/bin/env python
# -*- coding: utf-8 -*-u

"""
Purpose : Get intraday stock data
http://www.google.com/finance/getprices?i=60&p=10d&f=d,o,h,l,c,v&df=cpct&q=IBM
"""

from math import ceil

import requests
import pandas as pd
import datetime as dt
import numpy as np

class GoogleFinanceStock(object):
    root_url = "http://www.google.com/finance/getprices"

    def __init__(self, ticker, period=60, window=1, col="d,o,h,l,c,v", df='cpct'):
        self.ticker = ticker
        self.period = period
        self.window = window
        self.col = col
        self.df = df
        self._meta = {}

    def get_data(self, convert_to_datetime=True):
        payload = {'q': self.ticker, 'p': '{}d'.format(self.window), 'f': self.col,
                   'i': self.period, 'df': self.df}
        # split into lines
        res = requests.get(
            self.root_url, params=payload).text.strip().split('\n')
        self._meta = {l.split('=')[0]: l.split('=')[1] for l in res[1:7]}
        self._data = pd.DataFrame([l.split(',') for l in res[7:]], columns=self._meta[
                                  'COLUMNS'].split(','))
        self._meta['START_TIME_RAW'] = int(self._data.loc[0, 'DATE'][1:])
        self._meta['START_TIME'] = dt.datetime.fromtimestamp(self._meta['START_TIME_RAW'])
        index_day = self._data.loc[:,'DATE'].str.startswith('a')
        # create auxiliary columns for day change
        self._data['DATE_D'] = np.nan
        self._data.loc[index_day,'DATE_D'] = self._data.loc[index_day, 'DATE'].str.replace('a','').map(lambda x :int(x))
        self._data = self._data.fillna(method='ffill')
        # replace start_time for by 0 for better
        self._data.loc[index_day, 'DATE'] = 0
        self._data = self._data.astype(float) # convert everything to float
        if convert_to_datetime:
            self._data['DATE'] = (int(self._meta['INTERVAL']) * self._data['DATE']) + self._data['DATE_D'] - int(360*60)
            self._data['DATE'] = self._data['DATE'].map(lambda x: dt.datetime.fromtimestamp(x))
        return self._data, self._meta



if __name__ == "__main__":
    g1 = GoogleFinanceStock('GOOG') # 1 days history
    res = g1.get_data()
    print(res[0])
    g1.window = 10 # 10 days history
    res2 = g1.get_data()
    print(res2[0])
