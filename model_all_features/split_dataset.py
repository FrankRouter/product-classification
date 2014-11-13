#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np

df = pd.read_csv('data_resampled.csv', header=None, error_bad_lines=False,
                 names=['categoryid', 'categoryname', 'info'])
r = np.random.random_sample((len(df)))
train = df.ix[r >= 0.25]
test = df.ix[r < 0.25]
train.to_csv('train.csv', index=False, encoding='utf-8')
test.to_csv('test.csv', index=False, encoding='utf-8')
