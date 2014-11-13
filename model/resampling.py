#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import numpy as np

df = pd.read_csv('data_preprocessed.csv', header=None, error_bad_lines=False,
                 names=['categoryid', 'categoryname', 'prodname', 'navigation',
                        'merchant', 'brand'])
df = df.dropna()
# bootstrap sampling
bag = []
grouped = df.groupby(by=['categoryid'])
for categoryid, group in grouped:
    group = group.reset_index(drop=True)
    sampled_group = np.random.choice(group.shape[0], size=200)
    bag.append(group.ix[sampled_group])
resampled_data = pd.concat(bag)
resampled_data.to_csv('data_resampled.csv', header=False, index=False,
                      encoding='utf8')
