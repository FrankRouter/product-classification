#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import jieba
import pandas as pd
from sklearn.externals import joblib
from datetime import datetime
import numpy as np
import json

remove_chars = u'''`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?''' + \
    u'''·！@#￥%……&*（）——【】、『』|；‘’：“”，。《》？'''
trans_table = dict([(ord(c), None) for c in remove_chars])


def preprocess(cells):
    # remove punctuation
    # fillna is not reliable, unicode(cell) is necessary
    cleaned = [unicode(cell).translate(trans_table) for cell in cells.values]
    # chinese segment
    segmented = [u' '.join(jieba.cut(cell)) for cell in cleaned]
    return u' '.join(segmented)


def filter(row):
    if row['proba'] >= boundary[row['categoryid']]:
        return row['categoryid']
    else:
        return None

print('#' * 80)
print('%s\t%s' % (datetime.now(), 'Loading data...'))
jieba.load_userdict('productnames.dict')
category_dict = dict()
with open('../allcategories.csv') as f:
    for line in f:
        categoryid, categoryname = line.split(',')
        category_dict[categoryid.strip()] = categoryname.strip()
with open('boundary.json') as f:
    boundary = json.load(f)
test = pd.read_csv('test.csv', names=['prodname', 'navigation', 'merchant',
                   'brand'], encoding='utf-8')

print('%s\t%s' % (datetime.now(), 'Preprocessing...'))
test.fillna('')
test['preprocessed'] = test[['prodname', 'navigation', 'merchant', 'brand']] \
    .apply(preprocess, axis=1)

print('%s\t%s' % (datetime.now(), 'Predicting...'))
tfidf = joblib.load('bin/tfidf')
clf = joblib.load('bin/classifier')
test['categoryid'] = test['brand'].map(lambda x: '')
test['proba'] = test['brand'].map(lambda x: .0)
step = 20000
for idx in np.arange(0, test.shape[0], step):
    print('\t%s\trecords' % idx)
    # X is a dense matrix, so it consumes lots of memory
    X = tfidf.transform(test['preprocessed'].iloc[idx: idx + step].values)
    jll = clf.predict_proba(X)  # joint likelihood
    y_pred = clf.classes_[np.nanargmax(jll, axis=1)]
    max_proba = np.nanmax(jll, axis=1)
    test['categoryid'].iloc[idx: idx + step] = y_pred
    test['proba'].iloc[idx: idx + step] = max_proba
# filter by decision boundary
test['categoryid'] = test[['categoryid', 'proba']].apply(filter, axis=1)
test['categoryname'] = test['categoryid'].map(category_dict)

print('%s\t%s' % (datetime.now(), 'Outputing...'))
test[['categoryid', 'categoryname', 'prodname', 'navigation', 'merchant',
      'brand']].to_csv('result.csv', encoding='utf-8', index=False)
