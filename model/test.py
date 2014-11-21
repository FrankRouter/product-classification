#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import jieba
import pandas as pd
from sklearn.externals import joblib
from datetime import datetime

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

print('#' * 80)
print('%s\t%s' % (datetime.now(), 'Loading data...'))
jieba.load_userdict('productnames.dict')
category_dict = dict()
with open('../allcategories.csv') as f:
    for line in f:
        categoryid, categoryname = line.split(',')
        category_dict[categoryid.strip()] = categoryname.strip()

test = pd.read_csv('test.csv', names=['prodname', 'navigation', 'merchant',
                   'brand'], encoding='utf-8')

print('%s\t%s' % (datetime.now(), 'Preprocessing...'))
test.fillna('')
test['preprocessed'] = test[['prodname', 'navigation', 'merchant', 'brand']] \
    .apply(preprocess, axis=1)

print('%s\t%s' % (datetime.now(), 'Predicting...'))
tfidf = joblib.load('bin/tfidf')
clf = joblib.load('bin/classifier')
X = tfidf.transform(test['preprocessed'].values)
y_pred = clf.predict(X)
test['categoryid'] = y_pred
test['categoryname'] = test['categoryid'].map(category_dict)
print('%s\t%s' % (datetime.now(), 'Outputing...'))
test[['categoryid', 'categoryname', 'prodname', 'navigation', 'merchant',
      'brand']].to_csv('result.csv', encoding='utf-8', index=False)
