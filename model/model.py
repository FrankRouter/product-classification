#! /usr/bin/env python
# -*- coding: utf-8 -*-
# train two naive bayes model based on different feature sets
# then ensemble them

from __future__ import print_function
from __future__ import division

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
from collections import defaultdict
import codecs
import os
from datetime import datetime

# vectorizer
vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2),
                             min_df=5, max_df=0.5)
train = pd.read_csv('train.csv', dtype=object)
test = pd.read_csv('test.csv', dtype=object)
corpus = train['prodname'] + train['navigation'] \
    + train['merchant'] + train['brand']
X = vectorizer.fit_transform(corpus.values)
y = train['categoryid'].values

# train
categoryid_set = set(train['categoryid'].values)
print('*' * 80)
print('Training: ')
clf = MultinomialNB()
chunk = 50000
m = X.shape[0]
if m < chunk:
    clf.fit(X, y)
else:
    for i, idx in enumerate(np.split(np.arange(m), xrange(chunk, m, chunk))):
        print('\t%s\tTraining %d chunk' % (datetime.now(), (i + 1)))
        clf.partial_fit(X[idx], y[idx], classes=list(categoryid_set))

# test
print('*' * 80)
print('Testing: ')
X_test = vectorizer.transform(test['prodname'] + test['navigation'] +
                              test['merchant'] + test['brand'])
y_true = test['categoryid'].values
jll = clf.predict_proba(X_test)  # joint likelihood
y_pred = clf.classes_[np.argmax(jll, axis=1)]
max_proba = np.amax(jll, axis=1)


# trade off between acurry and recall
# search best decision boundry for each category
def search():
    print('*' * 80)
    print('Searching: ')
    boundary_of_category = dict()
    max_p_category = np.amax(jll, axis=0)  # max probability in each category
    min_p_category = np.amin(jll, axis=0)  # min probability in each category
    for categoryid in categoryid_set:
        print('\t%s\tSearching in %s' % (datetime.now(), categoryid))
        idx = np.where(clf.classes_ == categoryid)
        tp = (y_true == categoryid) & (y_pred == categoryid)
        fp = (y_true != categoryid) & (y_pred == categoryid)
        fn = (y_true == categoryid) & (y_pred != categoryid)
        proba_tp = np.sort(max_proba[tp])
        proba_fp = np.sort(max_proba[fp])
        proba_fn = np.sort(max_proba[fn])
        threshold = np.linspace(min_p_category[idx], max_p_category[idx], 100)
        tp_num = proba_tp.shape[0] - np.searchsorted(proba_tp, threshold)
        fp_num = proba_fp.shape[0] - np.searchsorted(proba_fp, threshold)
        fn_num = proba_fn.shape[0] + np.searchsorted(proba_tp, threshold)
        accuracy = np.true_divide(tp_num, (tp_num + fp_num))
        recall = np.true_divide(tp_num, (tp_num + fn_num))
        f1 = np.true_divide(2 * accuracy * recall, (accuracy + recall))
        idx_max_f1 = np.argmax(f1)
        boundary_of_category[categoryid] = threshold[idx_max_f1]
        y_pred[(max_proba < threshold[idx_max_f1])
               & (y_pred == categoryid)] = None
    with codecs.open('boundary.json', encoding='utf-8', mode='w') as f:
        json.dump(obj=boundary_of_category, fp=f, ensure_ascii=False,
                  encoding='utf-8', indent=4, separators=(',', ': '))

if os.environ.get('search'):
    search()

with open('report.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_pred), file=f)

# output model in human readable format
wordsdict = defaultdict(dict)
words = vectorizer.get_feature_names()
for i in xrange(len(clf.feature_count_)):
    weights = clf.feature_count_[i]
    class_id = clf.classes_[i]
    for j in xrange(len(weights)):
        if weights[j] > 0:
            word = words[j]
            wordsdict[word][class_id] = weights[j]
with codecs.open('weights_of_words.json', encoding='utf-8', mode='w') as f:
    json.dump(obj=wordsdict, fp=f, ensure_ascii=False, encoding='utf-8',
              indent=4, separators=(',', ': '))

classesdict = defaultdict(list)
for i in xrange(len(clf.feature_count_)):
    weights = clf.feature_count_[i]
    class_id = clf.classes_[i]
    for j in xrange(len(weights)):
        if weights[j] > 0:
            word = words[j]
            classesdict[class_id].append((word, weights[j]))
    classesdict[class_id] = sorted(classesdict[class_id], key=lambda x: x[1],
                                   reverse=True)
with codecs.open('weights_of_classes.json', encoding='utf-8', mode='w') as f:
    for class_id in classesdict:
        print(class_id, file=f)
        for word, weight in classesdict[class_id]:
            print('\t%s:%f' % (word, weight), file=f)
