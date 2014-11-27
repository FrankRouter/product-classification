#! /usr/bin/env python
# -*- coding: utf-8 -*-
# train multinomial naive bayes model based
# then search decision boundary for each category

from __future__ import print_function
from __future__ import division

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
import codecs
import os
from datetime import datetime
from sklearn.externals import joblib
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='train dataset', type=str,
                    default='train.csv')
parser.add_argument('-c', '--crossvalidate', help='cross validation dataset',
                    type=str, default='cv.csv')
parser.add_argument('-s', '--search', help='search decision boundary turned on',
                    action='store_true')
parser.add_argument('-p', '--persistence', action='store_true',
                    help='persistence model in human readable format')
args = parser.parse_args()

print('*' * 80)
print('Loading data...')
print('\t%s' % datetime.now())
train = pd.read_csv(args.train, dtype=object)
cv = pd.read_csv(args.crossvalidate, dtype=object)

print('*' * 80)
print('Vectorizing...')
print('\t%s' % datetime.now())
vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2),
                             min_df=10, max_df=0.5)
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

# cv
print('*' * 80)
print('cross validating: ')
print('\t%s' % datetime.now())
X_cv = vectorizer.transform(cv['prodname'] + cv['navigation'] +
                            cv['merchant'] + cv['brand'])
y_true = cv['categoryid'].values
jll = clf.predict_proba(X_cv)  # joint likelihood
y_pred = clf.classes_[np.nanargmax(jll, axis=1)]
max_proba = np.nanmax(jll, axis=1)


# trade off between acurry and recall
# search best decision boundry for each category
def search():
    print('*' * 80)
    print('Searching: ')
    boundary_of_category = dict()
    max_p_category = np.nanmax(jll, axis=0)  # max probability in each category
    min_p_category = np.nanmin(jll, axis=0)  # min probability in each category
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
        idx_max_f1 = np.nanargmax(f1)
        boundary_of_category[categoryid] = threshold[idx_max_f1]
        y_pred[(max_proba < threshold[idx_max_f1])
               & (y_pred == categoryid)] = None
    if args.persistence:
        with codecs.open('boundary.json', encoding='utf-8', mode='w') as f:
            json.dump(obj=boundary_of_category, fp=f, ensure_ascii=False,
                      encoding='utf-8', indent=4, separators=(',', ': '))

if args.search:
    search()

with open('report.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_pred), file=f)

# model persistence in binary
if not os.path.isdir('bin'):
    os.mkdir('bin')
joblib.dump(vectorizer, 'bin/tfidf')
joblib.dump(clf, 'bin/classifier')

if args.persistence:
    print('*' * 80)
    print('Outputting model in human readable format')
    output_dir = 'log_proba'
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)
    words = vectorizer.get_feature_names()
    for i in xrange(len(clf.feature_log_prob_)):
        pairs = []
        log_proba = clf.feature_log_prob_[i]
        class_id = clf.classes_[i]
        for j in xrange(len(log_proba)):
            word = words[j]
            pairs.append((word, log_proba[j]))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        output_path = os.path.join(output_dir, '%s.txt' % class_id)
        with codecs.open(output_path, encoding='utf-8', mode='w') as f:
            for word, weight in pairs:
                print('\t%s:%f' % (word, weight), file=f)

        if (i + 1) % 10 == 0:
            print('\t%s\t%d categories writen' % (datetime.now(), (i + 1)))
    if (i + 1) % 10 != 0:
        print('\t%s\t%d categories writen' % (datetime.now(), (i + 1)))

print('*' * 80)
print('Finish')
print('\t%s' % datetime.now())
