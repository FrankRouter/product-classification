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

# vectorizer
tv = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2), min_df=5,
                     max_df=0.5)
train = pd.read_csv('train.csv', dtype=object)
test = pd.read_csv('test.csv', dtype=object)
X = tv.fit_transform(train['info'].values)
y = train['categoryid'].values

# train
clf = MultinomialNB()
clf.fit(X, y)

# test
X_test = tv.transform(test['info'])
y_true = test['categoryid'].values
jll = clf.predict_proba(X_test)  # joint likelihood
y_pred = clf.classes_[np.argmax(jll, axis=1)]
max_proba = np.amax(jll, axis=1)

# trade off between acurry and recall
# search best decision boundry in each category
# categoryid_set = set(train['categoryid'].values)
# boundry_of_category = dict()
# for categoryid in categoryid_set:
#     max_f1 = .0
#     decision_boundary = .0
#     for threshold in np.arange(0, 0.5, 0.05):
#         tp = (y_true == categoryid) & (y_pred == categoryid) \
#             & (max_proba >= threshold)
#         fp = (y_true != categoryid) & (y_pred == categoryid) \
#             & (max_proba >= threshold)
#         fn = (y_true == categoryid) \
#             & ((y_pred != categoryid) | (max_proba < threshold))
#         accuracy = sum(tp) / (sum(tp) + sum(fp))
#         recall = sum(tp) / (sum(tp) + sum(fn))
#         f1 = 2 * accuracy * recall / (accuracy + recall)
#         if f1 > max_f1:
#             max_f1 = f1
#             decision_boundary = threshold
#     boundry_of_category[categoryid] = decision_boundary
#     y_pred[(max_proba < decision_boundary) & (y_pred == categoryid)] = None

with open('report.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_pred), file=f)

# write weights of words to json
wordsdict = defaultdict(dict)
words = tv.get_feature_names()
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
