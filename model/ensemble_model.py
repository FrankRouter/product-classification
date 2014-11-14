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
from scipy.stats import entropy

# vectorizer
vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2),
                             min_df=5, max_df=0.5)
train = pd.read_csv('train.csv', dtype=object)
test = pd.read_csv('test.csv', dtype=object)
corpus = train['prodname'] + train['navigation'] \
    + train['merchant'] + train['brand']
X1 = vectorizer.fit_transform(corpus.values)
X2 = vectorizer.transform(train['prodname'].values)
y = train['categoryid'].values

# train
clf1 = MultinomialNB()
clf1.fit(X1, y)
clf2 = MultinomialNB()
clf2.fit(X2, y)

# test
y_true = test['categoryid'].values
X1_test = vectorizer.transform(test['prodname'] + test['navigation'] +
                               test['merchant'] + test['brand'])
jll_1 = clf1.predict_proba(X1_test)  # joint likelihood
y1_pred = clf1.classes_[np.argmax(jll_1, axis=1)]
proba_1 = np.amax(jll_1, axis=1)
entropy_1 = entropy(jll_1.T)

X2_test = vectorizer.transform(test['prodname'])
jll_2 = clf2.predict_proba(X2_test)  # joint likelihood
y2_pred = clf2.classes_[np.argmax(jll_2, axis=1)]
proba_2 = np.amax(jll_2, axis=1)
entropy_2 = entropy(jll_2.T)

y_final_pred = np.where(entropy_1 < entropy_2, y1_pred, y2_pred)

with open('report1.txt', 'w') as f:
    print(metrics.classification_report(y_true, y1_pred), file=f)
with open('report2.txt', 'w') as f:
    print(metrics.classification_report(y_true, y2_pred), file=f)
with open('report3.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_final_pred), file=f)

# # write weights of words to json
# wordsdict = defaultdict(dict)
# words = vectorizer.get_feature_names()
# for i in xrange(len(clf.feature_count_)):
#     weights = clf.feature_count_[i]
#     class_id = clf.classes_[i]
#     for j in xrange(len(weights)):
#         if weights[j] > 0:
#             word = words[j]
#             wordsdict[word][class_id] = weights[j]
# with codecs.open('weights_of_words.json', encoding='utf-8', mode='w') as f:
#     json.dump(obj=wordsdict, fp=f, ensure_ascii=False, encoding='utf-8',
#               indent=4, separators=(',', ': '))

# classesdict = defaultdict(list)
# for i in xrange(len(clf.feature_count_)):
#     weights = clf.feature_count_[i]
#     class_id = clf.classes_[i]
#     for j in xrange(len(weights)):
#         if weights[j] > 0:
#             word = words[j]
#             classesdict[class_id].append((word, weights[j]))
#     classesdict[class_id] = sorted(classesdict[class_id], key=lambda x: x[1],
#                                    reverse=True)
# with codecs.open('weights_of_classes.json', encoding='utf-8', mode='w') as f:
#     for class_id in classesdict:
#         print(class_id, file=f)
#         for word, weight in classesdict[class_id]:
#             print('\t%s:%f' % (word, weight), file=f)
