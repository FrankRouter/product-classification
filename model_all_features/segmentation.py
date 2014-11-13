#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import jieba

jieba.load_userdict('productnames.dict')
with codecs.open('data_with_no_punctuations.csv', mode='r',
                 encoding='utf-8') as fr:
    with codecs.open('data_segmented.csv', mode='w', encoding='utf-8') as fw:
        for line in fr:
            row = line.strip().split('\t')
            try:
                s = ' '.join(row[2:4])
                s = jieba.cut(s)
                s = (' '.join(s)).strip()
                print('%s\t%s\t%s' % (row[0], row[1], s), file=fw)
            except:
                pass
