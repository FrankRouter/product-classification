#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs
import jieba
from datetime import datetime

jieba.load_userdict('productnames.dict')
remove_chars = u'''`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?''' + \
    u'''·！@#￥%……&*（）——【】、『』|；‘’：“”，。《》？'''
trans_table = dict([(ord(c), None) for c in remove_chars])
print('*' * 80)
print('Preprocessing...')
with codecs.open('data_preprocessed.csv', 'w', encoding='utf-8') as out,\
        codecs.open('data.csv', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # should be 6 columns
        cells = line.strip().split(',', 5)
        try:
            # remove punctuation
            cleaned = [cell.translate(trans_table) for cell in cells[2:]]
            # chinese segment
            segmented = [u' '.join(jieba.cut(cell)) for cell in cleaned]
            cells = cells[:2] + segmented
            print(u','.join(cells), file=out)
        except:
            pass

        if (i + 1) % 10000 == 0:
            print('\t%s\t%d lines processed' % (datetime.now(), (i + 1)))
    if (i + 1) % 10000 != 0:
        print('\t%s\t%d lines processed' % (datetime.now(), (i + 1)))
