#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs

remove_chars = u'''`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?''' + \
    u'''·！@#￥%……&*（）——【】、『』|；‘’：“”，。《》？'''
trans_table = dict([(ord(c), None) for c in remove_chars])
with codecs.open('data_with_no_punctuations.csv', 'w', encoding='utf-8') as new:
    with codecs.open('data.csv', encoding='utf-8') as f:
        for line in f:
            # maybe more than 3 comma
            row = line.strip().split(',', 3)
            try:
                s = u' '.join(row[2:])
                s = s.translate(trans_table)
                print('%s\t%s\t%s' % (row[0], row[1], s), file=new)
            except:
                pass
