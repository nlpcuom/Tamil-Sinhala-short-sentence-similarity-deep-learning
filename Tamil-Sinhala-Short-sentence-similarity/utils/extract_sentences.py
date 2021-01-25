#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : extract_sentences.py
@Author : Nilaxan Satgunanantham
@IndexNo : 179337B
@Email : nilaxan.17@cse.mrt.ac.lk
Time    : 1/20/2021 7:39 AM
Desc:
"""

import gensim
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def extract_sentences(train, test):
    """
    Extract sentences for making word2vec model.
    """
    df1 = pd.read_csv(train)
    df2 = pd.read_csv(test)

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['sentence1']:
                yield gensim.utils.simple_preprocess(row['sentence1'])
            if row['sentence2']:
                yield gensim.utils.simple_preprocess(row['sentence2'])
