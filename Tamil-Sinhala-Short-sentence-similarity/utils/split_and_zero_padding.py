#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import itertools


def split_and_zero_padding(df, max_seq_length):
    # Split to dicts
    split = {'left': df['sentence1_n'], 'right': df['sentence2_n']}

    # Zero padding
    for dataset, side in itertools.product([split], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset
