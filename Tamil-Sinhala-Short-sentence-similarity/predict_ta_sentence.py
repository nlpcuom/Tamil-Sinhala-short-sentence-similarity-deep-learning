#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf

from embeddings.fasttext.make_fasttext_embeddings import ta_fasttext_embeddings
from utils.split_and_zero_padding import split_and_zero_padding
from utils.distances.manhattan import ManDist

# File paths
TEST_CSV = 'data\\tamil\\test.csv'

# Pretrained word embedding model
path = "C:\\Users\\Nilaxan\\Documents\\GitHub\\pre-trained-word-embedding-models\\fastText\\facebook\\cc.ta.300.vec"

# Load training set
test_df = pd.read_csv(TEST_CSV)
for q in ['sentence1', 'sentence2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = ta_fasttext_embeddings(test_df, path=path, embedding_dim=embedding_dim, empty_fasttext=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

model = tf.keras.models.load_model('data\\tamil\\SiameseMaLSTM_Tamil_Sentences.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)
