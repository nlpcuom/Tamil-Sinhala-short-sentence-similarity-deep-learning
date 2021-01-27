#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd
import scipy
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embeddings.fasttext.make_fasttext_embeddings import si_fasttext_embeddings
from utils.split_and_zero_padding import split_and_zero_padding
from utils.distances.manhattan import ManDist

# File paths
TEST_CSV = 'data\\sinhala\\test-v1.csv'

# Pretrained word embedding model
path = "C:\\Users\\Nilaxan\\Documents\\GitHub\\pre-trained-word-embedding-models\\fastText\\facebook\\cc.si.300.vec"

# Load training set
test_df = pd.read_csv(TEST_CSV)
for q in ['sentence1', 'sentence2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = si_fasttext_embeddings(test_df, path=path, embedding_dim=embedding_dim, empty_fasttext=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

model = tf.keras.models.load_model('data\\sinhala\\SiameseMaLSTM_Sinhala_Sentences.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)
print('\n')
print("===========Ture-Values===========")
print(np.array(test_df['manual_similarity']))
sims = np.array(test_df['manual_similarity'])
print('\n')
print("===========Predict-Values===========")
print(np.round(np.array(prediction), decimals=2).flatten())
predicted_sims = np.round(np.array(prediction), decimals=2).flatten()
mean_squared_error = mean_squared_error(predicted_sims, sims)
pearson_correlation = scipy.stats.pearsonr(sims, predicted_sims)[0]
text_string = 'MSE=%.3f\nPearson Correlation=%.3f' % (mean_squared_error, pearson_correlation)
print('\n')
print('pearson correlation coefficient: ', pearson_correlation)
print('mean squared error: ', mean_squared_error)
print('\n')
print(text_string)

# Plot accuracy
plt.plot(predicted_sims)
plt.plot(sims)
plt.title('Pearson correlation coefficient')
plt.ylabel('Similarity score')
plt.xlabel('Number of sentences')
plt.legend(['Predicted', 'True'], loc='lower right')
plt.text(20, 0.2, text_string)

plt.savefig('images\\pearsonr-graph-sinhala.png')
