#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense

from embeddings.fasttext.make_fasttext_embeddings import si_fasttext_embeddings
from utils.split_and_zero_padding import split_and_zero_padding
from utils.distances.manhattan import ManDist


# File paths
TRAIN_CSV = "data\\sinhala\\train.csv"

# Pretrained word embedding model
path = "C:\\Users\\Nilaxan\\Documents\\GitHub\\pre-trained-word-embedding-models\\fastText\\facebook\\cc.si.300.vec"

# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['sentence1', 'sentence2']:
    train_df[q + '_n'] = train_df[q]

# Make FastText embeddings
embedding_dim = 300
max_seq_length = 20
use_fasttext = True

train_df, embeddings = si_fasttext_embeddings(train_df, path=path, embedding_dim=embedding_dim,
                                              empty_fasttext=not use_fasttext)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

##print(max(train_df.sentence1.map(lambda x: len(x)).max(), train_df.sentence2.map(lambda x: len(x)).max()))

X = train_df[['sentence1_n', 'sentence2_n']]
Y = np.where(train_df['manual_similarity'] >= 0.3, 1, 0)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train
Y_validation = Y_validation

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# Model variables
gpus = 1
# batch_size = 1024 * gpus
learning_rate = 0.001
batch_size = 16
n_epoch = 50
n_hidden = 50

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

# LSTM
x.add(LSTM(n_hidden))
x.add(Dense(1, activation="sigmoid"))

shared_model = x

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
model.summary()
shared_model.summary()

# Start trainings
training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

model.save('data\\sinhala\\SiameseMaLSTM_Sinhala_Sentences.h5')

# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('images\\history-graph-sinhala.png')

print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
print("Done.")
