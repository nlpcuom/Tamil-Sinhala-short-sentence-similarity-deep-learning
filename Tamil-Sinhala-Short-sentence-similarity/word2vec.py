#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import logging
import pandas as pd

from utils import extract_sentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_dataset = "C:\\Users\\Nilaxan\\Documents\\GitHub\Siamese-LSTM\\data\\train_new.csv"
test_dataset = "C:\\Users\\Nilaxan\\Documents\\GitHub\Siamese-LSTM\\data\\test_new.csv"

documents = list(extract_sentences(train_dataset, test_dataset))
logging.info("Reading completed from dataset files")

model = gensim.models.Word2Vec(documents, size=300)
model.train(documents, total_examples=len(documents), epochs=10)
model.save("C:\\Users\\Nilaxan\\Documents\\project\\Siamese-LSTM\\data\\short-sentence-pairs.w2v")

'''
def extract_tamil_sentences():
    """
    Extract Tamil sentences for making fastText model.
    """
    df1 = pd.read_csv("C:\\Users\\Nilaxan\\Documents\\GitHub\\Siamese-LSTM\\data\\tamil\\train.csv")
    df2 = pd.read_csv("C:\\Users\\Nilaxan\\Documents\\GitHub\\Siamese-LSTM\\data\\tamil\\test.csv")

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['sentence1']:
                yield gensim.utils.simple_preprocess(row['sentence1'])
            if row['sentence2']:
                yield gensim.utils.simple_preprocess(row['sentence2'])


tamil_documents = list(extract_tamil_sentences())
logging.info("Done reading Tamil dataset file")

model_tamil = gensim.models.Word2Vec(tamil_documents, size=300)
model_tamil.train(tamil_documents, total_examples=len(tamil_documents), epochs=10)
model_tamil.save("C:\\Users\\Nilaxan\\Documents\\GitHub\\Siamese-LSTM\\data\\tamil\\SiameseMaLSTM_Tamil_Sentences.w2v")
'''
## ---

'''
def extract_sinhala_questions():
    """
    Extract Sinhala sentences for making fastText model.
    """
    df1 = pd.read_csv("C:\\Users\\Nilaxan\\Documents\\project\Siamese-LSTM\\data\\train.csv")
    df2 = pd.read_csv("C:\\Users\\Nilaxan\\Documents\\project\Siamese-LSTM\\data\\test.csv")

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['sentence1']:
                yield gensim.utils.simple_preprocess(row['sentence1'])
            if row['sentence2']:
                yield gensim.utils.simple_preprocess(row['sentence2'])


sinhala_documents = list(extract_sinhala_questions())
logging.info("Done reading Sinhala dataset file")

model_sinhala = gensim.models.fasttext(documents, size=300)
model_sinhala.train(sinhala_documents, total_examples=len(sinhala_documents), epochs=10)
model_sinhala.save("C:\\Users\\Nilaxan\\Documents\\project\\Siamese-LSTM\\data\\sinhala_short_sentence_pairs.w2v")
'''
