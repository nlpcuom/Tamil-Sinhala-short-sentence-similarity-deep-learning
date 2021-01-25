#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : make_ta_word2vec_embeddings.py
@Author : Nilaxan Satgunanantham
@IndexNo : 179337B
@Email : nilaxan.17@cse.mrt.ac.lk
@Time    : 1/19/2021 8:05 PM
@Desc:
"""

from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np

from preprocessing.cleaning import text_to_word_list_sinhala
from preprocessing.cleaning import text_to_word_list_tamil
from preprocessing.cleaning import text_to_word_list_english

def en_word2vec_embeddings(df, path, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Remove Tamil Stopwords
    stops = set(stopwords.words('english'))

    # Load word2vec
    print("Loading word2vec model(it may takes 5-10 minutes) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format(path, encoding='utf8', binary=False)

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for question in ['question1', 'question2']:

            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list_english(row[question]):
                # Check for stopped words
                if word in stops:
                    continue

                # Check if a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append question as number representation
            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def ta_word2vec_embeddings(df, path, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Remove Tamil Stopwords
    stops = set(stopwords.words('tamil'))

    # Load word2vec
    print("Loading word2vec model(it may takes 5-10 minutes) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format(path, encoding='utf8', binary=False)

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for sentence in ['sentence1', 'sentence2']:

            q2n = []  # q2n -> sentence numbers representation
            for word in text_to_word_list_tamil(row[sentence]):
                # Check for stopped words
                if word in stops:
                    continue

                # Check if a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append question as number representation
            df.at[index, sentence + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def si_word2vec_embeddings(df, path, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Remove Sinhala Stopwords
    stops = set(stopwords.words('sinhala'))

    # Load word2vec
    print("Loading word2vec model(it may takes 5-10 minutes) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format(path, encoding='utf8', binary=False)

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for sentence in ['sentence1', 'sentence2']:

            q2n = []  # q2n -> sentence numbers representation
            for word in text_to_word_list_sinhala(row[sentence]):
                # Check for stopped words
                if word in stops:
                    continue

                # Check if a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append question as number representation
            df.at[index, sentence + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


class EmptyWord2Vec:
    """
    For test use.
    """
    vocab = {}
    word_vec = {}
