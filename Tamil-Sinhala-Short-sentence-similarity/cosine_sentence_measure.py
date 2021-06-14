#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : cosine_sentence_measure.py
@Author : Nilaxan Satgunanantham
@IndexNo : 179337B
@Email : nilaxan.17@cse.mrt.ac.lk
@Time    : 5/23/2021 8:06 AM
@Desc:
"""

import math
import re
from collections import Counter
import pandas as pd
import numpy as np
import scipy.stats

WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


# File paths
TEST_CSV = 'data\\tamil\\test-v2.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV)

predicted = []

for i in range(250):
    cosine = get_cosine(text_to_vector(test_df['sentence1'][i]), text_to_vector(test_df['sentence2'][i]))
    predicted.insert(i,cosine)
    print("Cosine:", cosine)

sims = np.array(test_df['manual_similarity'])
predicted_sims = pd.DataFrame(predicted)
predicted_sims = np.round(np.array(predicted_sims), decimals=2).flatten()
pearson_correlation = scipy.stats.pearsonr(sims, predicted_sims)[0]

print('\n')
print('pearson correlation coefficient: ', pearson_correlation)
