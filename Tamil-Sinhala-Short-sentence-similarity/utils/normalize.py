#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : normalize.py
@Author : Nilaxan Satgunanantham
@IndexNo : 179337B
@Email : nilaxan.17@cse.mrt.ac.lk
@Time    : 1/26/2021 7:54 PM
@Desc:
"""


def normalize(df, feature_names):
    result = df.copy()
    for feature_name in feature_names:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
