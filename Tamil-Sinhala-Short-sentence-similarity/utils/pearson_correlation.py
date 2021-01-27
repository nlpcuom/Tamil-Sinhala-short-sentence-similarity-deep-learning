#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : pearson_correlation.py.py
@Author : Nilaxan Satgunanantham
@IndexNo : 179337B
@Email : nilaxan.17@cse.mrt.ac.lk
@Time    : 1/26/2021 2:47 PM
@Desc:
"""

import keras.backend as K


def pearson_correlation(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)
