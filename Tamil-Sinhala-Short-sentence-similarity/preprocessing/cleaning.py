#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Name : cleaning.py
@Author : Nilaxan Satkunanantham
@Email : nilaxan.17@cse.mrt.ac.lk
@Time    : 1/19/2021 7:48 PM
@Desc:
"""

import re


def text_to_word_list_english(text):
    # Pre process and convert texts to a list of words in English
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def text_to_word_list_tamil(text):
    # Pre process and convert texts to a list of words in Tamil
    text = str(text)

    # Clean the Tamil short sentences

    # Regex for remove English alphabet & numbers from text
    text = re.sub(r"[A-Za-z0-9]", " ", text)
    # Regex for remove \^,!.\\\/'+-=? characters from text
    text = re.sub(r"[\^,!.\\\/'+-=?]", " ", text)
    # Regex for remove #$%&()*:;<>@_`{|}~\t\n characters from text
    text = re.sub(r"[#$%&()*:;<>@_`{|}~\t\n]", " ", text)
    # Regex for remove ][ characters from text
    text = re.sub(r"[\[\]]", " ", text)

    '''
    Tamil Letter || unicode || Corresponding Number
    ௦ | 0BE6 | zero
    ௧ | 0BE7 | one
    ௨ | 0BE8 | two
    ௩ | 0BE9 | three
    ௪ | 0BEA | four
    ௫ | 0BEB | five
    ௬ | 0BEC | six
    ௭ | 0BED | seven
    ௮ | 0BEE | eight
    ௯ | 0BEF | nine
    ௰ | 0BF0 | ten
    ௱ | 0BF1 | hundred
    ௲ | 0BF2 | thousand
    ௳ | 0BF3 | day
    ௴ | 0BF4 | month
    ௵ | 0BF5 | year
    ௶ | 0BF6 | debit
    ௷ | 0BF7 | credit
    ௸ | 0BF8 | as above
    ௹ | 0BF9 | rupee
    ௺ | 0BFA | numeral
    '''
    text = re.sub(r"\u0BE6", "", text)
    text = re.sub(r"\u0BE7", "", text)
    text = re.sub(r"\u0BE8", "", text)
    text = re.sub(r"\u0BE9", "", text)
    text = re.sub(r"\u0BEA", "", text)
    text = re.sub(r"\u0BEB", "", text)
    text = re.sub(r"\u0BEC", "", text)
    text = re.sub(r"\u0BED", "", text)
    text = re.sub(r"\u0BEE", "", text)
    text = re.sub(r"\u0BEF", "", text)
    text = re.sub(r"\u0BF0", "", text)
    text = re.sub(r"\u0BF1", "", text)
    text = re.sub(r"\u0BF2", "", text)
    text = re.sub(r"\u0BF3", "", text)
    text = re.sub(r"\u0BF4", "", text)
    text = re.sub(r"\u0BF5", "", text)
    text = re.sub(r"\u0BF6", "", text)
    text = re.sub(r"\u0BF7", "", text)
    text = re.sub(r"\u0BF8", "", text)
    text = re.sub(r"\u0BF9", "", text)
    text = re.sub(r"\u0BFA", "", text)

    text = text.split()

    return text


def text_to_word_list_sinhala(text):
    # Pre process and convert texts to a list of words in Sinhala
    text = str(text)

    # Clean the Sinhala short sentences

    # Regex for remove English alphabet & numbers from text
    text = re.sub(r"[A-Za-z0-9]", " ", text)
    # Regex for remove \^,!.\\\/'+-=? characters from text
    text = re.sub(r"[\^,!.\\\/'+-=?]", " ", text)
    # Regex for remove #$%&()*:;<>@_`{|}~\t\n characters from text
    text = re.sub(r"[#$%&()*:;<>@_`{|}~\t\n]", " ", text)
    # Regex for remove ][ characters from text
    text = re.sub(r"[\[\]]", " ", text)

    '''
    Sinhala Letter || unicode || Corresponding Number
    ෦ | 0DE6 | zero
    ෧ | 0DE7 | one
    ෨ | 0DE8 | two
    ෩ | 0DE9 | three
    ෪ | 0DEA | four
    ෫ | 0DEB | five
    ෬ | 0DEC | six
    ෭ | 0DED | seven
    ෮ | 0DEE | eight
    ෯ | 0DEF | nine
    ෴ | 0DF4 | Sinhala Punctuation Kunddaliya
    '''
    text = re.sub(r"\u0DE6", "", text)
    text = re.sub(r"\u0DE7", "", text)
    text = re.sub(r"\u0DE8", "", text)
    text = re.sub(r"\u0DE9", "", text)
    text = re.sub(r"\u0DEA", "", text)
    text = re.sub(r"\u0DEB", "", text)
    text = re.sub(r"\u0DEC", "", text)
    text = re.sub(r"\u0DED", "", text)
    text = re.sub(r"\u0DEE", "", text)
    text = re.sub(r"\u0DEF", "", text)
    text = re.sub(r"\u0DF4", "", text)

    text = text.split()

    return text
